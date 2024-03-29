import pickle
import re
import polars as pl
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import skew
import polars as pl
from sklearn import metrics, model_selection
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb
import datetime
import Levenshtein

from utils.utils import Logger, WriteSheet
from utils.utils import class_vars_to_dict
from utils.utils import add_random_feature

logger = Logger()

class ENV:
    input_dir = "/kaggle/input/"
    feature_dir = "./"
    model_dir = "./"
    output_dir = "./"
    commit_hash = ""
    save_to_sheet = True
    sheet_json_key = '/kaggle/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
    sheet_key = '1oDrxAG5aWZ4hkLVHlVwBzjaHo8fmr3pgASUxljFHb78'
    on_kaggle = True

class RCFG:
    run_name = 'exp003'
    debug = True
    debug_size = 100
    split_cnt = 5
    n_splits = 5
    cnt_seed = 3
    base_seed = 42
    preprocess_train = False
    predict = False
    load_model = False
    select_feature = True
    use_feature_rank = 500
    use_random_features = False
    threshold_random_features = 15
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse", 
        "random_state": 42, 
        "n_estimators": 12001, 
        "verbosity": -1, 
        "reg_lambda": 0.1, 
        "colsample_bytree": 0.8, 
        "subsample": 0.75, 
        "learning_rate": 0.01,
        "num_leaves": 19,
        "max_depth": 5, 
        "min_child_samples": 22
    }
    fix_data = False


activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete']
text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
special_char_to_text = {
    ' ': 'space', '\n': 'enter', '.': 'period', ',': 'comma', "'": 'apostrophe', '"': 'quotation',
    '-': 'hyphen', ';': 'semicolon', ':': 'colon', '?': 'question', '!': 'exclamation', '<': 'less_than',
    '>': 'greater_than', '/': 'slash', '\\': 'backslash', '@': 'at', '#': 'hash', '$': 'dollar',
    '%': 'percent', '^': 'caret', '&': 'ampersand', '*': 'asterisk', '(': 'left_parenthesis',
    ')': 'right_parenthesis', '_': 'underscore',
}


def count_by_values(df, colname, values, suffix=""):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for value in values:
        name = value
        if value in special_char_to_text:
            name = special_char_to_text[value]
        tmp_df = df.group_by('id').agg(pl.col(colname).is_in([value]).sum().alias(f'{colname}_{name}_cnt{suffix}'))
        fts  = fts.join(tmp_df, on='id', how='left') 
    return fts


def fix_data(df):
    
    logger.info('Start fix data.')

    # 長すぎるprocessを除外する
    cond_long_process = (df['down_event']=='Process') & (df['action_time']>=1000)
    logger.info(f'Remove long process: {sum(cond_long_process)}')
    df = df[~cond_long_process].copy()

    # 準備
    df['up_time_shift1'] = df.groupby('id')['up_time'].shift(1)
    df['down_time_shift1'] = df.groupby('id')['down_time'].shift(1)
    df['next_action_time_diff'] = (df['down_time'] - df['up_time_shift1'])
    df['down_time_diff'] = (df['down_time'] - df['down_time_shift1'])

    # down_timeが1秒以上逆転しているときは補正する
    df_tmp = df[df['down_time_diff'] < -1 * 1000]
    target_list = df_tmp[['id', 'event_id', 'down_time_diff']].to_dict(orient='records')
    logger.info(f'Fix down time reversal. {len(target_list)}')
    for data in target_list:
        idx = (df['id']==data['id']) & (df['event_id']>=data['event_id'])
        df.loc[idx, 'down_time'] += abs(data['down_time_diff'])
        df.loc[idx, 'up_time'] += abs(data['down_time_diff'])

    # down_time_diffが20分以上ある時は10秒に補正する
    df_tmp = df[df['down_time_diff'] > 20 * 60 * 1000]
    target_list = df_tmp[['id', 'event_id', 'down_time_diff']].to_dict(orient='records')
    logger.info(f'Fix too long interval. {len(target_list)}')
    for data in target_list:
        adjust = abs(data['down_time_diff']) - 10 * 1000
        idx = (df['id']==data['id']) & (df['event_id']>=data['event_id'])
        df.loc[idx, 'down_time'] -= adjust
        df.loc[idx, 'up_time'] -= adjust
    
    # todo: 最初の時点から20分以上空いているときは2分に補正する
    df_tmp = df[(df['event_id']==1)&(df['down_time']>=20* 60 * 1000)]
    target_list = df_tmp[['id', 'event_id', 'down_time']].to_dict(orient='records')
    logger.info(f'Fix too late start. {len(target_list)}')
    for data in target_list:
        adjust = data['down_time'] - 2 * 60 * 1000
        idx = (df['id']==data['id']) & (df['event_id']>=data['event_id'])
        df.loc[idx, 'down_time'] -= adjust
        df.loc[idx, 'up_time'] -= adjust
    
    return df
        


def dev_feats(df):

    logger.info("Count by values features")
    feats = count_by_values(df, 'activity', activities)
    feats = feats.join(count_by_values(df, 'text_change', text_changes), on='id', how='left') 
    feats = feats.join(count_by_values(df, 'down_event', events), on='id', how='left') 
    feats = feats.with_columns(
        down_event_ArrowUpDown_cnt=pl.col('down_event_ArrowUp_cnt') + pl.col('down_event_ArrowDown_cnt')
    ).drop('down_event_ArrowUp_cnt', 'down_event_ArrowDown_cnt')

    temp = df.group_by('id').agg(
        ((pl.col('activity')=='Remove/Cut') & (pl.col('text_change')==" ")).sum().alias('delete_space_cnt'),
        ((pl.col('activity')=='Remove/Cut') & (pl.col('text_change')==",")).sum().alias('delete_comma_cnt'),
        (pl.col('down_time').filter(pl.col('activity')=='Input').min().alias('first_input_down_time')),
        (pl.col('down_time').filter(pl.col('activity')=='Remove/Cut').min().alias('first_remove_down_time'))
    )
    feats = feats.join(temp, on='id', how='left')

    # text_change. ないほうがCVには良いがLBには悪い
    temp = df.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
    temp = temp.group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
    temp = temp.with_columns(
        input_word_count = pl.col('text_change').list.len(),
        input_word_length_mean = pl.col('text_change').map_elements(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
        input_word_length_max = pl.col('text_change').map_elements(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
        input_word_length_std = pl.col('text_change').map_elements(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
        input_word_length_skew = pl.col('text_change').map_elements(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0))
    )
    temp = temp.drop('text_change')
    feats = feats.join(temp, on='id', how='left') 

    
    logger.info("Numerical columns features")
    temp = df.group_by("id").agg(
        pl.sum('action_time').name.suffix('_sum'), 
        pl.mean(['down_time', 'action_time', 'cursor_position', 'word_count']).name.suffix('_mean'), 
        pl.std(['down_time', 'action_time', 'cursor_position', 'word_count']).name.suffix('_std'),
        pl.median(['down_time', 'action_time', 'cursor_position', 'word_count']).name.suffix('_median'), 
        pl.min(['down_time', 'up_time']).name.suffix('_min'), 
        pl.max(['event_id', 'down_time', 'action_time', 'cursor_position', 'word_count']).name.suffix('_max'),
        pl.quantile(['action_time', 'cursor_position', 'word_count'], 0.5).name.suffix('_quantile'),
        pl.col(['cursor_position', 'word_count', 'event_id']).filter(pl.col('down_time')<=20 * 60 * 1000).max().name.suffix('_max_20min'),
        pl.col(['cursor_position', 'word_count', 'event_id']).filter(pl.col('down_time')<=25 * 60 * 1000).max().name.suffix('_max_25min')
    )
    feats = feats.join(temp, on='id', how='left') 

    logger.info("Categorical columns features")
    temp  = df.group_by("id").agg(pl.n_unique(['down_event', 'up_event', 'text_change']))
    feats = feats.join(temp, on='id', how='left') 

    logger.info("Idle time features")
    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns(((pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.group_by("id").agg(
        inter_key_largest_lantency = pl.max('time_diff'),
        inter_key_median_lantency = pl.median('time_diff'),
        mean_pause_time = pl.mean('time_diff'),
        std_pause_time = pl.std('time_diff'),
        total_pause_time = pl.sum('time_diff'),
        pauses_minus_sec = pl.col('time_diff').filter(pl.col('time_diff') < 0).count(),                                   
        pauses_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).count(),
        pauses_1_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).count(),
        pauses_1_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).count(),
        pauses_2_sec = pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).count(),
        pauses_3_sec = pl.col('time_diff').filter(pl.col('time_diff') > 3).count(),
        pauses_10_sec = pl.col('time_diff').filter(pl.col('time_diff') > 10).count()
    )
    feats = feats.join(temp, on='id', how='left') 

    temp = df.with_columns(pl.col('word_count').shift(100).over('id').alias('word_count_shift200'))
    temp = temp.with_columns((pl.col('word_count') - pl.col('word_count_shift200')).alias('word_count_gap200'))
    temp = temp.group_by("id").agg(
        pl.mean('word_count_gap200').name.suffix('_mean'), 
        pl.std('word_count_gap200').name.suffix('_std'),
        pl.max('word_count_gap200').name.suffix('_max'),
        pl.median('word_count_gap200').name.suffix('_median'),
    )
    feats = feats.join(temp, on='id', how='left')

    # 2秒以内で入力/削除したストリーム
    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('time_diff')<2)
    temp = temp.with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last_distinct()).then(pl.count()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(
        pl.mean('P-bursts').name.suffix('_mean'), 
        pl.std('P-bursts').name.suffix('_std'),
        pl.count('P-bursts').name.suffix('_count'),
        pl.median('P-bursts').name.suffix('_median'), 
        pl.max('P-bursts').name.suffix('_max'),
        pl.first('P-bursts').name.suffix('_first'), # good for LB, bad for CV
        pl.last('P-bursts').name.suffix('_last'), # good for LB, bad for CV
    )
    feats = feats.join(temp, on='id', how='left') 

    # 0.3秒以内で入力したストリーム
    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns(((pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(((pl.col('activity').is_in(['Input']))&(pl.col('time_diff')<0.3)).alias('flag'))
    temp = temp.with_columns(pl.when(pl.col("flag") & pl.col("flag").is_last_distinct()).then(pl.count()).over(pl.col("flag").rle_id()).alias('P-bursts_v2'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(
        pl.mean('P-bursts_v2').name.suffix('_mean'), 
        pl.std('P-bursts_v2').name.suffix('_std'),
        pl.count('P-bursts_v2').name.suffix('_count'),
        pl.median('P-bursts_v2').name.suffix('_median'), 
        pl.max('P-bursts_v2').name.suffix('_max'),
        pl.col('P-bursts_v2').filter(pl.col('P-bursts_v2') > 1).count().name.suffix('_count_gt_1'),
        pl.col('P-bursts_v2').filter(pl.col('P-bursts_v2') > 3).count().name.suffix('_count_gt_3')
    )
    feats = feats.join(temp, on='id', how='left') 

    # 削除のストリーム
    temp = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('activity').is_in(['Remove/Cut']))
    temp = temp.with_columns(pl.when(pl.col("activity") & pl.col("activity").is_last_distinct()).then(pl.count()).over(pl.col("activity").rle_id()).alias('R-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(
        pl.mean('R-bursts').name.suffix('_mean'), 
        pl.std('R-bursts').name.suffix('_std'), 
        pl.median('R-bursts').name.suffix('_median'), 
        pl.max('R-bursts').name.suffix('_max'),
        pl.first('R-bursts').name.suffix('_first'), # good for LB, bad for CV
        pl.last('R-bursts').name.suffix('_last'), # good for LB, bad for CV
    )
    feats = feats.join(temp, on='id', how='left')


    feats = feats.with_columns(
        (pl.col('event_id_max_25min') / pl.col('event_id_max')).alias('event_id_max_25min_rate'),
        (pl.col('event_id_max_20min') / pl.col('event_id_max')).alias('event_id_max_20min_rate'),
        (pl.col('cursor_position_max_25min') / pl.col('cursor_position_max')).alias('cursor_position_max_25min_rate'),
        (pl.col('cursor_position_max_20min') / pl.col('cursor_position_max')).alias('cursor_position_max_20min_rate'),
        (pl.col('word_count_max_25min') / pl.col('word_count_max')).alias('word_count_max_25min_rate'),
        (pl.col('word_count_max_20min') / pl.col('word_count_max')).alias('word_count_max_20min_rate'),
    ).drop(
        'event_id_max_25min', 
        'event_id_max_20min', 
        'cursor_position_max_25min', 
        'cursor_position_max_20min', 
        'word_count_max_25min',
        'word_count_max_20min'
    )

    return feats


def create_shortcuts(df):

    event_df = df[['id', 'event_id', 'down_event']].copy(deep=True)
    event_df['down_event_shift_1'] = event_df['down_event'].shift(periods=1)
    event_df['down_event_shift_2'] = event_df['down_event'].shift(periods=2)
    event_df = event_df[['id', 'event_id', 'down_event_shift_2', 'down_event_shift_1', 'down_event']]

    ctrl_left_df = ((event_df['down_event_shift_1'] == 'Control') & (event_df['down_event'] == 'ArrowLeft')).groupby(event_df['id']).sum().reset_index(name='count')
    ctrl_right_df = ((event_df['down_event_shift_1'] == 'Control') & (event_df['down_event'] == 'ArrowRight')).groupby(event_df['id']).sum().reset_index(name='count')
    ctrl_shift_left_df = ((event_df['down_event_shift_2'] == 'Control') & (event_df['down_event_shift_1'] == 'Shift') & (event_df['down_event'] == 'ArrowLeft')).groupby(event_df['id']).sum().reset_index(name='count')
    ctrl_shift_right_df = ((event_df['down_event_shift_2'] == 'Control') & (event_df['down_event_shift_1'] == 'Shift') & (event_df['down_event'] == 'ArrowRight')).groupby(event_df['id']).sum().reset_index(name='count')

    kb_shortcut_df = pd.DataFrame(event_df['id'].unique(), columns=['id'])
    kb_shortcut_df['ctrl_shift_count'] = ctrl_left_df['count'] + ctrl_right_df['count'] + ctrl_shift_left_df['count'] + ctrl_shift_right_df['count']

    return kb_shortcut_df

def product_to_keys(logs, essays):
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]


def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]


def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)
def quantile10(x):
    return x.quantile(0.10)
def quantile82(x):
    return x.quantile(0.82)
def quantile90(x):
    return x.quantile(0.90)
def quantile95(x):
    return x.quantile(0.95)

def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def get_essay_df(df):
    df       = df[df.activity != 'Nonproduction']
    temp     = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id','word_len']].groupby(['id']).agg(
        ['count', 'mean', 'max', q3, 'sum', quantile82, quantile90, quantile95]
    )
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len!=0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id','sent_len']].groupby(['id']).agg(
        ['count', 'mean', 'std', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum', quantile10, quantile90, quantile95]
    ), df[['id','sent_word_count']].groupby(['id']).agg(
        ['mean', 'min', 'std', 'max', 'first', 'last', q1, 'median', q3, 'sum', quantile90]
    )], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df

def parag_feats(df):
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    df = df[df.paragraph_len!=0].reset_index(drop=True)
    
    paragraph_agg_df = pd.concat([df[['id','paragraph_len']].groupby(['id']).agg(
        ['count', 'mean', 'std', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']
    ), df[['id','paragraph_word_count']].groupby(['id']).agg(
        ['mean', 'std', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']
    )], axis=1) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def word_feats_v2(df):
    
    df_base = pd.DataFrame(df['id'].unique(), columns=['id'])
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].str.len()
    df = df[df['word_len']>0].copy()

    for i in range(1, 10):
        df[f'word_lag{i}'] = df.groupby('id')['word'].shift(i)
        df[f'word_len_lag{i}'] = df.groupby('id')['word_len'].shift(i)
    df['word_len_sum5'] = df[['word_len'] + [f'word_len_lag{i}' for i in range(1, 5)]].sum(axis=1)
    df['word_len_sum10'] = df[['word_len'] + [f'word_len_lag{i}' for i in range(1, 10)]].sum(axis=1)

    df_words = df.groupby('id').agg(
        word_feats_apostrophe_cnt = ('word', lambda x: ((x.str.endswith("'q")) & (x.str.len()<=5)).sum()),
        word_feats_long_apostrophe_cnt = ('word', lambda x: ((x.str.contains("'")) & (x.str.len()>=8)).sum()),
        # word_feats_len35_len_sum5 = ('word_len_sum5', lambda x: (x>=35).sum()),
        word_feats_median_len_sum10 = ('word_len_sum10', 'median'),
        word_feats_quantile95_len_sum5 = ('word_len_sum5', quantile95)
    ).reset_index()
    
    df_result = df_base.merge(df_words, on='id', how='left').fillna(0)

    return df_result

def sent_feats_v2(df):

    df_base = pd.DataFrame(df['id'].unique(), columns=['id'])

    df['sent'] = df['essay'].apply(lambda x: re.split('(?<=[\\.|\\?|\\!])',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    df = df[df['sent'].str.contains('q')].copy()

    df['first'] = df['sent'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else '')
    df['first_two'] = df['sent'].apply(lambda x: ' '.join(x.split()[:1]) if len(x.split()) > 1 else '')
    df['first_three'] = df['sent'].apply(lambda x: ' '.join(x.split()[:2]) if len(x.split()) > 2 else '')
    df['first_four'] = df['sent'].apply(lambda x: ' '.join(x.split()[:3]) if len(x.split()) > 3 else '')
    df['fist_eight_len'] = df['sent'].apply(lambda x: len(x) if len(x.split()) > 7 else np.nan)
    df['last'] = df['sent'].apply(lambda x: x[-1] if len(x) > 0 else '')
    df['last_lag1'] = df.groupby('id')['last'].shift(1)
    df['last_consec2'] = df['last'] + df['last_lag1']
    
    df_first = df.groupby('id').agg(
        sent_feats_first_word_long_comma = ('first', lambda x: ((x.str.len() > 6) & (x.str.endswith(','))).sum()),
        sent_feats_first_word_one_letter = ('first', lambda x: (x=='q').sum()),
        sent_feats_first_two_word_short = ('first_two', lambda x: ((x != '') & (x.str.len() <= 7)).sum()),
        sent_feats_first_three_comma = ('first_three', lambda x: ((x != '') & (x.str.endswith(','))).sum()),
        sent_feats_first_four_comma = ('first_four', lambda x: ((x != '') & (x.str.endswith(','))).sum()),
        sent_feats_first_eight_mean = ('fist_eight_len', 'mean'),
        sent_feats_after_apostroph = ('sent', lambda x: (x.str.contains(".{20,}'")).sum()),
        sent_feats_hyphen = ('sent', lambda x: (x.str.contains('-')).sum()),
        sent_feats_double_hyphen = ('sent', lambda x: (x.str.count('-')==2).sum()),
        sent_feats_long_hyphen = ('sent', lambda x: (x.str.contains('-.{30,}')).sum()),
        sent_feats_apostroph = ('sent', lambda x: (x.str.contains("'")).sum()),
        sent_feats_double_quotation = ('sent', lambda x: (x.str.contains('".{2,}"')).sum()),
        sent_feats_long_double_quotation = ('sent', lambda x: (x.str.contains('".{20,}"')).sum()),
        sent_feats_colon = ('sent', lambda x: ((x.str.contains(':')) | (x.str.contains(';'))).sum()),
        sent_feats_long_comma = ('sent', lambda x: ((x.str.contains('.{30,},')) & (x.str.contains(',.{30,}'))).sum()),
        sent_feats_double_comma = ('sent', lambda x: ((x.str.len() >= 50) & (x.str.count(",")==2)).sum()),
        sent_feats_triple_comma = ('sent', lambda x: ((x.str.len() >= 80) & (x.str.count(",")>=3)).sum()),
        sent_feats_last_question = ('last', lambda x: (x=='?').sum()),
        sent_feats_last_question_double = ('last_consec2', lambda x: (x=='??').sum()),
        sent_feats_last_exclamation = ('last', lambda x: (x=='!').sum())
    ).reset_index()

    df_first['sent_feats_first_three_four_comma'] = df_first['sent_feats_first_three_comma'] + df_first['sent_feats_first_four_comma']
    df_first = df_first.drop(['sent_feats_first_three_comma', 'sent_feats_first_four_comma'], axis=1)

    df_result = df_base.merge(df_first, on='id', how='left').fillna(0)
    
    return df_result


def essay_diff_feats(log, essay_df):

    essays_15min_df = get_essay_df(log[log['down_time']<=15*60*1000])
    essays_15min_df.rename(columns={'essay': 'essay15'}, inplace=True)
    df_total = essay_df[['id', 'essay']].merge(essays_15min_df, on='id', how='left')
    df_total['len_15min'] = df_total['essay15'].str.len().fillna(0)
    
    def edit_distance_first(x):
        l = max(int(x['len_15min']) - 100, 1)
        e, e15 = x['essay'], x['essay15']
        if type(e) != str or type(e15) != str or min(len(e), len(e15)) < l:
            return 0
        else:
            return Levenshtein.distance(e[:l], e15[:l])
    
    df_total['edit_distance_first_15min'] = df_total.apply(edit_distance_first, axis=1)
    df_total.drop(['essay', 'essay15', 'len_15min'], axis=1, inplace=True)

    return df_total


class Runner():

    def __init__(
        self,
    ):
        
        tqdm.pandas()
        logger.info(f'Commit hash: {ENV.commit_hash}')
        if ENV.save_to_sheet:
            logger.info('Initializing Google Sheet.')
            self.sheet = WriteSheet(
                sheet_json_key = ENV.sheet_json_key,
                sheet_key = ENV.sheet_key
            )

        if ENV.on_kaggle:
            from kaggle_secrets import UserSecretsClient
            self.user_secrets = UserSecretsClient()

    def load_dataset(self,):

        self.train_logs = pd.read_csv(f'{ENV.input_dir}linking-writing-processes-to-writing-quality/train_logs.csv')
        self.train_scores = pd.read_csv(f'{ENV.input_dir}linking-writing-processes-to-writing-quality/train_scores.csv')
        self.test_logs = pd.read_csv(f'{ENV.input_dir}linking-writing-processes-to-writing-quality/test_logs.csv')
        self.ss_df = pd.read_csv(f'{ENV.input_dir}linking-writing-processes-to-writing-quality/sample_submission.csv')

        self.train_essays = pd.read_csv(f'{ENV.input_dir}/writing-quality-challenge-constructed-essays/train_essays_fast.csv')

        if RCFG.debug:
            logger.info(f'Debug mode. Get only first {RCFG.debug_size} ids.')
            target_id = self.train_logs['id'].unique()[:RCFG.debug_size]
            self.train_logs = self.train_logs[self.train_logs['id'].isin(target_id)]
    
    def _add_features(self, df, mode='train'):

        feats = dev_feats(pl.from_pandas(df))
        feats = feats.to_pandas()
    
        logger.info('Add simple essay features.')
        essays = get_essay_df(df)
        feats = feats.merge(word_feats(essays), on='id', how='left')
        feats = feats.merge(sent_feats(essays), on='id', how='left')
        feats = feats.merge(parag_feats(essays), on='id', how='left')
        feats = feats.merge(word_feats_v2(essays), on='id', how='left')
        feats = feats.merge(sent_feats_v2(essays), on='id', how='left')

        logger.info('Add features based on comparison of essays at different timestamps.')
        feats = feats.merge(essay_diff_feats(df, essays), on='id', how='left')

        logger.info('Add other features.')
        feats = feats.merge(get_keys_pressed_per_second(df), on='id', how='left')
        feats = feats.merge(product_to_keys(df, essays), on='id', how='left')
        feats = feats.merge(create_shortcuts(df), on='id', how='left')

        feats['comma_sent_len_rate'] = feats['down_event_comma_cnt'] / feats['sent_len_sum']
        feats['sent_feats_first_two_word_short_rate'] = feats['sent_feats_first_two_word_short'] / feats['sent_count']
        feats = feats.drop(['sent_feats_first_two_word_short'], axis=1)

        return feats


    def preprocess(self,):

        logger.info('Start preprocessing.')
        
        if RCFG.preprocess_train:
            logger.info('Preprocess train data. Create features for train data.')
            if RCFG.fix_data:  
                self.train_logs = fix_data(self.train_logs)
            train_feats = self._add_features(self.train_logs, mode='train')
            self.train_feats = train_feats.merge(self.train_scores, on='id', how='left')
            self.train_feats.to_csv(f'{ENV.output_dir}train_feats.csv', index=False)
        else:
            logger.info(f'Load train data from {ENV.feature_dir}train_feats.csv.')
            self.train_feats = pd.read_csv(f'{ENV.feature_dir}train_feats.csv')

        if RCFG.predict:        
            logger.info('Preprocess test data. Get essays of test data.')
            logger.info('Create features for test data.')
            if RCFG.fix_data:  
                self.test_logs = fix_data(self.test_logs)
            self.test_feats = self._add_features(self.test_logs, mode='test')

    def _train_fold_seed(self, mode='first', split_id=0):

        oofscore = []
        target_col = ['score']
        self.train_cols = [col for col in self.train_feats.columns if col not in ['score', 'id', 'fold']]
        params = RCFG.lgbm_params.copy()
        last = False if mode == 'first' and RCFG.select_feature else True

        for seed_id in range(RCFG.cnt_seed): 
            logger.info(f'Start training for seed_id {seed_id}.')            
            oof_valid_preds = np.zeros(self.train_feats.shape[0])

            for fold in range(RCFG.n_splits):
                seed = RCFG.base_seed + split_id * (RCFG.n_splits * RCFG.cnt_seed) + seed_id * RCFG.n_splits + fold
                logger.info(f'Start training with model seed {seed} for seed_id {seed_id} fold {fold}.')

                if mode == 'second':
                    cond = (self.feature_importance_df['split_id'] == split_id) & (self.feature_importance_df['fold'] == fold)
                    feature_df = self.feature_importance_df[cond].groupby('feature').mean()
                    feature_df = feature_df.sort_values(by="importance", ascending=False).reset_index()
                    if RCFG.use_random_features:
                        dummy_random_idx = feature_df[feature_df['feature'].str.startswith('dummy_random')].index[RCFG.threshold_random_features]
            
                        self.train_cols = feature_df[(~feature_df['feature'].str.contains('ngram')) | (feature_df.index <= dummy_random_idx)]['feature'].tolist()
                        self.train_cols = [c for c in self.train_cols if not c.startswith('dummy_random')]
                    else:
                        self.train_cols = feature_df[feature_df.index <= RCFG.use_feature_rank]['feature'].tolist()
                        
                    if seed_id == 0:
                        logger.info(f'self.train_cols: {len(self.train_cols)}')

                logger.info(f'Start training for fold {fold}.')
                train_idx = self.train_feats[self.train_feats['fold'] != fold].index
                valid_idx = self.train_feats[self.train_feats['fold'] == fold].index
                
                X_train, y_train = self.train_feats.iloc[train_idx][self.train_cols], self.train_feats.iloc[train_idx][target_col]
                X_valid, y_valid = self.train_feats.iloc[valid_idx][self.train_cols], self.train_feats.iloc[valid_idx][target_col]
                
                params['random_state'] = seed
                if not last: 
                    params['learning_rate'] = 0.05 
                    params['colsample_bytree'] = 0.6

                model = lgb.LGBMRegressor(**params)
                early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
                verbose_callback = lgb.log_evaluation(100)
                model.fit(
                    X_train, y_train, eval_set=[(X_valid, y_valid)],  
                    callbacks=[early_stopping_callback, verbose_callback]
                )
                valid_predict = model.predict(X_valid)
                oof_valid_preds[valid_idx] = valid_predict
                self.models_dict[f'{split_id}_{seed_id}_{fold}'] = model

                rmse = np.round(metrics.mean_squared_error(y_valid, valid_predict, squared=False), 6)
                logger.info(f'Seed_id {seed_id} fold {fold} rmse: {rmse}, best iteration: {model.best_iteration_}')

            self.oof_valid_preds_df[f'oof_{split_id}_{seed_id}'] = oof_valid_preds
            oof_score = np.round(metrics.mean_squared_error(self.train_feats[target_col], oof_valid_preds, squared=False), 6)
            logger.info(f'oof score for seed_id {seed_id}: {oof_score}')
            oofscore.append(oof_score)

        cvscore = np.round(np.mean(oofscore), 6)
        logger.info(f'CV score: {cvscore}')
        if mode == 'first':
            self.first_oofscores[split_id] = oofscore
            self.first_cvscore[split_id] = cvscore
        else:
            self.second_oofscores[split_id] = oofscore
            self.second_cvscore[split_id] = cvscore

    def train(self,):

        if RCFG.debug:
            logger.info('Debug mode. Decrease training time.')
            RCFG.split_cnt = 2
            RCFG.cnt_seed = 2
            RCFG.n_splits = 2

        if RCFG.use_random_features and RCFG.select_feature:
            logger.info('Add random features.')
            self.train_feats = add_random_feature(self.train_feats)

        logger.info(f'Start training. train_feats shape: {self.train_feats.shape}')
        self.feature_importance_df = pd.DataFrame()
        self.models_dict = {}
        self.first_oofscores = {}
        self.second_oofscores = {}
        self.first_cvscore = {}
        self.second_cvscore = {}
        self.oof_valid_preds_df = pd.DataFrame(self.train_feats['id'].unique(), columns=['id'])

        for split_id in range(RCFG.split_cnt):
            kf = model_selection.StratifiedKFold(n_splits=RCFG.n_splits, random_state= 42+ split_id, shuffle=True)
            for fold, (_, valid_idx) in enumerate(kf.split(self.train_feats, self.train_feats['score'].astype(str))):
                self.train_feats.loc[valid_idx, 'fold'] = fold

            logger.info('--------------------------------------------------')
            logger.info(f'Train LightGBM with split seed: {42+split_id}.') 
            logger.info('--------------------------------------------------')   
            self._train_fold_seed(mode='first', split_id=split_id)

            logger.info('Calculate feature importance.')
            for fold in range(RCFG.n_splits):
                for seed_id in range(RCFG.cnt_seed):
                    model = self.models_dict[f'{split_id}_{seed_id}_{fold}']
                    model.importance_type = 'gain'
                    fold_importance_df = pd.DataFrame()
                    fold_importance_df["feature"] = self.train_cols
                    fold_importance_df["importance"] = model.feature_importances_
                    fold_importance_df["split_id"] = split_id
                    fold_importance_df["seed_id"] = seed_id
                    fold_importance_df["fold"] = fold
                    self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)

            if RCFG.select_feature:
                logger.info('--------- Retrain LightGBM with selected features. ---------')
                self._train_fold_seed(mode='second', split_id=split_id)

        oof_column = [c for c in self.oof_valid_preds_df.columns if c.startswith('oof')]
        self.oof_valid_preds_df['mean_oof'] = self.oof_valid_preds_df[oof_column].mean(axis=1)
        self.oof_valid_preds_df = self.oof_valid_preds_df.merge(self.train_scores, on='id')
        self.oof_valid_preds_df['se'] = np.round((self.oof_valid_preds_df['mean_oof'] - self.oof_valid_preds_df['score']) ** 2, 5)
        self.final_score = self.oof_valid_preds_df['se'].mean() ** 0.5

        self.cv_old = np.mean(list(self.second_cvscore.values())) if RCFG.select_feature else np.mean(list(self.first_cvscore.values()))
        logger.info(f'final cv score (old): {self.cv_old}')
        logger.info(f'final cv score (new): {self.final_score}')
        self.oof_valid_preds_df.to_csv(f'{ENV.output_dir}oof_valid_preds.csv', index=False)

        self.feature_importance_df.to_csv(f'{ENV.output_dir}feature_importance.csv', index=False)
        with open(f'{ENV.output_dir}models_dict.pickle', 'wb') as f:
            logger.info(f'save models_dict to {ENV.output_dir}models_dict.pickle')
            pickle.dump(self.models_dict, f)

    def write_sheet(self, ):
        logger.info('Write scores to google sheet.')
        nowstr_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))

        data = [
            nowstr_jst, 
            ENV.commit_hash, 
            class_vars_to_dict(RCFG), 
            self.first_oofscores, 
            self.first_cvscore, 
            sum(self.first_cvscore.values()) / len(self.first_cvscore.values()),
            self.second_oofscores, 
            self.second_cvscore, 
            self.cv_old,
            self.final_score
        ]
        self.sheet.write(data, sheet_name='cvscores')
    

    def predict(self, ):

        if RCFG.load_model:
            logger.info('Load LightGBM model.')
            with open(f'{ENV.model_dir}models_dict.pickle', 'rb') as f:
                self.models_dict = pickle.load(f)
        
        logger.info('Start prediction.')
        test_predict_list = []
        for split_id in range(RCFG.split_cnt):
            for seed_id in range(RCFG.cnt_seed): 
                for fold in range(RCFG.n_splits):
                    model = self.models_dict[f'{split_id}_{seed_id}_{fold}']
                    train_cols = model.feature_name_
                    logger.info(f'model: {split_id}_{seed_id}_{fold}, train_cols: {len(train_cols)}')
                    X_test = self.test_feats[train_cols]
                    test_predict = model.predict(X_test)
                    test_predict_list.append(test_predict)
        
        logger.info('Save submission.csv')
        self.test_feats['score'] = np.mean(test_predict_list, axis=0)
        self.test_feats[['id', 'score']].to_csv("submission.csv", index=False)