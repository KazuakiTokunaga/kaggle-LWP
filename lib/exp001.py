# https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features

import gc
import os
import itertools
import pickle
import re
import time
from random import choice, choices
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from itertools import cycle
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble, decomposition, tree
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb
import copy
import datetime

from utils.utils import Logger, WriteSheet
from utils.utils import class_vars_to_dict
from utils.utils import add_random_feature
from utils.essay import getEssays


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
    run_name = 'exp001'
    debug = True
    debug_size = 100
    n_splits = 5
    cnt_seed = 5
    base_seed = 42
    preprocess_train = False
    predict = False
    load_model = False
    select_feature = True
    use_feature_rank = 200
    use_random_features = True
    threshold_random_features = 15
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse",
        'random_state': 42,
        "n_estimators" : 12001,
        "verbosity": -1,
        'reg_lambda': 0.3, 
        'colsample_bytree': 0.8, 
        'subsample': 0.8,
        'learning_rate': 0.1, 
        'num_leaves': 22, 
        'max_depth': 6, 
        'min_child_samples': 18
    }


def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = ['count', 'mean', 'std', 'min', 'max', 'first', 'last', 'sem', q1, 'median', q3, 'skew', pd.DataFrame.kurt, 'sum']

def split_essays_into_words(df):
    essay_df = df
    essay_df['word'] = essay_df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    essay_df = essay_df.explode('word')
    essay_df['word_len'] = essay_df['word'].apply(lambda x: len(x))
    essay_df = essay_df[essay_df['word_len'] != 0]
    return essay_df

def compute_word_aggregations(word_df):
    word_agg_df = word_df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    # for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
    #     word_agg_df[f'word_len_ge_{word_l}_count'] = word_df[word_df['word_len'] >= word_l].groupby(['id']).count().iloc[:, 0]
    #     word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def get_countvectorizer_features(df):

    def get_ngram_df(df, ngram = (1,2), thre=0.03):

        count_vectorizer = CountVectorizer(ngram_range=ngram, min_df=thre)
        X_tokenizer_train = count_vectorizer.fit_transform(df['essay']).todense()
        df_train_index = pd.Index(df['id'].unique(), name = 'id')
        feature_names = count_vectorizer.get_feature_names_out()
        return pd.DataFrame(data=X_tokenizer_train, index = df_train_index, columns=feature_names)
    
    df1 = get_ngram_df(df, ngram=(1,2), thre=0.03)
    df2 = get_ngram_df(df, ngram=(3,3), thre=0.1)
    return pd.concat([df1, df2], axis=1)


def split_essays_into_sentences(df):
    essay_df = df
    #essay_df['id'] = essay_df.index
    essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    essay_df = essay_df.explode('sent')
    essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.sent_len!=0].reset_index(drop=True)
    return essay_df

def compute_sentence_aggregations(df):
    sent_agg_df = pd.concat(
        [df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    )
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index

    # New features intoduced here: https://www.kaggle.com/code/mcpenguin/writing-processes-to-quality-baseline-v2
    for sent_l in [50, 60, 75, 100]:
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = df[df['sent_len'] >= sent_l].groupby(['id']).count().iloc[:, 0]
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_agg_df[f'sent_len_ge_{sent_l}_count'].fillna(0)
    
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df

def split_essays_into_paragraphs(df):
    essay_df = df
    #essay_df['id'] = essay_df.index
    essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
    essay_df = essay_df.explode('paragraph')
    # Number of characters in paragraphs
    essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.paragraph_len!=0].reset_index(drop=True)
    return essay_df

def compute_paragraph_aggregations(df):
    paragraph_agg_df = pd.concat(
        [df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    ) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df


class Preprocessor:
    
    def __init__(self, seed):
        self.seed = seed
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]
        
        self.idf = defaultdict(float)


    def _get_count_dataframe(self, df, colname, target_list):
        """
        ログのcolnameのカウントを取得してカラムにする
        カウントを取得するのはtarget_listに含まれるもののみ
        """

        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in target_list:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
    
        return ret


    def _tf_idf_transform(self, df, ret):
        """
        カウントデータをtf-idfに変換する
        """

        cnts = ret.sum(1)
        for col in ret.columns:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret
    

    def get_count(self, df, colname, target_list):
        ret = self._get_count_dataframe(df, colname, target_list)
        return self._tf_idf_transform(df, ret)


    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations: # 全てのpuncluationを区別せずにカウントする
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret}) # ここで含まれるカラムは1つのみ
        return ret


    def get_input_words(self, df):
        # =>はactivityがReplaceの時のみ出現する
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()

        # この集計はactivityが'Remove/Cut'や'Move'のときを正しく考慮していない
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    

    def make_feats(self, df):
        
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        print("Engineering time data")
        for gap in self.gaps:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        print("Engineering statistical summaries for features")
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'quantile', 'sem', 'mean']),
            ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt])
            ])
        
        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        print("Engineering activity counts data")
        activity_df = self.get_count(df, 'activity', self.activities)

        print("Engineering event counts data")
        down_df = self.get_count(df, 'down_event', self.events)
        up_df = self.get_count(df, 'up_event', self.events)
        
        print("Engineering text change counts data")
        text_change_df = self.get_count(df, 'text_change', self.text_changes)

        print("Engineering punctuation counts data")
        punctuations_df = self.match_punctuations(df)

        for df in [activity_df, down_df, up_df, text_change_df, punctuations_df]:
            feats = pd.concat([feats, df], axis=1)

        print("Engineering input words data")
        input_words_df = self.get_input_words(df)
        feats = feats.merge(input_words_df, on='id', how='left')

        print("Engineering ratios data")
        feats = feats.copy()
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']

        return feats


class Runner():

    def __init__(
        self,
    ):
        
        self.data_to_write = []
        self.logger = Logger()
        tqdm.pandas()

        if ENV.save_to_sheet:
            self.logger.info('Initializing Google Sheet.')
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
            self.logger.info(f'Debug mode. Get only first {RCFG.debug_size} ids.')
            target_id = self.train_logs['id'].unique()[:RCFG.debug_size]
            self.train_logs = self.train_logs[self.train_logs['id'].isin(target_id)]
    
    def _add_features(self,
        df, 
        df_essay,
        mode = 'train'
    ):

        # 22カラム
        word_df = split_essays_into_words(df_essay)
        word_agg_df = compute_word_aggregations(word_df)

        # 700カラム
        countvectorize_df = get_countvectorizer_features(df_essay)

        # 31カラム
        sent_df = split_essays_into_sentences(df_essay)
        sent_agg_df = compute_sentence_aggregations(sent_df)

        # 27カラム
        paragraph_df = split_essays_into_paragraphs(df_essay)
        paragraph_agg_df = compute_paragraph_aggregations(paragraph_df)

        # 286カラム
        preprocessor = Preprocessor(seed=42)
        feats = preprocessor.make_feats(df)

        if mode == 'train':
            self.nan_cols = feats.columns[feats.isna().any()].tolist()
            feats = feats.drop(columns=self.nan_cols)

        # 45カラム（preprocessと重なりが多そう）
        agg_fe_df = df.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
            ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum']
        )
        agg_fe_df.columns = ['_'.join(x) for x in agg_fe_df.columns]
        agg_fe_df = agg_fe_df.add_prefix("tmp_")
        agg_fe_df.reset_index(inplace=True)

        # 9カラム
        df['up_time_lagged'] = df.groupby('id')['up_time'].shift(1).fillna(df['down_time'])
        df['time_diff'] = abs(df['down_time'] - df['up_time_lagged']) / 1000
        group = df.groupby('id')['time_diff']
        largest_lantency = group.max()
        smallest_lantency = group.min()
        median_lantency = group.median()
        initial_pause = df.groupby('id')['down_time'].first() / 1000
        pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
        pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
        pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
        pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
        pauses_3_sec = group.apply(lambda x: (x > 3).sum())

        eD592674 = pd.DataFrame({
            'id': df['id'].unique(),
            'largest_lantency': largest_lantency,
            'smallest_lantency': smallest_lantency,
            'median_lantency': median_lantency,
            'initial_pause': initial_pause,
            'pauses_half_sec': pauses_half_sec,
            'pauses_1_sec': pauses_1_sec,
            'pauses_1_half_sec': pauses_1_half_sec,
            'pauses_2_sec': pauses_2_sec,
            'pauses_3_sec': pauses_3_sec,
        }).reset_index(drop=True)

        feats = feats.merge(agg_fe_df, on='id', how='left')
        feats = feats.merge(eD592674, on='id', how='left')
        feats = feats.merge(word_agg_df, on='id', how='left')
        feats = feats.merge(sent_agg_df, on='id', how='left')
        feats = feats.merge(paragraph_agg_df, on='id', how='left')
        feats = feats.merge(countvectorize_df, on='id', how='left')

        return feats


    def preprocess(self,):

        self.logger.info('Start preprocessing.')
        
        if RCFG.preprocess_train:
            self.logger.info('Prepprocess train data. Create features for train data.')
            train_feats = self._add_features(self.train_logs, self.train_essays, mode='train')
            self.train_feats = train_feats.merge(self.train_scores, on='id', how='left')
            self.train_feats.to_csv(f'{ENV.output_dir}train_feats.csv', index=False)
        else:
            self.logger.info(f'Load train data from {ENV.feature_dir}train_feats.csv.')
            self.train_feats = pd.read_csv(f'{ENV.feature_dir}train_feats.csv')

        if RCFG.predict:        
            self.logger.info('Preprocess test data. Get essays of test data.')
            test_essays = getEssays(self.test_logs)
            self.logger.info('Create features for test data.')
            self.test_feats = self._add_features(self.test_logs, test_essays, mode='predict')

    def train(self,):

        if RCFG.debug:
            self.logger.info('Debug mode. Decrease training time.')
            RCFG.cnt_seed = 2
            RCFG.n_splits = 3

        if RCFG.use_random_features:
            self.logger.info('Add random features.')
            self.train_feats = add_random_feature(self.train_feats)

        target_col = ['score']
        drop_cols = ['id']
        self.train_cols = [col for col in self.train_feats.columns if col not in ['score', 'id', 'fold']]

        # Code comes from here: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
        self.models_dict = {}
        self.scores = []
        params = RCFG.lgbm_params
        
        kf = model_selection.KFold(n_splits=RCFG.n_splits, random_state= 1030, shuffle=True)
        for fold, (_, valid_idx) in enumerate(kf.split(self.train_feats)):
            self.train_feats.loc[valid_idx, 'fold'] = fold

        self.logger.info(f'Start training. train_feats shape: {self.train_feats.shape}')
        for i in range(RCFG.cnt_seed): 
            seed = RCFG.base_seed + i
            self.logger.info(f'Start training for seed {seed}.')
            
            oof_valid_preds = np.zeros(self.train_feats.shape[0])
            
            for fold in range(RCFG.n_splits):

                self.logger.info(f'Start training for fold {fold}.')
                train_idx = self.train_feats[self.train_feats['fold'] != fold].index
                valid_idx = self.train_feats[self.train_feats['fold'] == fold].index
                
                X_train, y_train = self.train_feats.iloc[train_idx][self.train_cols], self.train_feats.iloc[train_idx][target_col]
                X_valid, y_valid = self.train_feats.iloc[valid_idx][self.train_cols], self.train_feats.iloc[valid_idx][target_col]
                
                params['random_state'] = 42 * fold + seed
                params['learning_rate'] = 0.1

                model = lgb.LGBMRegressor(**params)
                early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
                verbose_callback = lgb.log_evaluation(100)
                model.fit(
                    X_train, y_train, eval_set=[(X_valid, y_valid)],  
                    callbacks=[early_stopping_callback, verbose_callback],
                )
                valid_predict = model.predict(X_valid)
                oof_valid_preds[valid_idx] = valid_predict
                self.models_dict[f'{fold}_{i}'] = model

                rmse = np.round(metrics.mean_squared_error(y_valid, valid_predict, squared=False), 6)
                self.logger.info(f'Seed {seed} fold {fold} rmse: {rmse}, best iteration: {model.best_iteration_}')

            oof_score = np.round(metrics.mean_squared_error(self.train_feats[target_col], oof_valid_preds, squared=False), 6)
            self.logger.info(f'oof score for seed {seed}: {oof_score}')
            self.scores.append(oof_score)

        self.cvscore = np.round(np.mean(self.scores), 6)
        self.logger.info(f'CV score: {self.cvscore}')
        self.data_to_write += self.scores.copy() + [self.cvscore]

        self.logger.info('Calculate feature importance.')
        self.feature_importance_df = pd.DataFrame()
        for fold in range(RCFG.n_splits):
            model = self.models_dict[f'{fold}_0']
            model.importance_type = 'gain'
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = self.train_cols
            fold_importance_df["importance"] = model.feature_importances_
            fold_importance_df["fold"] = fold
            self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)
        self.feature_importance_df.to_csv(f'{ENV.output_dir}feature_importance.csv', index=False)

        if RCFG.select_feature:
            self.logger.info('Retrain LightGBM with selected features.')
            self.scores = []
            for i in range(RCFG.cnt_seed):
                seed = RCFG.base_seed + i

                oof_valid_preds = np.zeros(self.train_feats.shape[0])
                for fold in range(RCFG.n_splits):
                    self.logger.info(f'Start training for fold {fold}.')

                    # feature_importanceで上位200位の特徴量だけを用いる
                    feature_df = self.feature_importance_df[self.feature_importance_df['fold'] == fold].groupby('feature').mean()
                    feature_df = feature_df.sort_values(by="importance", ascending=False).reset_index()

                    if RCFG.use_random_features:
                        # dummy_randomから始まる特徴量のうち5番めの特徴量のindexを取得する
                        dummy_random_idx = feature_df[feature_df['feature'].str.startswith('dummy_random')].index[RCFG.threshold_random_features]
                        # dummy_random_idxより上位にある特徴量のみを取得する
                        feature_col = feature_df[feature_df.index <= dummy_random_idx]['feature'].tolist()
                        feature_col = [c for c in feature_col if not c.startswith('dummy_random')]
                    else:
                        feature_col = feature_df.head(RCFG.use_feature_rank)['feature'].tolist()
                    
                    if i == 0:
                        self.logger.info(f'feature_col: {len(feature_col)}')

                    train_idx = self.train_feats[self.train_feats['fold'] != fold].index
                    valid_idx = self.train_feats[self.train_feats['fold'] == fold].index
                    X_train, y_train = self.train_feats.iloc[train_idx][feature_col], self.train_feats.iloc[train_idx][target_col]
                    X_valid, y_valid = self.train_feats.iloc[valid_idx][feature_col], self.train_feats.iloc[valid_idx][target_col]

                    params['random_state'] = 42 * fold + seed
                    params['learning_rate'] = 0.02
                    
                    model = lgb.LGBMRegressor(**params)
                    early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
                    verbose_callback = lgb.log_evaluation(100)
                    model.fit(
                        X_train, y_train, eval_set=[(X_valid, y_valid)],  
                        callbacks=[early_stopping_callback, verbose_callback],
                    )
                    valid_predict = model.predict(X_valid)
                    oof_valid_preds[valid_idx] = valid_predict
                    self.models_dict[f'{fold}_{i}'] = model

                    rmse = np.round(metrics.mean_squared_error(y_valid, valid_predict, squared=False), 6)
                    self.logger.info(f'Seed {seed} fold {fold} rmse: {rmse}')

                oof_score = np.round(metrics.mean_squared_error(self.train_feats[target_col], oof_valid_preds, squared=False), 6)
                self.logger.info(f'oof score for seed {seed}: {oof_score}')
                self.scores.append(oof_score)
        
            self.cvscore = np.round(np.mean(self.scores), 6)
            self.logger.info(f'CV score: {self.cvscore}')
            self.data_to_write += self.scores.copy() + [self.cvscore]

        # self.models_dictをpickleで保存
        with open(f'{ENV.output_dir}models_dict.pickle', 'wb') as f:
            self.logger.info(f'save models_dict to {ENV.output_dir}models_dict.pickle')
            pickle.dump(self.models_dict, f)
    

    def write_sheet(self, ):
        self.logger.info('Write scores to google sheet.')

        nowstr_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        data = [nowstr_jst, ENV.commit_hash, class_vars_to_dict(RCFG)] + self.data_to_write
        self.sheet.write(data, sheet_name='cvscores')
    

    def predict(self, ):

        if RCFG.load_model:
            self.logger.info('Load LightGBM model.')
            with open(f'{ENV.model_dir}models_dict.pickle', 'rb') as f:
                self.models_dict = pickle.load(f)
        
        self.logger.info('Start prediction.')
        test_predict_list = []
        for i in range(RCFG.cnt_seed): 
            for fold in range(RCFG.n_splits):
                model = self.models_dict[f'{fold}_{i}']
                train_cols = model.feature_name_
                X_test = self.test_feats[train_cols]
                test_predict = model.predict(X_test)
                test_predict_list.append(test_predict)
        
        self.logger.info('Save submission.csv')
        self.test_feats['score'] = np.mean(test_predict_list, axis=0)
        self.test_feats[['id', 'score']].to_csv("submission.csv", index=False)


    def run(self,):
        
        pass

