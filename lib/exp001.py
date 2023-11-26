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
import lightgbm as lgb
import copy
import datetime

from utils.utils import Logger, WriteSheet
from utils.utils import class_vars_to_dict


class ENV:
    input_dir = "/kaggle/input/"
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
    seed_cv = 42
    cnt_seed = 5
    base_seed = 42
    n_splits = 10


# Function to construct essays copied from here (small adjustments): https://www.kaggle.com/code/yuriao/fast-essay-constructor

def processingInputs(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        # Input[0] = activity
        # Input[1] = cursor_position
        # Input[2] = text_change
        # Input[3] = id
        # If activity = Replace
        if Input[0] == 'Replace':
            # splits text_change at ' => '
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue

        # If activity = Paste    
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue

        # If activity = Remove/Cut
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue

        # If activity = Move...
        if "M" in Input[0]:
            # Gets rid of the "Move from to" text
            croppedTxt = Input[0][10:]              
            # Splits cropped text by ' To '
            splitTxt = croppedTxt.split(' To ')              
            # Splits split text again by ', ' for each item
            valueArr = [item.split(', ') for item in splitTxt]              
            # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            # Skip if someone manages to activiate this by moving to same place
            if moveData[0] != moveData[2]:
                # Check if they move text forward in essay (they are different)
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] +\
                    essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] +\
                    essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue                

        # If activity = input
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def getEssays(df):
    # Copy required columns
    textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
    # Get rid of text inputs that make no change
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']     
    # construct essay, fast 
    tqdm.pandas()
    essay=textInputDf.groupby('id')[['activity','cursor_position', 'text_change']].progress_apply(
        lambda x: processingInputs(x))      
    # to dataframe
    essayFrame=essay.to_frame().reset_index()
    essayFrame.columns=['id','essay']
    # Returns the essay series
    return essayFrame


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
    for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_df[word_df['word_len'] >= word_l].groupby(['id']).count().iloc[:, 0]
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df

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
    
    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf
            
        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
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
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering ratios data")
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


    def preprocess(self,):

        self.logger.info('Start preprocessing.')

        self.logger.info('Get essays of test data.')
        test_essays = getEssays(self.test_logs)

        self.logger.info('Start creating features based on sentences and paragraphs.')
        train_word_df = split_essays_into_words(self.train_essays)
        train_word_agg_df = compute_word_aggregations(train_word_df)

        train_sent_df = split_essays_into_sentences(self.train_essays)
        train_sent_agg_df = compute_sentence_aggregations(train_sent_df)

        train_paragraph_df = split_essays_into_paragraphs(self.train_essays)
        train_paragraph_agg_df = compute_paragraph_aggregations(train_paragraph_df)

        
        test_word_agg_df = compute_word_aggregations(split_essays_into_words(test_essays))
        test_sent_agg_df = compute_sentence_aggregations(split_essays_into_sentences(test_essays))
        test_paragraph_agg_df = compute_paragraph_aggregations(split_essays_into_paragraphs(test_essays))

        self.logger.info('Start creating features based on logs.')
        preprocessor = Preprocessor(seed=42)
        train_feats = preprocessor.make_feats(self.train_logs)
        nan_cols = train_feats.columns[train_feats.isna().any()].tolist()
        train_feats = train_feats.drop(columns=nan_cols)
        
        preprocessor = Preprocessor(seed=42)
        test_feats = preprocessor.make_feats(self.test_logs)
        test_feats = test_feats.drop(columns=nan_cols)

        train_agg_fe_df = self.train_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
            ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
        train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
        train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
        train_agg_fe_df.reset_index(inplace=True)

        test_agg_fe_df = self.test_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
            ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
        test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
        test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
        test_agg_fe_df.reset_index(inplace=True)

        train_feats = train_feats.merge(train_agg_fe_df, on='id', how='left')
        test_feats = test_feats.merge(test_agg_fe_df, on='id', how='left')

        data = []

        for logs in [self.train_logs, self.test_logs]:
            logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])
            logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged']) / 1000

            group = logs.groupby('id')['time_diff']
            largest_lantency = group.max()
            smallest_lantency = group.min()
            median_lantency = group.median()
            initial_pause = logs.groupby('id')['down_time'].first() / 1000
            pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
            pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
            pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
            pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
            pauses_3_sec = group.apply(lambda x: (x > 3).sum())

            data.append(pd.DataFrame({
                'id': logs['id'].unique(),
                'largest_lantency': largest_lantency,
                'smallest_lantency': smallest_lantency,
                'median_lantency': median_lantency,
                'initial_pause': initial_pause,
                'pauses_half_sec': pauses_half_sec,
                'pauses_1_sec': pauses_1_sec,
                'pauses_1_half_sec': pauses_1_half_sec,
                'pauses_2_sec': pauses_2_sec,
                'pauses_3_sec': pauses_3_sec,
            }).reset_index(drop=True))

        train_eD592674, test_eD592674 = data

        train_feats = train_feats.merge(train_eD592674, on='id', how='left')
        test_feats = test_feats.merge(test_eD592674, on='id', how='left')
        train_feats = train_feats.merge(self.train_scores, on='id', how='left')

        # Adding the additional features to the original feature set

        train_feats = train_feats.merge(train_word_agg_df, on='id', how='left')
        train_feats = train_feats.merge(train_sent_agg_df, on='id', how='left')
        train_feats = train_feats.merge(train_paragraph_agg_df, on='id', how='left')
        test_feats = test_feats.merge(test_word_agg_df, on='id', how='left')
        test_feats = test_feats.merge(test_sent_agg_df, on='id', how='left')
        test_feats = test_feats.merge(test_paragraph_agg_df, on='id', how='left')

        train_feats.to_csv(f'{ENV.output_dir}train_feats.csv', index=False)
        self.train_feats = train_feats
        self.test_feats = test_feats


    def train(self,):

        if RCFG.debug:
            RCFG.cnt_seed = 2
            RCFG.n_splits = 3

        target_col = ['score']
        drop_cols = ['id']
        train_cols = [col for col in self.train_feats.columns if col not in ['score', 'id']]

        # Code comes from here: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
        self.models_dict = {}
        self.scores = []
        best_params = {
            'reg_alpha': 0.007678095440286993, 
            'reg_lambda': 0.34230534302168353, 
            'colsample_bytree': 0.627061253588415, 
            'subsample': 0.854942238828458, 
            'learning_rate': 0.038697981947473245, 
            'num_leaves': 22, 
            'max_depth': 37, 
            'min_child_samples': 18
        }

        self.logger.info('Start training.')
        for i in range(RCFG.cnt_seed): 
            seed = RCFG.base_seed + i
            self.logger.info(f'Start training for seed {seed}.')

            kf = model_selection.KFold(n_splits=RCFG.n_splits, random_state= seed, shuffle=True)
            oof_valid_preds = np.zeros(self.train_feats.shape[0])
            
            for fold, (train_idx, valid_idx) in enumerate(kf.split(self.train_feats)):
                self.logger.info(f'Start training for fold {fold}.')
                
                X_train, y_train = self.train_feats.iloc[train_idx][train_cols], self.train_feats.iloc[train_idx][target_col]
                X_valid, y_valid = self.train_feats.iloc[valid_idx][train_cols], self.train_feats.iloc[valid_idx][target_col]
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    'random_state': 42,
                    "n_estimators" : 12001,
                    "verbosity": -1,
                    **best_params
                }
                model = lgb.LGBMRegressor(**params)
                early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
                verbose_callback = lgb.log_evaluation(100)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
                        callbacks=[early_stopping_callback, verbose_callback],
                )
                valid_predict = model.predict(X_valid)
                oof_valid_preds[valid_idx] = valid_predict
                self.models_dict[f'{fold}_{i}'] = model

            oof_score = metrics.mean_squared_error(self.train_feats[target_col], oof_valid_preds, squared=False)
            self.scores.append(oof_score)


    def write_sheet(self, ):
        self.logger.info('Write scores to google sheet.')

        nowstr_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        data = [nowstr_jst, ENV.commit_hash, class_vars_to_dict(RCFG)] + self.scores
        self.sheet.write(data, sheet_name='cvscores')
    

    def predict(self, ):

        test_predict_list = []
        for i in range(RCFG.cnt_seed): 
            for fold in range(RCFG.n_splits):
                model = self.models_dict[f'{fold}_{i}']
                train_cols = model.feature_name_
                X_test = self.test_feats[train_cols]
                test_predict = model.predict(X_test)
                test_predict_list.append(test_predict)
        
        self.test_feats['score'] = np.mean(test_predict_list, axis=0)
        self.test_feats[['id', 'score']].to_csv("submission.csv", index=False)


    def run(self,):
        
        pass

