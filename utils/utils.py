import logging
import os
import subprocess
import json
import datetime
import numpy as np
import pandas as pd
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from oauth2client.service_account import ServiceAccountCredentials


class Logger:

    def __init__(self, log_path=''):
        self.general_logger = logging.getLogger('general')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(f'{log_path}general.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def now_string(self):
        return str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))


class WriteSheet:

    def __init__(self, 
        sheet_json_key,
        sheet_key,
    ):

        import gspread
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(sheet_json_key, scope)
        gs = gspread.authorize(credentials)
        self.worksheet = gs.open_by_key(sheet_key)
    

    def write(self, data, sheet_name, table_range='A1'):
        sheet = self.worksheet.worksheet(sheet_name)

        # 辞書のみJSONに変換、ほかはそのままにして、書き込む
        data_json = [json.dumps(d, ensure_ascii=False) if type(d) == dict else d for d in data]
        sheet.append_row(data_json, table_range=table_range)


def get_commit_hash(repo_path='/kaggle/working/kaggle-LWP'):

    wd = os.getcwd()
    os.chdir(repo_path)
    
    cmd = "git show --format='%H' --no-patch"
    hash_value = subprocess.check_output(cmd.split()).decode('utf-8')[1:-3]

    os.chdir(wd)

    return hash_value


def add_random_feature(df, n=31):
    
    height = df.shape[0]

    np.random.seed(342)
    data = np.random.randint(1, 1000, size=(height, n))

    df_rand = pd.DataFrame(data, columns=[f'dummy_random{i}' for i in range(n)])    
    df = pd.concat([df, df_rand], axis=1)
    
    return df


def class_vars_to_dict(cls):
    return {key: value for key, value in cls.__dict__.items() if not key.startswith("__") and not callable(value)}


def print_gpu_utilization(logger):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")


def save_model_as_kaggle_dataset(self, title, dir):

    metadata = {
        "title": title,
        "id": f"kazuakitokunaga/{title}",
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(f'{dir}/dataset-metadata.json', 'w') as f:
        json.dump(metadata, f)

    subprocess.call(f'kaggle datasets version -r zip -p {dir} -m "Updateddata"'.split())


class Wandbclient:
    
    def __init__(self, key):

        import wandb
        self.wandb = wandb
        self.wandb.login(key = key)
    
    def start(self, run_name, config):        
        self.wandb.init(
            project='llmdetect',
            name=run_name,
            config=config,
        )
    
    def finish(self):
        self.wandb.finish()