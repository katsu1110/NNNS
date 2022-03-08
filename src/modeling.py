import numpy as np
import pandas as pd
import os
import sys
import shutil
import math
import random
import gc
import pathlib
import datetime
import pytz
import joblib
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn import preprocessing, metrics

# import tensorflow as tf 
# import tensorflow_addons as tfa
# import kerastuner as kt # keras tuner!
import xgboost as xgb

import wandb
import numerapi

from utils import init_logger, upload, validation_metrics

# ----------------------------
# config
# ----------------------------
EXPERIMENT_NAME = 'XGB_baseline'
SLOT_NAME = 'KATSU_MINAMISAWAK'
EXAMPLE_COL = None
cwd = os.getcwd()
INPUT_PATH = f'{cwd}/input'
OUTPUT_PATH = f'{cwd}/output'

# authentification
from dotenv import load_dotenv
load_dotenv(pathlib.Path(f'{cwd}/.env'))
NUMERAI_KEY = os.getenv('NUMERAI_KEY')
NUMERAI_SECRET = os.getenv('NUMERAI_SECRET')

# ----------------------------
# modeling
# ----------------------------
def read_data(input_path):
    """Read pre-stored data
    """
    train = pd.read_parquet(os.path.join(input_path, 'train.parquet'))
    tournament = pd.read_parquet(os.path.join(input_path, 'tournament.parquet'))
    return train, tournament

def get_features_target(df):
    """Get feature name list and target name
    """
    features = df.columns[df.columns.str.startswith('feature')].values.tolist()
    target = df.columns[df.columns.str.startswith('target')].values.tolist()[0]
    return features, target

# ================== EDIT START ================== 

def params_dispatcher():
    """Define hyperparameters
    """
    params = {
        'colsample_bytree': 0.1,
        'learning_rate': 0.01,
        'max_depth': 5,
        'seed': 0,
        'n_estimators': 2000,
        # 'tree_method': 'gpu_hist' # Let's use GPU for a faster experiment
    }
    return params

def fit_model(
    params, 
    train,
    tournament, 
    features=['feature_dexterity4', 'feature_charisma76'], 
    target='target',
    pred_col='prediction'
    ):
    """Write model fitting logic
    """
    # create a train dataset
    train_df = train[features + [target]].dropna(subset=[target])
    train_set = {'X': train_df[features], 'y': train_df[target].values}
    train_set = xgb.DMatrix(train_set['X'], label=train_set['y'])
    
    # create a valid dataset
    valid_df = tournament.query('data_type == "validation"')[features + [target]].dropna(subset=[target])
    valid_set = {'X': valid_df[features], 'y': valid_df[target].values}
    valid_set = xgb.DMatrix(valid_set['X'], label=valid_set['y'])

    # fit
    model = xgb.train(
        params, 
        train_set,
        num_boost_round=params['n_estimators'],
        evals=[(train_set, 'train'), (valid_set, 'eval')],
        callbacks=[wandb.xgboost.wandb_callback()],
        early_stopping_rounds=100, 
        )

    # inference
    tournament[pred_col] = np.nan
    tournament.loc[tournament['data_type'] == 'validation', pred_col] = model.predict(valid_set)
    
    # save model
    save_file_path = os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, 'xgb_model.pkl')
    joblib.dump(model, save_file_path)
    return tournament

# ================== EDIT END ==================

# ----------------------------
# run
# ----------------------------
def main():
    # --------------------------------------
    # initialize
    # --------------------------------------
    # make output folder
    os.makedirs(os.path.join(OUTPUT_PATH, EXPERIMENT_NAME), exist_ok=True)
    
    # initialize logger
    today = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    logger = init_logger(
        log_file=os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, f'{today}.log')
        )
    logger.info('Start Logging...')

    # --------------------------------------
    # modeling
    # --------------------------------------
    # define hyperparameters
    params = params_dispatcher()

    # init wandb
    run = wandb.init(project=EXPERIMENT_NAME, entity='katsu1110', config=params, reinit=True)
    
    # get data
    train, tournament = read_data(INPUT_PATH)
    logger.info('Data read!')

    # get features and target
    features, target = get_features_target(train)
    logger.info('target: {}'.format(target))
    logger.info('{} features: {}'.format(len(features), features))

    # fit model and perform inference
    tournament = fit_model(
        params,
        train,
        tournament, 
        features, 
        target,
        pred_col='prediction'
        )

    # --------------------------------------
    # validation scoring
    # --------------------------------------
    # compute validation score
    val_df, corr_per_era = validation_metrics(
        tournament.query('data_type == "validation"'),
        pred_cols=tournament.columns[tournament.columns.str.startswith('pred')].values.tolist(),
        example_col=EXAMPLE_COL,
        fast_mode=False
        )

    # save validation score
    val_df.to_csv(os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, 'val_score.csv'), index=False)
    logger.info(val_df.to_markdown())

    # --------------------------------------
    # upload to wandb
    # --------------------------------------
    # validation metrics
    table = wandb.Table(data=val_df)
    wandb.log({'Validation metrics': table})    
    
    # correlation per era
    for pred_col, corr in corr_per_era.items():    
        data = [[label, val] for (label, val) in zip(corr.values[:, 0], corr.values[:, 1])]
        table = wandb.Table(data=data, columns=["era", "corr"])
        wandb.log(
            {f"Corr per era ({target})" : wandb.plot.bar(table, "era", "corr", title=f"Corr per era ({pred_col})")}
            )
    run.finish()

    # --------------------------------------
    # submit
    # --------------------------------------
    # napi = numerapi.NumerAPI(NUMERAI_KEY, NUMERAI_SECRET)
    # upload(napi, tournament, upload_type='diagnostics', slot_name=SLOT_NAME, logger=logger)

    # # move the submitted file to the experiment folder
    # shutil.move('./prediction.csv', os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, 'prediction.csv'))
    logger.info('Prediction is saved...ALL DONE!')

if __name__ == '__main__':
    main()