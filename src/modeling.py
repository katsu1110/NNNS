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
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn import preprocessing, metrics

# import tensorflow as tf 
# import tensorflow_addons as tfa
# import kerastuner as kt # keras tuner!
import xgboost as xgb
import numerapi

from utils import init_logger, upload, validation_metrics

# ----------------------------
# config
# ----------------------------
EXPERIMENT_NAME = 'XGB_baseline'
SLOT_NAME = 'KATSU_MINAMISAWAK'
EXAMPLE_COL = None
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
INPUT_PATH = f'{parent_dir}/input'
OUTPUT_PATH = f'{parent_dir}/output'

# authentification
from dotenv import load_dotenv
load_dotenv()
NUMERAI_KEY = os.getenv('PUBLIC_ID')
NUMERAI_SECRET = os.getenv('SECRET_KEY')

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
    target = df.columns[df.columns.str.startswith('feature')].values.tolist()[0]
    return features, target

# ================== EDIT START ================== 

def model_dispatcher():
    """Define your model
    """
    # define model
    params = {
        'colsample_bytree': 0.1,
        'learning_rate': 0.01,
        'max_depth': 5,
        'seed': 0,
        'n_estimators': 2000,
        'tree_method': 'gpu_hist' # Let's use GPU for a faster experiment
    }
    model = xgb.XGBRegressor(**params)
    return model

def fit_model(model, 
        train,
        tournament=None, 
        features=['feature_dexterity4', 'feature_charisma76'], 
        target='target'
        ):
    """Write model fitting logic
    """
    # create a train dataset
    train_df = train[features + [target]].dropna(subset=[target])
    train_set = {'X': train_df[features], 'y': train_df[target].values}

    # training
    if tournament is None: # no validation
        # fit
        model.fit(train_set['X'], train_set['y'], verbose=100)
    else:
        # create a valid dataset
        valid_df = tournament.query('data_type == "validation"')[features + [target]].dropna(subset=[target])
        valid_set = {'X': valid_df[features], 'y': valid_df[target].values}

        # fit
        model.fit(
            train_set['X'], train_set['y'], 
            eval_set=[(valid_set['X'], valid_set['y'])],
            early_stopping_rounds=100, 
            verbose=100
            )
    
    # save model
    save_file_path = os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, 'xgb_model.pkl')
    joblib.dump(model, save_file_path)
    return model

def inference(model, tournament, features, pred_col='prediction'):
    """Write model inference logic
    """
    tournament[pred_col] = model.predict(tournament[features])
    return tournament

# ================== EDIT END ==================

# ----------------------------
# run
# ----------------------------
def main():
    # initialize logger
    today = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    logger = init_logger(
        log_file=os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, f'{today}.log')
        )
    logger.info('Start Logging...')

    # get data
    train, tournament = read_data(INPUT_PATH)
    logger.info('Data read!')

    # get features and target
    features, target = get_features_target(train)
    logger.info('target: {}'.format(target))
    logger.info('{} features: {}'.format(len(features), features))
    
    # create model
    model = model_dispatcher()

    # fit model
    model = fit_model(
        model, 
        train,
        tournament, 
        features, 
        target
        )

    # inference
    tournament = inference(model, tournament)

    # compute validation score
    val_df = validation_metrics(
        tournament.query('data_type == "validation"'),
        pred_cols=tournament.columns[tournament.columns.str.startswith('pred')].values.tolist(),
        example_col=EXAMPLE_COL,
        fast_mode=False
        )

    # save validation score
    val_df.to_csv(os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, 'val_score.csv')), index=False)
    logger.info(val_df.to_markdown())

    # submit
    napi = numerapi.NumerAPI(NUMERAI_KEY, NUMERAI_SECRET)
    upload(napi, tournament, upload_type='diagnostics', slot_name=SLOT_NAME, logger=logger)

    # move the submitted file to the experiment folder
    shutil.move('./prediction.csv', os.path.join(OUTPUT_PATH, EXPERIMENT_NAME, 'prediction.csv'))
    logger.info('Prediction is saved...ALL DONE!')

if __name__ == '__main__':
    main()