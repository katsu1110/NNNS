import gc
import os
import pathlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# config
TRAIN_PATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
TOURNAMENT_PATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
SAVE_PATH = '/home/katsuhisa/input'

def get_int(x):
    """Convert era column to int
    """
    try:
        return int(str(x).split('era')[-1])
    except:
        return 1000

def cast_features2int(df):
    """Cast numerai quantile features to int for saving memory
    """
    cols = df.columns
    features = cols[cols.str.startswith('feature')].values.tolist()
    df_cast = (df[features].fillna(0.5) * 4).astype(np.int8)
    df_ = df[list(set(cols.values.tolist()) - set(features))]
    df = pd.concat([df_, df_cast], axis=1)
    df = df[cols]
    return df

def main():
    # fetch train data
    train = pd.read_csv(TRAIN_PATH).pipe(cast_features2int)
    train['era'] = train['era'].apply(get_int)

    # fetch tournament data
    tournament = pd.read_csv(TOURNAMENT_PATH).pipe(cast_features2int)
    tournament['era'] = tournament['era'].apply(get_int)

    # save
    cwd = os.getcwd()
    SAVE_PATH = f'{cwd}/input'
    train.to_parquet(pathlib.Path(f'{SAVE_PATH}/train.parquet'))
    tournament.to_parquet(pathlib.Path(f'{SAVE_PATH}/tournament.parquet'))
    print(f'Train & Tournament data stored in the {SAVE_PATH}!')

if __name__ == '__main__':
    main()