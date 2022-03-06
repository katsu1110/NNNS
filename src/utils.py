import os
import requests
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
import json
from scipy.stats import skew, kurtosis
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

# --------------------------------
# config
# --------------------------------
ERA_COL = "era"
TARGET_COL = "target_nomi"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"

# --------------------------------
# logging
# --------------------------------
def init_logger(log_file='train.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

# --------------------------------
# upload
# --------------------------------
def upload(napi, sub_df, upload_type='diagnostics', slot_name='XXX', logger=None):
    """Upload prediction to Numerai    
    """
    
    # fetch model slot id
    model_slots = napi.get_models()
    slot_id = model_slots[slot_name.lower()]
    
    # format submission dataframe
    sdf = sub_df.index.to_frame()
    sdf['data_type'] = sub_df['data_type'].values
    sdf['prediction'] = sub_df['prediction'].values
    
    # upload
    if upload_type.lower() == 'diagnostics': # diagnostics
        sdf.query('data_type == "validation"')[['id', 'prediction']].to_csv(f'./prediction.csv', index=False)
        try:
            napi.upload_diagnostics(f'./prediction.csv', model_id=slot_id, )
            if logger is None:
                print(f'{slot_name} submitted for diagnositics!')
            else:
                logger.info(f'{slot_name} submitted for diagnositics!')
        except Exception as e:
            if logger is None:
                print(f'Submission to diagnostics ERR...{slot_name}: {e}')
            else:
                logger.info(f'Submission to diagnostics ERR...{slot_name}: {e}')
    else: # predictions for the tournament data
        in_data = ['test', 'live']
        sdf.query('data_type in @in_data')[['id', 'prediction']].to_csv(f'./prediction.csv', index=False)
        try:
            napi.upload_predictions('./prediction.csv', model_id=slot_id, version=2)
            if logger is None:
                print(f'{slot_name} submitted for predictions!')
            else:
                logger.info(f'{slot_name} submitted for predictions!')
        except Exception as e:
            if logger is None:
                print(f'Submission ERR...{slot_name}: {e}')
            else:
                logger.info(f'Submission ERR...{slot_name}: {e}')

# --------------------------------
# scoring
# --------------------------------
def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df, prediction_col):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_col],
                                          feature_cols)[prediction_col]
    scores = df.groupby("era").apply(
        lambda x: (unif(x["neutral_sub"]).corr(x[TARGET_COL]))).mean()
    return np.mean(scores)


def fast_score_by_date(df, columns, target, tb=None, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())


def validation_metrics(validation_data, pred_cols, example_col=None, fast_mode=False):
    validation_stats = pd.DataFrame()
    feature_cols = [c for c in validation_data if c.startswith("feature_")]
    for pred_col in pred_cols:
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(ERA_COL).apply(
            lambda d: unif(d[pred_col]).corr(d[TARGET_COL]))

        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std

        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe

        rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                      min_periods=1).max()
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
        validation_stats.loc["max_drawdown", pred_col] = max_drawdown

        payout_scores = validation_correlations.clip(-0.25, 0.25)
        payout_daily_value = (payout_scores + 1).cumprod()

        apy = (
            (
                (payout_daily_value.dropna().iloc[-1])
                ** (1 / len(payout_scores))
            )
            ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
            - 1
        ) * 100

        validation_stats.loc["apy", pred_col] = apy

        if not fast_mode:
            # Check the feature exposure of your validation predictions
            max_per_era = validation_data.groupby(ERA_COL).apply(
                lambda d: d[feature_cols].corrwith(d[pred_col]).abs().max())
            max_feature_exposure = max_per_era.mean()
            validation_stats.loc["max_feature_exposure", pred_col] = max_feature_exposure

            # Check feature neutral mean
            feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_col)
            validation_stats.loc["feature_neutral_mean", pred_col] = feature_neutral_mean

            # Check top and bottom 200 metrics (TB200)
            tb200_validation_correlations = fast_score_by_date(
                validation_data,
                [pred_col],
                TARGET_COL,
                tb=200,
                era_col=ERA_COL
            )

            tb200_mean = tb200_validation_correlations.mean()[pred_col]
            tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
            tb200_sharpe = tb200_mean / tb200_std

            validation_stats.loc["tb200_mean", pred_col] = tb200_mean
            validation_stats.loc["tb200_std", pred_col] = tb200_std
            validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe

        # MMC over validation
        if example_col is not None:
            mmc_scores = []
            corr_scores = []
            for _, x in validation_data.groupby(ERA_COL):
                series = neutralize_series(unif(x[pred_col]), (x[example_col]))
                mmc_scores.append(np.cov(series, x[TARGET_COL])[0, 1] / (0.29 ** 2))
                corr_scores.append(unif(x[pred_col]).corr(x[TARGET_COL]))

            val_mmc_mean = np.mean(mmc_scores)
            val_mmc_std = np.std(mmc_scores)
            corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
            corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

            validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
            validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

            # Check correlation with example predictions
            per_era_corrs = validation_data.groupby(ERA_COL).apply(lambda d: unif(d[pred_col]).corr(unif(d[example_col])))
            corr_with_example_preds = per_era_corrs.mean()
            validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds

    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()