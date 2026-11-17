"""Shared helpers for fitting the local (player) and element (position) xgboost models.

Used by xgboost_fit_local_models.py and xgboost_fit_element_models.py.
"""
import math
import re
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, STATUS_FAIL


GROW_POLICY = ['depthwise', 'lossguide']


def should_keep_column(column_name, threshold):
    """Keep fixed features (no number) and temporal features with a lag below threshold."""
    try:
        # Extract all numbers from the column name
        numbers = re.findall(r'\d+', column_name)
        for number in numbers:
            # If any number is lower than the threshold, return False
            if int(number) < threshold:
                return True
            else:
                return False
    except:
        print(column_name)

    return True


def first_Xpct_unique(seq, X):
    """Return the first X fraction (by first-appearance order) of the unique values in seq."""
    # preserve order of first appearance
    seen = {}
    uniq_in_order = []
    for v in seq:
        if v not in seen:
            seen[v] = True
            uniq_in_order.append(v)
    n_keep = math.ceil(X * len(uniq_in_order))  # use ceil to keep at least one for small groups
    return set(uniq_in_order[:n_keep])


def normalize_feature_flags(hyperparams, features):
    """Make the feature on/off flags of stored hyperparameters consistent.

    Features are switched with hp.choice(feature, [False, True]). hyperopt stores
    the index of the choice, so with this order the index (0/1) has the same
    truth value as the sampled value (False/True).

    Hyperparameters found before this fix ('feature_flags_v2' missing) were
    searched without removing any features (the flags had no effect), so for those
    all features are kept."""
    if not hyperparams:
        return hyperparams

    if 'feature_flags_v2' not in hyperparams:
        for feature in features:
            hyperparams[feature] = 1
        hyperparams['feature_flags_v2'] = 0

    return hyperparams


def build_match_index(train_X):
    """One integer per match. Players of both teams in the same match get the same index."""
    match_ind = pd.factorize(
        train_X[['string_team', 'was_home', 'string_opp_team', 'kickoff_time']]
        .apply(lambda row: '-'.join(row.astype(str)), axis=1)
    )[0]

    for ind in list(set(match_ind)):
        matches = np.where(match_ind == ind)[0]

        if len(matches) > 0:

            df_match = train_X.iloc[matches[0]]

            if df_match['was_home']:
                kick_off = df_match['kickoff_time']
                team_a = df_match['string_team']
                team_b = df_match['string_opp_team']

                #find the opponent team (same match)
                selected = (train_X['string_opp_team'] == team_a) & (train_X['string_team'] == team_b) & (train_X['kickoff_time'] == kick_off) & (train_X['was_home']==0)

                match_ind[selected.values.to_numpy(dtype=bool)] = ind

                if sum(selected) < 6:
                    print(ind, sum(selected), kick_off, team_b, team_a)
                elif sum(selected) > 11:
                    print(ind, sum(selected), kick_off, team_b, team_a)

                if len(matches) < 6:
                    print(ind, len(matches), kick_off, team_a, team_b)
                elif len(matches) > 11:
                    print(ind,len(matches), kick_off, team_a, team_b)

    return match_ind


def mask_unseen_categories(X, reference):
    """Set categorical values in X that are not present in reference to NaN."""
    X = X.copy()
    for column in X.columns:
        if isinstance(X[column].dtype, pd.CategoricalDtype):
            mask = X[column].isin(reference[column])
            X.loc[~mask, column] = np.nan
    return X


def xgb_params(hyperparams):
    """xgboost parameters from a set of hyperparameters."""
    grow_policy = hyperparams['grow_policy']
    if grow_policy == 0 or grow_policy == 1:
        grow_policy = GROW_POLICY[grow_policy]

    return {
        'max_depth': int(hyperparams['max_depth']),
        'min_split_loss': hyperparams['min_split_loss'],
        'reg_lambda': hyperparams['reg_lambda'],
        'reg_alpha': hyperparams['reg_alpha'],
        'min_child_weight': int(hyperparams['min_child_weight']),
        'learning_rate': hyperparams['learning_rate'],
        'subsample': hyperparams['subsample'],
        'colsample_bytree': hyperparams['colsample_bytree'],
        'colsample_bylevel': hyperparams['colsample_bylevel'],
        'colsample_bynode': hyperparams['colsample_bynode'],
        'max_delta_step': hyperparams['max_delta_step'],
        'grow_policy': grow_policy,
        'max_leaves': int(hyperparams['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(hyperparams['max_bin']),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
        }


def select_columns(X, hyperparams, check_features, unknown_features=()):
    """Columns to use: lags below temporal_window, without switched off and unknown features."""
    threshold = int(hyperparams['temporal_window'])

    columns = [col for col in X.columns if should_keep_column(col, threshold)]

    #remove features that are switched off
    for feat in check_features:
        if feat in hyperparams.keys() and not hyperparams[feat]:
            #keep if it does not have a number in front or first is not a digit (i.e. the fixed features)
            columns = [col for col in columns
                       if col != feat and (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit())]

    #remove features that are unknown at prediction time
    columns = [col for col in columns if col not in unknown_features]

    return columns


def season_fit_eval_masks(seasons, match_inds, eval_fraction):
    """Split matches in fit and eval: the first (1-eval_fraction) matches of every season are fit."""
    match_inds = np.asarray(match_inds)

    seasons_df = pd.DataFrame({'season': pd.Series(seasons).values, 'match_ind': match_inds})

    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (seasons_df.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-eval_fraction))
                          .to_dict())

    fit_sample = list(set().union(*season_selection.values()))

    fits_mask = pd.Series(match_inds).isin(fit_sample).values
    return fits_mask, ~fits_mask


def fit_xgboost(X, y, match_inds, hyperparams, columns, mask_eval_categories):
    """Fit a model on columns of X with early stopping on the last matches of every season."""
    fits_mask, evals_mask = season_fit_eval_masks(X['season'], match_inds, hyperparams['eval_fraction'])

    fit_X = X[columns].iloc[fits_mask].copy()
    eval_X = X[columns].iloc[evals_mask].copy()
    fit_y = y.iloc[fits_mask].copy()
    eval_y = y.iloc[evals_mask].copy()

    #make sure all categories in eval_X are present in fit_X
    if mask_eval_categories:
        eval_X = mask_unseen_categories(eval_X, fit_X)

    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]

    return xgb.train(
        params=xgb_params(hyperparams),
        num_boost_round=int(hyperparams['n_estimators']),
        early_stopping_rounds=int(hyperparams['early_stopping_rounds']),
        dtrain=dfit,
        evals=evals,
        verbose_eval=False  # Set to True if you want to see detailed logging
        )


def predict_xgboost(model, X, columns):
    return model.predict(xgb.DMatrix(data=X[columns], enable_categorical=True))


def holdout_error(hyperparams, train_X, train_y, train_match_inds, test_X, test_y, check_features, unknown_features):
    """Fit on train and return the mean squared error on test. None if all features are switched off."""
    columns = select_columns(train_X, hyperparams, check_features, unknown_features)

    #all features switched off. not possible to fit a model
    if len(columns) == 0:
        return None

    model = fit_xgboost(train_X, train_y, train_match_inds, hyperparams, columns, mask_eval_categories=False)

    return mean_squared_error(test_y, predict_xgboost(model, test_X, columns))


def two_fold_loss(hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds,
                  check_features, unknown_features, seconds_per_loss_unit=None):
    """hyperopt loss: mean error of fitting on cv/predicting val and fitting on val/predicting cv.

    If seconds_per_loss_unit is given, the fitting time divided by it is added to the loss."""
    start = time.time()

    val_error = holdout_error(hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, check_features, unknown_features)
    if val_error is None:
        return {'loss': np.inf, 'status': STATUS_FAIL}

    cv_error = holdout_error(hyperparams, val_X, val_y, val_match_inds, cv_X, cv_y, check_features, unknown_features)

    total_error = np.mean([val_error, cv_error])

    if seconds_per_loss_unit:
        total_error += (time.time() - start)/seconds_per_loss_unit

    return {'loss': total_error, 'status': STATUS_OK}


def cross_fit_predictions(hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds,
                          check_features, unknown_features):
    """Out-of-fold predictions: fit on cv and predict val, then fit on val and predict cv.

    Yields (predictions, train_y, test_y, test_is_val) for each of the two folds."""
    for X1, y1, m1, X2, y2, test_is_val in ((cv_X, cv_y, cv_match_inds, val_X, val_y, True),
                                             (val_X, val_y, val_match_inds, cv_X, cv_y, False)):

        X1 = X1.reset_index(drop=True)

        columns = select_columns(X1, hyperparams, check_features, unknown_features)
        model = fit_xgboost(X1, y1, m1, hyperparams, columns, mask_eval_categories=True)

        yield predict_xgboost(model, X2, columns), y1, y2, test_is_val


def training_metadata(X):
    """Feature names and categories of the training data, used to align data at prediction time."""
    metadata = {
            "feature_names": X.columns.tolist(),
            "categorical_columns": {},
        }

    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            metadata["categorical_columns"][col] = {
                "categories": X[col].cat.categories.tolist(),
                "ordered": X[col].cat.ordered,
                }

    return metadata


def remove_out_of_range(predictions, y):
    """Set predictions outside of the range of y to NaN. Returns the number of removed values."""
    too_high = predictions > np.max(y)
    too_low = predictions < np.min(y)
    predictions[too_high] = np.nan
    predictions[too_low] = np.nan
    return sum(too_high) + sum(too_low)
