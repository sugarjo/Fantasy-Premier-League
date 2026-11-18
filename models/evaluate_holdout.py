"""Honest comparison of the local, element and all-players models on a held-out season.

Every level is rebuilt using only the seasons before holdout_season, and the held-out
season is then predicted level by level:

- local:   out-of-fold local predictions within the earlier seasons (as in
           xgboost_fit_local_models.py). The held-out season is predicted by local models
           fitted on all earlier seasons.
- element: fitted on the earlier seasons (with the out-of-fold local predictions).
           Out-of-fold element predictions within the earlier seasons are made for the all
           model. The held-out season is predicted with its local predictions from above.
- all:     fitted on the earlier seasons (with the out-of-fold element predictions), and
           predicts the held-out season with its element predictions from above.

No model ever sees a target of the held-out season, so the errors on it are not affected
by the stacking leak. Seasons after the held-out season are not used.

The stored hyperparameters are reused (local model files, hyperparams{pos}.pkl and
hyperparams.pkl). They were selected with data that includes the held-out season, which
is a small optimistic bias in the hyperparameter choice, but not in the predictions.

Nothing in the normal pipeline is changed or overwritten. The predictions for the held-out
season are saved to holdout_predictions.pkl.
"""
import os
import sys
import pickle

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

#shared functions for the local, element and all models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl_model_utils import (TEMPORAL_FEATURES, UNKNOWN_FEATURES, LOCAL_FEATURES, ELEMENT_FEATURES, ALL_FEATURES,
                             normalize_feature_flags, build_match_index, season_fit_eval_masks,
                             cross_fit_predictions, select_columns, fit_xgboost, predict_xgboost,
                             remove_out_of_range, keep_feature_columns)

#the last complete season. seasons after this are not used
holdout_season = '2025-26'

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'

try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'

data_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl"
local_model_directory = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\local_models"
output_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\holdout_predictions.pkl"


def load_trial_hyperparams(path, features):
    """Best hyperparameters of a stored hyperopt Trials object."""
    with open(path, 'rb') as f:
        trials = pickle.load(f)
    hyperparams = {field: val[0] for field, val in trials.best_trial['misc']['vals'].items()}
    return normalize_feature_flags(hyperparams, features)


def min_eval_fraction(seasons_a, seasons_b, match_inds_a, match_inds_b):
    """Smallest eval_fraction that gives at least one eval match per season (as in the fitting scripts)."""
    counts = []
    for seasons, match_inds in ((seasons_a, match_inds_a), (seasons_b, match_inds_b)):
        df = pd.DataFrame({'season': np.asarray(seasons), 'match_ind': np.asarray(match_inds)})
        counts.append(df.groupby('season')['match_ind'].nunique().max())
    return 1/min(counts)


def with_min_eval_fraction(hyperparams, fraction):
    hyperparams = dict(hyperparams)
    if hyperparams['eval_fraction'] < fraction:
        hyperparams['eval_fraction'] = fraction
    return hyperparams


def drop_unused_categories(X):
    X = X.copy()
    for column in X.select_dtypes(['category']).columns:
        X[column] = X[column].cat.remove_unused_categories()
    empty = [col for col in X.select_dtypes(['category']).columns if len(X[col].cat.categories) == 0]
    return X.drop(columns=empty)


def nan_mse(y, predictions):
    return np.nanmean((np.asarray(y, dtype=float) - np.asarray(predictions, dtype=float))**2)


with open(data_path, 'rb') as file:
    original_data = pickle.load(file)

#only matches with at least 60 minutes, as in the fitting scripts
data = original_data.loc[original_data["minutes"] >= 60].copy()

#remove predictions from the normal pipeline. they are made again below
data = data.drop(columns=[c for c in ['split', 'local_predictions', 'local_const_predictions', 'element_predictions', 'all_predictions'] if c in data.columns])

#no data after the held-out season
seasons = data['season'].astype(str)
data = data.loc[seasons <= holdout_season].copy()
is_holdout = (data['season'].astype(str) == holdout_season).values
is_train = ~is_holdout

print('Training rows:', is_train.sum(), 'Held-out rows:', is_holdout.sum())

y = data['total_points'].astype(int)
X = data.drop(columns=TEMPORAL_FEATURES)

match_ind = build_match_index(X)
X = X.drop(['kickoff_time'], axis=1)

data['split'] = np.nan
for col in ['local_predictions', 'local_const_predictions', 'element_predictions', 'all_predictions']:
    data[col] = np.nan


#--------------------------------------------------------------------------------------------
#LOCAL MODELS
#--------------------------------------------------------------------------------------------
print('\nLocal models')

unlikely_values = 0

names = X.name.unique()
for ind, name in enumerate(names):

    sel = (X.name == name).values
    sel_train = sel & is_train
    sel_holdout = sel & is_holdout

    if ind % 100 == 0:
        print(ind, '/', len(names))

    #no earlier data: nan for the held-out season
    if sel_train.sum() == 0:
        continue

    #mean of the earlier seasons for the held-out season
    data.loc[sel_holdout, 'local_const_predictions'] = y[sel_train].mean()

    #at least one train and val
    if sel_train.sum() < 2:
        continue

    #the same categories for the earlier seasons and the held-out season
    player_X = keep_feature_columns(drop_unused_categories(X.loc[sel]), LOCAL_FEATURES)
    train_X = player_X.loc[sel_train[sel]].reset_index(drop=True)
    holdout_X = player_X.loc[sel_holdout[sel]]
    train_y = y[sel_train]
    train_match_inds = match_ind[sel_train]
    train_idx = data.index[sel_train]

    #first half of the matches of every season for cv, the second half for validation (as in the local script)
    cvs_mask, vals_mask = season_fit_eval_masks(train_X['season'], train_match_inds, 0.5)
    data.loc[train_idx[cvs_mask], 'split'] = 1
    data.loc[train_idx[vals_mask], 'split'] = 0

    cv_X, val_X = train_X.loc[cvs_mask], train_X.loc[vals_mask]
    cv_y, val_y = train_y[cvs_mask], train_y[vals_mask]

    #out-of-fold constant predictions
    data.loc[train_idx[vals_mask], 'local_const_predictions'] = np.mean(cv_y)
    data.loc[train_idx[cvs_mask], 'local_const_predictions'] = np.mean(val_y)

    #too few matches for a model: leave one out constant and no local predictions (as in the local script)
    _, matches_season1 = np.unique(cv_X.season, return_counts=True)
    _, matches_season2 = np.unique(val_X.season, return_counts=True)

    if len(matches_season1) == 0 or len(matches_season2) == 0 or max(matches_season1) <= 3 or max(matches_season2) <= 3:
        total = train_y.sum()
        data.loc[train_idx, 'local_const_predictions'] = ((total - train_y)/(len(train_y) - 1)).values
        continue

    #stored hyperparameters of the player
    model_path = local_model_directory + f'\\{name}.sav'
    try:
        with open(model_path, 'rb') as f:
            hyperparams = pickle.load(f)['hyperparams']
    except:
        hyperparams = []

    if not hyperparams:
        continue

    hyperparams = normalize_feature_flags(dict(hyperparams), LOCAL_FEATURES)
    hyperparams = with_min_eval_fraction(hyperparams, 1/min([max(matches_season1), max(matches_season2)]))

    #out-of-fold local predictions within the earlier seasons
    for predictions, y1, y2, test_is_val in cross_fit_predictions(hyperparams, cv_X, cv_y, train_match_inds[cvs_mask],
                                                                 val_X, val_y, train_match_inds[vals_mask],
                                                                 LOCAL_FEATURES, UNKNOWN_FEATURES):
        unlikely_values += remove_out_of_range(predictions, y1)
        data.loc[train_idx[vals_mask if test_is_val else cvs_mask], 'local_predictions'] = predictions

    #held-out season: local model fitted on all earlier seasons
    if sel_holdout.sum() > 0:
        columns = select_columns(train_X, hyperparams, LOCAL_FEATURES, UNKNOWN_FEATURES)
        model = fit_xgboost(train_X, train_y, train_match_inds, hyperparams, columns, mask_eval_categories=True)
        data.loc[sel_holdout, 'local_predictions'] = predict_xgboost(model, holdout_X, columns)

#as in the pick script: held-out local predictions outside of the range of the training predictions are nan
train_range = data.loc[is_train, 'local_predictions']
outside = is_holdout & ((data['local_predictions'] > train_range.max()) | (data['local_predictions'] < train_range.min())).values
data.loc[outside, 'local_predictions'] = np.nan

print('Unlikely local values set to nan:', unlikely_values + outside.sum())


#--------------------------------------------------------------------------------------------
#ELEMENT MODELS
#--------------------------------------------------------------------------------------------
print('\nElement models')

#players with fewer than two matches are not used for training (as in the element script)
matches_per_player = data.loc[is_train, 'name'].astype(str).value_counts()
enough_matches = data['name'].astype(str).map(matches_per_player).fillna(0).values >= 2

X = data.drop(columns=TEMPORAL_FEATURES + ['kickoff_time'])

for pos in range(1, 5):

    sel = (X.element_type == pos).values
    sel_train = sel & is_train & enough_matches
    sel_holdout = sel & is_holdout

    position_X = keep_feature_columns(drop_unused_categories(X.loc[sel_train | sel_holdout]), ELEMENT_FEATURES)
    train_X = position_X.loc[data.index[sel_train]].reset_index(drop=True)
    holdout_X = position_X.loc[data.index[sel_holdout]]
    train_y = y[sel_train]
    train_match_inds = match_ind[sel_train]
    train_idx = data.index[sel_train]

    #the same halves as the local models
    cvs_mask = (data.loc[sel_train, 'split'] == 0).values
    vals_mask = ~cvs_mask

    hyperparams = load_trial_hyperparams(main_directory + f'\\models\\hyperparams{pos}.pkl', ELEMENT_FEATURES)
    hyperparams = with_min_eval_fraction(hyperparams, min_eval_fraction(train_X.loc[cvs_mask, 'season'], train_X.loc[vals_mask, 'season'],
                                                                        train_match_inds[cvs_mask], train_match_inds[vals_mask]))

    #out-of-fold element predictions within the earlier seasons (features for the all model)
    for predictions, y1, y2, test_is_val in cross_fit_predictions(hyperparams, train_X.loc[cvs_mask], train_y[cvs_mask], train_match_inds[cvs_mask],
                                                                 train_X.loc[vals_mask], train_y[vals_mask], train_match_inds[vals_mask],
                                                                 ELEMENT_FEATURES, UNKNOWN_FEATURES):
        remove_out_of_range(predictions, y1)
        data.loc[train_idx[vals_mask if test_is_val else cvs_mask], 'element_predictions'] = predictions

    #held-out season: element model fitted on all earlier seasons
    columns = select_columns(train_X, hyperparams, ELEMENT_FEATURES, UNKNOWN_FEATURES)
    model = fit_xgboost(train_X, train_y, train_match_inds, hyperparams, columns, mask_eval_categories=True)
    predictions = predict_xgboost(model, holdout_X, columns)

    #as in the pick script: outside of the range of the training predictions is nan
    train_range = data.loc[train_idx, 'element_predictions']
    predictions[(predictions > train_range.max()) | (predictions < train_range.min())] = np.nan

    data.loc[sel_holdout, 'element_predictions'] = predictions

    print('Position', pos, 'done')


#--------------------------------------------------------------------------------------------
#ALL MODEL
#--------------------------------------------------------------------------------------------
print('\nAll model')

X = data.drop(columns=TEMPORAL_FEATURES + ['kickoff_time'])

sel_train = is_train & enough_matches

all_X = keep_feature_columns(drop_unused_categories(X.loc[sel_train | is_holdout]), ALL_FEATURES)
train_X = all_X.loc[data.index[sel_train]].reset_index(drop=True)
holdout_X = all_X.loc[data.index[is_holdout]]
train_y = y[sel_train]
train_match_inds = match_ind[sel_train]

cvs_mask = (data.loc[sel_train, 'split'] == 0).values
vals_mask = ~cvs_mask

hyperparams = load_trial_hyperparams(main_directory + '\\models\\hyperparams.pkl', ALL_FEATURES)
hyperparams = with_min_eval_fraction(hyperparams, min_eval_fraction(train_X.loc[cvs_mask, 'season'], train_X.loc[vals_mask, 'season'],
                                                                    train_match_inds[cvs_mask], train_match_inds[vals_mask]))

columns = select_columns(train_X, hyperparams, ALL_FEATURES, UNKNOWN_FEATURES)
model = fit_xgboost(train_X, train_y, train_match_inds, hyperparams, columns, mask_eval_categories=True)

#as in the pick script: clipped to the range of points in the training data
data.loc[is_holdout, 'all_predictions'] = np.clip(predict_xgboost(model, holdout_X, columns), train_y.min(), train_y.max())


#--------------------------------------------------------------------------------------------
#RESULTS ON THE HELD-OUT SEASON
#--------------------------------------------------------------------------------------------
holdout = data.loc[is_holdout].copy()
train_rows = data.loc[is_train]
position_means = train_rows['total_points'].astype(float).groupby(train_rows['element_type'].astype(int)).mean()
holdout['position_mean'] = holdout['element_type'].astype(int).map(position_means)

prediction_columns = ['position_mean', 'local_const_predictions', 'local_predictions', 'element_predictions', 'all_predictions']

#rows where all models have a prediction, for a fair comparison
complete = holdout[prediction_columns].notna().all(axis=1)

print('\nHeld-out season:', holdout_season, '| rows:', len(holdout), '| rows with all predictions:', complete.sum())
print(f"{'':28s}{'MSE (own rows)':>16s}{'rows':>8s}{'MSE (common rows)':>20s}")
for col in prediction_columns:
    has_prediction = holdout[col].notna()
    print(f"{col:28s}{nan_mse(holdout.loc[has_prediction, 'total_points'], holdout.loc[has_prediction, col]):16.3f}{has_prediction.sum():8d}"
          f"{nan_mse(holdout.loc[complete, 'total_points'], holdout.loc[complete, col]):20.3f}")

print('\nMSE per position (common rows)')
print(f"{'position':>10s}" + ''.join(f'{c[:12]:>14s}' for c in prediction_columns))
for pos in range(1, 5):
    rows = complete & (holdout['element_type'] == pos)
    print(f'{pos:10d}' + ''.join(f"{nan_mse(holdout.loc[rows, 'total_points'], holdout.loc[rows, c]):14.3f}" for c in prediction_columns))

holdout.to_pickle(output_path)
print('\nSaved held-out predictions to', output_path)
