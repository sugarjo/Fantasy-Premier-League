import os
import sys
import re
import pickle

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

from matplotlib import pyplot as plt

import xgboost as xgb

from hyperopt import Trials, fmin, hp, atpe
from hyperopt.early_stop import no_progress_loss

#shared functions for the local, element and all models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl_model_utils import (GROW_POLICY, TEMPORAL_FEATURES, UNKNOWN_FEATURES, LOCAL_FEATURES, ELEMENT_FEATURES, ALL_FEATURES, normalize_feature_flags, build_match_index,
                             mask_unseen_categories, two_fold_loss, cross_fit_predictions,
                             select_columns, fit_xgboost, training_metadata, remove_out_of_range)

#runs the update
update_hyperparams = False
#retrain but does not check the updated hyperparams
check_hyperparameters = True

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"


try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


method = 'xgboost'

season_dfs = []


#optimize hyperparameters. uses the data of all players (set below)
def objective_xgboost(space):
    
    if space["grow_policy"] == 0 or space["grow_policy"] == 1:
        space["grow_policy"] = GROW_POLICY[space["grow_policy"]]
    
    #fit on the first half and predict the second, and vice versa
    return two_fold_loss(space, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds,
                         check_features, unknown_features)

               
with open(r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl", 'rb') as file:
    original_data = pickle.load(file)                

selected = original_data["minutes"] >= 60
train_data = original_data.loc[selected].copy()


#remove players with few matches
unique_names = train_data.name.unique()

#two for train and val
n_tresh = 2

for unique_ind, name in enumerate(unique_names):
    selected = (train_data.name == name)

    if sum(selected) < n_tresh:
        train_data = train_data.loc[~selected]


#included for all windows, but not current
temporal_features = list(TEMPORAL_FEATURES)


#the non digit version of these features will be removed
unknown_features = list(UNKNOWN_FEATURES)



train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)




#get one index for each match. players of both teams in a match get the same index
match_ind = build_match_index(train_X)

train_X = train_X.drop(['kickoff_time'], axis=1)

#include all
sel_name = train_X.element_type == -1  
sel_name = ~sel_name
 
# Get the 80% of the first matches every season...
train_copy = train_X.loc[sel_name].copy()        
        

# Identify categorical columns
categorical_columns = train_copy.select_dtypes(['category']).columns

# Reset categories for each categorical column
for column in categorical_columns:
    train_copy[column] = train_copy[column].cat.remove_unused_categories()
        
train_copy = train_copy.reset_index(drop=True)
train_copy['match_ind'] = pd.Series(match_ind[sel_name])



cvs_mask = train_copy.split == 0  # Mask for cross-validation sample
#vals_mask =train_copy.split == 1  # Mask for validation, simply the inverse of cvs_mask
vals_mask = ~cvs_mask  


 
 
cv_X = train_copy.loc[cvs_mask.values].copy()
val_X =  train_copy.loc[vals_mask.values].copy()
cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()

#make sure all categories in val_x and cv_x are present in the data
val_X = mask_unseen_categories(val_X, train_copy)
cv_X = mask_unseen_categories(cv_X, train_copy)

#match index of the two halves
cv_match_inds = match_ind[sel_name][cvs_mask]
val_match_inds = match_ind[sel_name][vals_mask]

cv_match_count = np.max(train_copy.loc[cvs_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
val_match_count = np.max(train_copy.loc[vals_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
min_eval_fraction = 1/np.min([cv_match_count, val_match_count])  


#individual_features = ['season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_difficulty', 'ict_index', 'transfers_in', 'transfers_out', 'bps'] #, 'difficulty']

# element_features = ['season', 'string_team', 'points_per_played_game', 'points_per_game', 'transfers_in', 'transfers_out', 'minutes', 'was_home', 'season', 'ict_index', 'defcon', 'total_points', 'name',
 #         'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'string_opp_team', 'own_difficulty', 'opp_difficulty', 'local_predictions', 'local_const_predictions', 'influence', 'threat', 'creativity', 'bps'] #, 'difficulty']


all_features = list(ALL_FEATURES)

check_features = all_features


keep_cols = []
#remove the features not needed
for feat in cv_X.keys():
    no_digit_feat = re.sub(r'\d+', '', feat)
    
    if no_digit_feat in check_features:
        keep_cols.append(feat)

cv_X = cv_X[keep_cols]
val_X = val_X[keep_cols]
train_copy = train_copy[keep_cols]


 
 # #include feature search in the hyperparams
 # check_features = ['transfers_in', 'transfers_out', 'minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
 #         'total_points', 'expected_goals', 'expected_assists', 'points_per_played_game', 'was_home', 'season',
 #         'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon', 'name', 'points_per_game', 'string_opp_team', 'own_difficulty', 'opp_difficulty'] #, 'difficulty']
 
 
do_remove_features= ['names', 'points_per_game', 'points_per_played_game', 'season']   


    
hyperparam_path = main_directory + f'\models\hyperparams.pkl'
with open(hyperparam_path, 'rb') as f:
    old_trials = pickle.load(f)

hyperparams = old_trials.best_trial['misc']['vals']
#reformat the lists
old_hyperparams = {}
for field, val in hyperparams.items():
    old_hyperparams[field] = val[0]

#hyperparameters stored before the feature flag fix keep all features
old_hyperparams = normalize_feature_flags(old_hyperparams, check_features)
    

if old_hyperparams['eval_fraction'] < min_eval_fraction:
    old_hyperparams['eval_fraction'] = min_eval_fraction
 
loss = objective_xgboost(old_hyperparams)
old_loss = loss['loss']


print('Old loss: ', old_loss)

space={'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 4000, 1),
       'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(130), 1),
       'max_delta_step': hp.uniform('max_delta_step', 0, 2000),
       'max_depth': hp.qloguniform("max_depth", np.log(1), np.log(300), 1),
       'min_child_weight' : hp.uniform('min_child_weight', 0, 800),
       'max_leaves': hp.quniform('max_leaves', 0, 400, 1),
        'min_split_loss': hp.loguniform('min_split_loss', np.log(0.001), np.log(300)), #log?
        'n_estimators': hp.quniform('n_estimators', 2, 40000, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 1500),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 6000),
        'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.45),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(6)),
        'subsample': hp.uniform('subsample', 0.1, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
        'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
        'grow_policy': hp.choice('grow_policy', [0, 1]), #1
        'temporal_window': hp.quniform('temporal_window', 1, 7, 1),
    }

for feature in check_features:
    # Add a new entry in the dictionary with the feature as the key
    # and hp.quniform('n_estimators', 0, 2, 1) as the value
    space[feature] = hp.choice(feature, [False, True])

#marks hyperparameters where the feature flags work (see normalize_feature_flags)
space['feature_flags_v2'] = hp.choice('feature_flags_v2', [True])

   
trials = Trials() 
        
loss = np.inf
tid = 0

tested_hyperparam_path = main_directory + f'\models\\tested_hyperparams.pkl'


#temp_hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
if update_hyperparams:
    
    while loss > 60 and tid < 200:
    
        #optmimize hyperparameters. use all training data
        new_hyperparams = fmin(fn = objective_xgboost,
                        space = space,
                        algo = atpe.suggest,
                        trials = trials,
                        max_evals=99999999,
                        early_stop_fn=no_progress_loss(750)
                        )
        
        pickle.dump(trials, open(tested_hyperparam_path, "wb"))
        
        loss = trials.best_trial['result']['loss']
        tid = len(trials)
    
    
    for field, val in new_hyperparams.items():
         #new_hyperparams[field] = val[0]
         print(field, val)
    
    
else:

    
    
    with open(tested_hyperparam_path, 'rb') as f:
        new_trials = pickle.load(f)
        
    hyperparams = new_trials.best_trial['misc']['vals']
    #reformat the lists
    new_hyperparams = {}
    for field, val in hyperparams.items():
        new_hyperparams[field] = val[0]
        
    new_hyperparams = normalize_feature_flags(new_hyperparams, check_features)
        
    
    if new_hyperparams['eval_fraction'] < min_eval_fraction:
        new_hyperparams['eval_fraction'] = min_eval_fraction
        
    loss = objective_xgboost(new_hyperparams)
    new_loss = loss['loss']
     
    print('New loss: ', new_loss)
    
    
    if new_loss < old_loss and check_hyperparameters:
        #print('Element error: ', new_loss)
        
        print('Overwriting old loss')
        pickle.dump(new_trials, open(hyperparam_path, "wb"))
        trials = new_trials
        
        mse = new_loss
        
        
    else:
        
        print('Element error: ', old_loss)
        
        print('Keep old loss')
        trials = old_trials
        
        mse = old_loss
        
        
        
    trial_hyperparams = trials.best_trial['misc']['vals']
    
    best_hyperparams = {}
    for field, val in trial_hyperparams.items():
        best_hyperparams[field] = val[0]
        #print(field, val[0])
        
    best_hyperparams = normalize_feature_flags(best_hyperparams, check_features)
        
    
    
     
    if best_hyperparams['eval_fraction'] < min_eval_fraction:
        best_hyperparams['eval_fraction'] = min_eval_fraction
    
    random_error = []
    local_const_error = []
    local_error = []
    element_error = []
    all_error = []
    
    unlikely_values = 0
    
     
    subset_idx = train_data.index[sel_name]        # index of selected rows
    
    for val_pred, y1, y2, test_is_val in cross_fit_predictions(best_hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds, check_features, unknown_features):
        
        #remove values that are outside of the training range
        unlikely_values += remove_out_of_range(val_pred, y1)
        
        random_error.append(np.mean(np.abs((y2 - np.mean(y1))**2)))
        
        rows_to_update = subset_idx[vals_mask if test_is_val else cvs_mask]
        
        train_data.loc[rows_to_update, 'all_predictions'] = val_pred
        
        local_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_predictions'])**2))
        
        local_const_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_const_predictions'])**2))
        
        element_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'element_predictions'])**2))
        
        #nanmean since predictions outside of the training range are set to nan
        all_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'all_predictions'])**2))
         
    print('Mean error:', np.mean(random_error))
    print('Local constant error:', np.mean(local_const_error))
    print('Local error:', np.mean(local_error))
    print('Element error:', np.mean(element_error))
    print('All error:', np.mean(all_error))
            
                     
    #train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\model_data.pkl')  # Set index=False to not include row indices
    
    X1 = train_copy.reset_index(drop=True)
    
    columns = select_columns(X1, best_hyperparams, check_features, unknown_features)
    
    model = fit_xgboost(X1, train_y[sel_name], match_ind[sel_name], best_hyperparams, columns, mask_eval_categories=True)
    
    xgb.plot_importance(model, importance_type='gain',
                    max_num_features=20, show_values=False)
    plt.show()
    
    
    
    
    
    summary = {'model': model, 'train_features': training_metadata(X1[columns]), 'hyperparameters': best_hyperparams}#, 'all_rows': original_df}
    
    model_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\all_model.sav"
    
    pickle.dump(summary, open(model_path, 'wb'))
    
    print('Unlikely values set to nan:', unlikely_values, train_data.shape[0])
        