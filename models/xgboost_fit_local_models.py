import os
import sys
import re
import pickle

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

from hyperopt import Trials, fmin, hp, atpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate

#shared functions for the local and element models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl_model_utils import (GROW_POLICY, TEMPORAL_FEATURES, UNKNOWN_FEATURES, LOCAL_FEATURES, ELEMENT_FEATURES, ALL_FEATURES, normalize_feature_flags, build_match_index,
                             mask_unseen_categories, two_fold_loss, cross_fit_predictions,
                             select_columns, fit_xgboost, season_fit_eval_masks, training_metadata, remove_out_of_range)

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"


try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'

#optimize everything. Typically used when hyperparams ranges change. Does not predict new points
optimize_mode = False

#this will exclude the players with frozen data
exclude_frozen_players = False

current_season = '2026-27'
max_temporal_window = 12


#optimize hyperparameters. uses the data of the current player (set in the loop below)
def objective_xgboost(space):
    
    if space["grow_policy"] == 0 or space["grow_policy"] == 1:
        space["grow_policy"] = GROW_POLICY[space["grow_policy"]]
    
    #fit on the first half and predict the second, and vice versa. penalize slow models
    return two_fold_loss(space, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds,
                         check_features, unknown_features, seconds_per_loss_unit=25)

               
import requests

#get statistics of all players
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
statistics = r.json()

elements_df = pd.DataFrame(statistics['elements'])


names = elements_df.first_name + ' ' + elements_df.second_name
        

with open(r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl", 'rb') as file:
    original_data = pickle.load(file)                

selected = original_data["minutes"] >= 60
train_data = original_data.loc[selected].copy()


#included for all windows, but not current
temporal_features = list(TEMPORAL_FEATURES)


#the non digit version of these features will be removed
unknown_features = list(UNKNOWN_FEATURES)

train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)          


#get one index for each match. players of both teams in a match get the same index
match_ind = build_match_index(train_X)
                
train_X = train_X.drop(['kickoff_time'], axis=1)  




unique_names = train_X.name.unique()

hyper_vals = {'max_depth': [],
            'min_split_loss': [], #log?
            'reg_lambda' : [],
            'reg_alpha': [],
            'min_child_weight' : [],
            'learning_rate': [],
            'early_stopping_rounds': [],
            'n_estimators': [],
            'max_delta_step': [],
            'max_leaves': [],
            'max_bin':  [],
            'temporal_window': [],
            'losses': [],
            'num_matches': [],
        }


train_data['local_predictions'] = np.nan
train_data['local_const_predictions'] = np.nan

original_data['local_predictions'] = np.nan
original_data['local_const_predictions'] = np.nan

last_hyperparams = []

unlikely_values = 0

#train local models
for ind, name in enumerate(unique_names[::-1]):
    
    if ind > 0:
        if best_hyperparams:
            last_hyperparams = best_hyperparams.copy()
    
    
    
    model_path = rf"\\platon.uio.no\med-imb-u1\jorgels\fantasy\local_models\{name}.sav"
    
    num_matches_train = -1
    sel_name = train_X.name == name
    
    print(ind, '/', len(unique_names), name, sum(sel_name))
    
    if optimize_mode:
        search_hyperparams = True
    else:
        search_hyperparams = False
    
    best_hyperparams = []
    old_hyperparams = []
    new_hyperparams = []
    tested_hyperparams = []
    
    #start from an empty summary. otherwise the summary of the previous player is used if loading fails
    summary = {}
    
    if os.path.isfile(model_path):
        try:
            print('Loading model')
            with open(model_path, "rb") as f:
                summary = pickle.load(f)
                
            loading_fail = False
            
            
            # new_hyperparams = {}
            # for field, val in summary["tested_hyperparams"].items():
                
            #     if val == 0 or val==1:
            #         new_hyperparams[field] = int(val)
            #     else:
            #         new_hyperparams[field] = val
                    
            # old_hyperparams = {}
            # for field, val in summary["hyperparams"].items():
                
            #     if val == 0 or val==1:
            #         old_hyperparams[field] = int(val)
            #     else:
            #         old_hyperparams[field] = val
                
            #only compare hyperparams if there are new data
            #do not optimize if not more matches
            try:
                new_hyperparams = summary["tested_hyperparams"]
                try:
                    a = new_hyperparams["num_matches"]
                except:
                    new_hyperparams["num_matches"] = sum(sel_name)
            except:
                new_hyperparams = []


            try:
                old_hyperparams = summary["hyperparams"]
                try:
                    a = old_hyperparams["num_matches"]
                except:
                    old_hyperparams["num_matches"] = sum(sel_name)
                    
            except:
                old_hyperparams = []

            
            #if we have both
            if old_hyperparams and new_hyperparams:
                if (old_hyperparams["num_matches"] == sum(sel_name)) and (new_hyperparams["num_matches"] == sum(sel_name)):
                    print('No new data for old and new hyperparameters. Use new')
                    old_hyperparams = []
                    best_hyperparams = new_hyperparams
                    tested_hyperparams = []
                    
                elif new_hyperparams["num_matches"] == sum(sel_name):
                    print('No new data for new hyperparameters. Use old')
                    new_hyperparams = []
                    best_hyperparams = old_hyperparams
                    tested_hyperparams = new_hyperparams
            #if we have only old
            elif old_hyperparams:
                new_hyperparams = []
                best_hyperparams = old_hyperparams
                tested_hyperparams = []
            #if we have only new
            else:
                new_hyperparams = []
                best_hyperparams = []
                tested_hyperparams = []
                search_hyperparams = True
                    
                    

            
            # #delete later:
            # try:
            #     summary["hyperparams"]["num_matches"] = summary["num_matches"]
            #     num_matches_train = summary["hyperparams"]["num_matches"]
            # except:
            #     num_matches_train = -1
            
            #if not optimizing or no new data (and these players shuld be exlcuded) and we have hyperparams
            if best_hyperparams and (not optimize_mode or (exclude_frozen_players and ((sum(sel_name) == num_matches_train) or (name not in names)))):

                print('Do not search')                        
                search_hyperparams = False
            
        except:
            loading_fail = True
            search_hyperparams = True
            
    else:
        loading_fail = True
        search_hyperparams = True
                
    # Get the 80% of the first matches every season...
    train_copy = train_X.loc[sel_name].copy()
    
    # Identify categorical columns
    categorical_columns = train_copy.select_dtypes(['category']).columns

    # Reset categories for each categorical column
    for column in categorical_columns:
        train_copy[column] = train_copy[column].cat.remove_unused_categories()
        
    empty_category_cols = [col for col in train_copy.select_dtypes(include=["category"]).columns
       if len(train_copy[col].cat.categories) == 0]
    
    train_copy = train_copy.drop(columns=empty_category_cols)
        
        
        
    train_copy = train_copy.reset_index(drop=True)

    #the first half of the matches of every season are used for cv, the second half for validation
    cvs_mask, vals_mask = season_fit_eval_masks(train_copy['season'], match_ind[sel_name], 0.5)
    cvs_mask = pd.Series(cvs_mask)
    vals_mask = pd.Series(vals_mask)
    
    #set the split for later use
    subset_idx = train_data.index[sel_name]        # index of selected rows
    rows_to_update = subset_idx[cvs_mask]  
    train_data.loc[rows_to_update, 'split'] = 1
    rows_to_update = subset_idx[vals_mask]
    train_data.loc[rows_to_update, 'split'] = 0
    
    #at least one train and val
    if sum(sel_name) < 2:
        if sum(sel_name) > 0:
            summary = {'model': [], 'train_features': [], 'hyperparams': [], "local_constant": np.mean(train_y.loc[sel_name])}#, 'all_rows': original_df}
            pickle.dump(summary, open(model_path, 'wb'))
            
        print('Skip name. Only a few matches:', sum(sel_name))
        continue
    
    
    
    cv_X = train_copy.loc[cvs_mask.values].copy()
    val_X =  train_copy.loc[vals_mask.values].copy()
    cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
    val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()
    
    #make sure all categories in val_x and cv_x are present in the player data
    val_X = mask_unseen_categories(val_X, train_copy)
    cv_X = mask_unseen_categories(cv_X, train_copy)
    
    #match index of the two halves
    cv_match_inds = match_ind[sel_name][cvs_mask]
    val_match_inds = match_ind[sel_name][vals_mask]

    mean_cv = np.mean(cv_y)
    
    subset_idx = train_data.index[sel_name]        # index of selected rows
    rows_to_update = subset_idx[vals_mask]         # further filtered rows
    train_data.loc[rows_to_update, 'local_const_predictions'] = mean_cv
    original_data.loc[rows_to_update, 'local_const_predictions'] = mean_cv
    
    mean_val = np.mean(val_y)
    #train_error = np.mean(np.abs((cv_y - mean_cv)**2))
    average_error2 = np.mean(np.abs((cv_y - mean_val)**2))
    
    rows_to_update = subset_idx[cvs_mask]         # further filtered rows
    train_data.loc[rows_to_update, 'local_const_predictions'] = mean_val
    original_data.loc[rows_to_update, 'local_const_predictions'] = mean_val
    
    #min_eval_fraction = 1/(len(unique_integers) * 0.80)#len(np.unique(cv_stratify))/cv_X.shape[0]
    #we need at least one match every season
    
    _, matches_season1 = np.unique(cv_X.season, return_counts=True)
    _, matches_season2 = np.unique(val_X.season, return_counts=True)
    
    if len(matches_season1) == 0  or len(matches_season2) == 0:
        print('Skip name. Only a few matches:', sum(sel_name))
        
        #run one leave out 
        subset_idx = train_data.index[sel_name]        # index of selected rows
        
        for s_ind in range(len(subset_idx)):
            loo_points = np.mean(train_y[subset_idx[:s_ind].to_list() + subset_idx[s_ind+1:].to_list()])
                        
            rows_to_update = subset_idx[s_ind]         # further filtered rows
            train_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
            original_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
            
            summary = {'model': [], 'train_features': [], 'hyperparams': [], "local_constant": np.mean(train_y.loc[sel_name])}#, 'all_rows': original_df}
            pickle.dump(summary, open(model_path, 'wb'))
        
        continue
    
    else:    
    
        min_eval_fraction = 1/min([max(matches_season1), max(matches_season2)])
        
        #at least four matches to make a model...
        if max(matches_season1) <= 3 or max(matches_season2) <= 3:
            
            #run one leave out 
            subset_idx = train_data.index[sel_name]        # index of selected rows
            
            for s_ind in range(len(subset_idx)):
                loo_points = np.mean(train_y[subset_idx[:s_ind].to_list() + subset_idx[s_ind+1:].to_list()])
                            
                rows_to_update = subset_idx[s_ind]         # further filtered rows
                train_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
                original_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
                
            summary = {'model': [], 'train_features': [], 'hyperparams': [], "local_constant": np.mean(train_y.loc[sel_name])}#, 'all_rows': original_df}
            pickle.dump(summary, open(model_path, 'wb'))
                
                
            print('Skip name. Only a few matches:', sum(sel_name))
            continue
    
    
    
    individual_features = list(LOCAL_FEATURES)
    
    #all_features = ['total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'ict_index',  'influence', 'threat', 'creativity', 'bps', 'expected_goals', 'expected_assists',
         #   'expected_goals_conceded', 'defcon', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'own_difficulty'] #, 'difficulty']
    
    
    keep_cols = []
    #remove the features not needed
    for feat in cv_X.keys():
        no_digit_feat = re.sub(r'\d+', '', feat)
        
        if no_digit_feat in individual_features:
            keep_cols.append(feat)

    cv_X = cv_X[keep_cols]
    val_X = val_X[keep_cols]
    train_copy = train_copy[keep_cols]
    
    # #include feature search in the hyperparams
    # check_features = ['transfers_in', 'transfers_out', 'minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
    #         'total_points', 'expected_goals', 'expected_assists', 'points_per_played_game', 'was_home', 'season',
    #         'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon', 'name', 'points_per_game', 'string_opp_team', 'own_difficulty', 'opp_difficulty'] #, 'difficulty']
    
    check_features = individual_features
    #do_remove_features= ['names', 'points_per_game', 'points_per_played_game', 'season']   
    
    temporal_window = len(train_copy)
    if temporal_window > max_temporal_window:
        temporal_window = max_temporal_window
        
        
    space={'max_depth': hp.qloguniform("max_depth", np.log(1), np.log(90), 1), 
            'min_split_loss': hp.loguniform('min_split_loss', np.log(0.001), np.log(20)), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 650),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 30),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 15),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(5.5)),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 500, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.51),
            'n_estimators': hp.quniform('n_estimators', 2, 300000, 1), #up
            'max_delta_step': hp.uniform('max_delta_step', 0, 10500),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1
            'max_leaves': hp.quniform('max_leaves', 0, 2500, 1), #up
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(40), 1),
            'temporal_window': hp.quniform('temporal_window', 1, temporal_window, 1),
        }

    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [False, True])
    
    #marks hyperparameters where the feature flags work (see normalize_feature_flags)
    space['feature_flags_v2'] = hp.choice('feature_flags_v2', [True])
        
        
    #hyperparameters stored before the feature flag fix keep all features
    old_hyperparams = normalize_feature_flags(old_hyperparams, check_features)
    new_hyperparams = normalize_feature_flags(new_hyperparams, check_features)
    best_hyperparams = normalize_feature_flags(best_hyperparams, check_features)
    last_hyperparams = normalize_feature_flags(last_hyperparams, check_features)
        
    if search_hyperparams:
        
        if ind == 0:
            trials = Trials() 
            
        else:
            if last_hyperparams:
                if last_hyperparams['eval_fraction'] < min_eval_fraction:
                    last_hyperparams['eval_fraction'] = min_eval_fraction
                last_hyperparams["num_matches"] = sum(sel_name)
                
                if last_hyperparams["grow_policy"] == 'lossguide':
                    last_hyperparams["grow_policy"] = 1
                elif last_hyperparams["grow_policy"] == 'depthwise':
                    last_hyperparams["grow_policy"] = 0
                    
                trials = generate_trials_to_calculate([last_hyperparams])
                
            else:
                trials = Trials() 
                print('had to generate new trials')
            
        #trials = Trials() 
        print('Search for hyperparameters')

        loss = np.inf
        tid = len(trials)
        
        while loss > 60 and tid < 200:
    
            #optmimize hyperparameters. use all training data
            new_hyperparams = fmin(fn = objective_xgboost,
                            space = space,
                            algo = atpe.suggest,
                            trials = trials,
                            max_evals=99999999,
                            early_stop_fn=no_progress_loss(40)
                            )
            
            loss = trials.best_trial['result']['loss']
            tid = len(trials)
            
            new_hyperparams["losses"] = trials.best_trial['result']['loss']
            new_hyperparams["num_matches"] = sum(sel_name)
                    
            if optimize_mode:
                best_hyperparams = new_hyperparams
            elif old_hyperparams:
                best_hyperparams = old_hyperparams
            else:
                best_hyperparams = new_hyperparams
    
    else:
        new_loss = np.inf
        old_loss = np.inf
        
        if new_hyperparams and old_hyperparams:
            new_loss = objective_xgboost(new_hyperparams)["loss"]
            old_loss = objective_xgboost(old_hyperparams)["loss"]
            print('Old', old_loss)
            print('New', new_loss)
        elif old_hyperparams:
            old_loss = 0
        elif new_hyperparams:
            new_loss = 0
            
            
        if new_loss < old_loss:
            best_hyperparams = new_hyperparams
            print('Use new')
        else:
            best_hyperparams = old_hyperparams
            print('Use old')
    
    for k in best_hyperparams.keys():
        if k in hyper_vals:
            hyper_vals[k].append(best_hyperparams[k])

    #do not continue if don't have new data
    if optimize_mode:
        summary['tested_hyperparams'] = new_hyperparams
        pickle.dump(summary, open(model_path, 'wb'))
        continue
    else:
        tested_hyperparams = new_hyperparams
    
    

        
        
    print('Predict for hierarchy')
    
    if best_hyperparams['grow_policy'] == 0 or best_hyperparams['grow_policy'] == 1:
        best_hyperparams['grow_policy'] = GROW_POLICY[best_hyperparams['grow_policy']]
        
        
    #get cross-valiated local predictions
    subset_idx = train_data.index[sel_name]        # index of selected rows
    
    for val_pred, y1, y2, test_is_val in cross_fit_predictions(best_hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds, check_features, unknown_features):
        
        val_error = mean_squared_error(y2,  val_pred)
        
        #remove values that are outside of the training range
        unlikely_values += remove_out_of_range(val_pred, y1)
        
        random_error = np.mean(np.abs((y2 - np.mean(y1))**2))
        
        rows_to_update = subset_idx[vals_mask if test_is_val else cvs_mask]
        
        train_data.loc[rows_to_update, 'local_predictions'] = val_pred
        original_data.loc[rows_to_update, 'local_predictions'] = val_pred
        
        print('done', random_error-val_error)
    
    
    player_predictions = train_data[sel_name]
    
    #store the model for the predictions...
    #for the players who are not part of current season we don't need the models...
    #might want to skip for first round(some players might return)
        
    #if not active players
    if not loading_fail and not name in names.values:
        summary = {'model': [], 'train_features': [], 'hyperparams': best_hyperparams, "local_constant": [], "predictions": player_predictions}#, 'all_rows': original_df}
        pickle.dump(summary, open(model_path, 'wb'))
        continue    
    
    #skip if not new hyperparams or data has not changed
    if search_hyperparams or not sum(sel_name) == best_hyperparams["num_matches"]:
        
        print('Train for predictions', name, sum(sel_name))

        X1 = train_copy.reset_index(drop=True)
        
        columns = select_columns(X1, best_hyperparams, check_features, unknown_features)
        
        model = fit_xgboost(X1, train_y[sel_name], match_ind[sel_name], best_hyperparams, columns, mask_eval_categories=True)
                    
        
        summary = {'model': model, 'train_features': training_metadata(X1[columns]), 'hyperparams': best_hyperparams, 'tested_hyperparams': tested_hyperparams, "local_constant": np.mean(train_y.loc[sel_name]), "predictions": player_predictions}#, 'all_rows': original_df}
    
        pickle.dump(summary, open(model_path, 'wb'))
    
    
        

for k in hyper_vals:
    plt.figure()
    #print(k, np.max(np.array(hyper_vals[k])[selected]))
    plt.hist(np.array(hyper_vals[k]))
    plt.title(k)
    
plt.show()

local_error = np.nanmean((train_y - train_data['local_predictions'])**2)
print('Local error:', local_error)
        
train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\model_data.pkl')  # Set index=False to not include row indices


print('Unlikely values set to nan:', unlikely_values, train_data.shape[0])

