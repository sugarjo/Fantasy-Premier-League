import os
import re
import pickle
import random
import math
import time

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

import xgboost as xgb


from hyperopt import STATUS_OK, Trials, fmin, hp, atpe, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate

from difflib import SequenceMatcher

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"


try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


update_models = True
update_hyperparams = False


current_season = '2025-26'

# Function to correct string_team based on the majority
def correct_string_team(group):
    # Count occurrences of each string_team
    counts = group['string_team'].value_counts()
    majority_team = counts.idxmax()  # Get the majority string_team
    # Replace incorrect string_team with the majority_team
    group['string_team'] = majority_team
    return group

def sequence_matcher_similarity(s1, s2):
    similarity = SequenceMatcher(None, ' '.join(sorted(s1.split())), ' '.join(sorted(s2.split()))).ratio()
    first_name_similarity = SequenceMatcher(None, s1.split()[0], s2.split()[0]).ratio()
    if len(s1.split()) > 1 and len(s2.split()) > 1:
        second_name_similarity = SequenceMatcher(None, s1.split()[1], s2.split()[1]).ratio()
    else:
        second_name_similarity = np.nan

    return similarity, first_name_similarity, second_name_similarity


def clean_string(input_string):
    # Replace underscores with spaces
    cleaned_string = input_string.replace('_', ' ')
    cleaned_string = input_string.replace("'", "")
    # Remove all numbers
    cleaned_string = re.sub(r'\d+', '', cleaned_string)
    return cleaned_string.strip()  # Optional: strip leading/trailing spaces

def custom_metric(pred_y, dtrain):

    # Targets
    y = dtrain.get_label()

    mse = mean_squared_error(y, pred_y)

    return 'MSE60', mse

def custom_objective(pred_y, dtrain):
    #https://stackoverflow.com/questions/59683944/creating-a-custom-objective-function-in-for-xgboost-xgbregressor
    
    # Targets
    y = dtrain.get_label()

    errors = pred_y - y
    #grad = 0.5 * errors
    grad = 2 * errors
    hess = np.zeros_like(pred_y) + 2
    #hess = np.ones_like(pred_y)

    return grad, hess


def quantile_objective(pred_y, dtrain):

    y = dtrain.get_label()

    q = 0.5
    
    errors = pred_y - y
    
    #multiply by two to make it comparable to the custom objective (gradients are ~half of those)
    grad = 4 * (np.where(errors > 0, (1-q) * errors, q * errors))
    hess = np.zeros_like(pred_y) + 2
    
    return grad, hess


# Define a function to check if the column name meets the criteria
def should_keep_column(column_name, threshold):
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

#optimize hyperparameters
def objective_xgboost(space):
    
    
    
   #print(space)
    
    
    space["grow_policy"] = grow_policy[space["grow_policy"]]
    
    pars = {
        'max_depth': int(space['max_depth']),
        'min_split_loss': space['min_split_loss'],
        'reg_lambda': space['reg_lambda'],
        'reg_alpha': space['reg_alpha'],
        'min_child_weight': int(space['min_child_weight']),
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'colsample_bylevel': space['colsample_bylevel'],
        'colsample_bynode': space['colsample_bynode'],
        'max_delta_step': space['max_delta_step'],
        'grow_policy': space['grow_policy'],
        'max_leaves': int(space['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(space['max_bin']),
        'disable_default_eval_metric': 1
        }
    
    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(space['temporal_window'])
    
    

    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    objective_X = cv_X[columns_to_keep]    
    HO_X = val_X[columns_to_keep].copy()
      
    #remove features that are listed in space
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                HO_X = HO_X[columns_to_keep]
                
    #remove features that are unknown
    columns_to_keep = []
    for feat in objective_X.keys():
        keep = True
        for uk in unknown_features:
            if feat == uk:
                keep = False
        
        if keep:
            columns_to_keep.append(feat)
            
    objective_X = objective_X[columns_to_keep]
    HO_X = HO_X[columns_to_keep]
                
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name][cvs_mask])

    
    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y.loc[fits_mask.values].copy()
    eval_y = cv_y.loc[evals_mask.values].copy()

    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)
    dval_objective = xgb.DMatrix(data=HO_X, label=val_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]
    
    start = time.time()           # seconds since epoch (float)

    model = xgb.train(
    params=pars,
    num_boost_round=int(space['n_estimators']),
    early_stopping_rounds= int(space['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )
    

    val_pred = model.predict(dval_objective)
    
    val_error = mean_squared_error(val_y,  val_pred)
    #print('done1', val_error)

    cv_loss  = objective_xgboost_custom(space, val_X, val_y, cv_X, cv_y, vals_mask)
    cv_error = cv_loss['loss']
    #print('done2', cv_error)  
    
    end = time.time()
    elapsed = end - start

    total_error = np.mean([val_error, cv_error]) + elapsed/25
    
    #print(cv_loss["status"])
        
    #print(total_error, type(total_error))        

    return {'loss': total_error, 'status': STATUS_OK }

def objective_xgboost_custom(space, cv_X, cv_y, val_X, val_y, cvs_mask):
    
    #print(space)
    
    #space["grow_policy"] = grow_policy[space["grow_policy"]]
    
    pars = {
        'max_depth': int(space['max_depth']),
        'min_split_loss': space['min_split_loss'],
        'reg_lambda': space['reg_lambda'],
        'reg_alpha': space['reg_alpha'],
        'min_child_weight': int(space['min_child_weight']),
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'colsample_bylevel': space['colsample_bylevel'],
        'colsample_bynode': space['colsample_bynode'],
        'max_delta_step': space['max_delta_step'],
        'grow_policy': space['grow_policy'],
        'max_leaves': int(space['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(space['max_bin']),
        'disable_default_eval_metric': 1
        }
    
    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(space['temporal_window'])
    
    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    objective_X = cv_X[columns_to_keep]    
    HO_X = val_X[columns_to_keep].copy()
      
    #remove features that are listed in space
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                HO_X = HO_X[columns_to_keep]
                
    #remove features that are unknown
    columns_to_keep = []
    for feat in objective_X.keys():
        keep = True
        for uk in unknown_features:
            if feat == uk:
                keep = False
        
        if keep:
            columns_to_keep.append(feat)
            
    objective_X = objective_X[columns_to_keep]
    HO_X = HO_X[columns_to_keep]
                
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name][cvs_mask])

    
    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y.loc[fits_mask.values].copy()
    eval_y = cv_y.loc[evals_mask.values].copy()               
                
    
    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)
    dval_objective = xgb.DMatrix(data=HO_X, label=val_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]

    model = xgb.train(
    params=pars,
    num_boost_round=int(space['n_estimators']),
    early_stopping_rounds= int(space['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )

    

    val_pred = model.predict(dval_objective)
    
    val_error = mean_squared_error(val_y,  val_pred)
    # print('done1', val_error)
    
    #cv_loss = objective_xgboost(space, cv_X=val_X, cv_y=val_y, val_X=cv_X, val_y=cv_y, cvs_mask=vals_mask)
    #cv_error = cv_loss['loss']
    #print('done2', cv_error)
    
    
    #total_error = np.mean([val_error, cv_error])
   

    return {'loss': val_error, 'status': STATUS_OK }

               


with open(r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl", 'rb') as file:
    original_data = pickle.load(file)                

selected = original_data["minutes"] >= 60
train_data = original_data.loc[selected].copy()


#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon']


#the non digit version of these features will be removed
unknown_features = ['minutes', 'ict_index', 'total_points', 'own_team_points', 'own_element_points', 'defcon']

train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)          

# Identify categorical columns
categorical_columns = train_X.select_dtypes(['category']).columns

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()

#get one index for each match
match_ind = pd.factorize(
    train_X[['string_team', 'was_home', 'string_opp_team', 'kickoff_time']]
    .apply(lambda row: '-'.join(row.astype(str)), axis=1)
)[0]
   
#get 20% of the last matches for each season (to avoid leakage of data from points per game etc)
# Step 1: Get unique integers using a set
unique_integers = list(set(match_ind))

for ind in unique_integers:
    matches = np.where(match_ind == ind)[0]
    
    if len(matches)>0:
    
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
                
train_X = train_X.drop(['kickoff_time'], axis=1)

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()
    

# define a helper that returns the first 80% (by first-appearance order) of unique match_ind
def first_Xpct_unique(seq, X):
    # preserve order of first appearance
    seen = {}
    uniq_in_order = []
    for v in seq:
        if v not in seen:
            seen[v] = True
            uniq_in_order.append(v)
    n_keep = math.ceil(X * len(uniq_in_order))  # use ceil to keep at least one for small groups
    return set(uniq_in_order[:n_keep])




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

trials = Trials() 

#train local models
for ind, name in enumerate(unique_names):
    
    print(ind, '/', len(unique_names), name)
    
    model_path = rf"\\platon.uio.no\med-imb-u1\jorgels\fantasy\local_models\{name}.sav"
    
    if not update_models or not update_hyperparams:
        if os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                summary = pickle.load(f)
                
            best_hyperparams = summary["hyperparams"]
            
            for k in best_hyperparams.keys():
                if k in hyper_vals:
                    hyper_vals[k].append(best_hyperparams[k])
        else:
            ('No model')
            continue
                    
    if not update_models:
        continue            
                    
    sel_name = train_X.name == name
    
    # Get the 80% of the first matches every season...
    train_copy = train_X.loc[sel_name].copy()
    train_copy = train_copy.reset_index(drop=True)
    train_copy['match_ind'] = pd.Series(match_ind[sel_name])
    match_ind_df = pd.Series(match_ind[sel_name]) 

    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (train_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 0.5))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    train_sample = list(set().union(*season_selection.values()))
    
    
    # Step 2: Calculate 20% of the unique integers
    
    # unique_integers = list(set(match_ind))
    
    # num_to_select = max(1, int(len(unique_integers) * 0.80))  # Ensure at least one is selected
    
    # Step 3: Randomly select 20% of the unique integers
    random.seed(42)
    
    # train_sample = random.sample(unique_integers, num_to_select)
    
    # vals = [x not in train_sample for x in match_ind_df]
    # cvs = [x in train_sample for x in match_ind_df]
    
    cvs_mask = pd.Series(match_ind_df).isin(train_sample)  # Mask for cross-validation sample
    vals_mask = ~cvs_mask  # Mask for validation, simply the inverse of cvs_mask
    
    #set the split for later use


    subset_idx = train_data.index[sel_name]        # index of selected rows
    rows_to_update = subset_idx[cvs_mask]  
    train_data.loc[rows_to_update, 'split'] = 1
    rows_to_update = subset_idx[vals_mask]
    train_data.loc[rows_to_update, 'split'] = 0
    
    #at least one train and val
    if sum(sel_name) < 2:
        continue
    
    cvs_match_integers = list(set(match_ind_df[cvs_mask]))
    
    
    cv_X = train_copy.loc[cvs_mask.values].copy()
    val_X =  train_copy.loc[vals_mask.values].copy()
    cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
    val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()
    
    #make sure all categories in val_x is present in cv_x
    for column in val_X.columns:
        if isinstance(val_X[column].dtype, pd.CategoricalDtype):
            # Get the values in the current column of val_X
            val_values = val_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(train_copy[column])
            
            # Set values that are not present in cv_X[column] to NaN
            val_X.loc[~mask, column] = np.nan
            
    #make sure all categories in val_x is present in cv_x
    for column in cv_X.columns:
        if isinstance(cv_X[column].dtype, pd.CategoricalDtype):
            # Get the values in the current column of val_X
            val_values = cv_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(train_copy[column])
            
            # Set values that are not present in cv_X[column] to NaN
            cv_X.loc[~mask, column] = np.nan
            
    
        
        
        
    mean_cv = np.mean(cv_y)
    
    subset_idx = train_data.index[sel_name]        # index of selected rows
    rows_to_update = subset_idx[vals_mask]         # further filtered rows
    train_data.loc[rows_to_update, 'local_const_predictions'] = mean_cv
    
    mean_val = np.mean(val_y)
    #train_error = np.mean(np.abs((cv_y - mean_cv)**2))
    average_error2 = np.mean(np.abs((cv_y - mean_val)**2))
    
    rows_to_update = subset_idx[cvs_mask]         # further filtered rows
    train_data.loc[rows_to_update, 'local_const_predictions'] = mean_val
    
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
        
        continue
    
    else:    
    
        min_eval_fraction = 1/min([max(matches_season1), max(matches_season2)])
        
        if max(matches_season1) <= 1 or max(matches_season2) <= 1:
            
            #run one leave out 
            subset_idx = train_data.index[sel_name]        # index of selected rows
            
            for s_ind in range(len(subset_idx)):
                loo_points = np.mean(train_y[subset_idx[:s_ind].to_list() + subset_idx[s_ind+1:].to_list()])
                            
                rows_to_update = subset_idx[s_ind]         # further filtered rows
                train_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
                
                
            print('Skip name. Only a few matches:', sum(sel_name))
            continue
    
    grow_policy = ['depthwise', 'lossguide']
    
    
    individual_features = ['season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_difficulty', 'ict_index', 'transfers_in', 'transfers_out', 'expected_goals', 'expected_assists',
            'expected_goals_conceded', 'defcon', 'own_element_points', 'season']
    
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
    

        
    space={'max_depth': hp.qloguniform("max_depth", 1, np.log(100), 1), 
            'min_split_loss': hp.loguniform('min_split_loss', 0, np.log(40)), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 550),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 70),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 70),
            'learning_rate': hp.loguniform('learning_rate', 0, np.log(7)),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 800, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.51),
            'n_estimators': hp.quniform('n_estimators', 2, 90000, 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 3600),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1
            'max_leaves': hp.quniform('max_leaves', 0, 800, 1),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(100), 1),
            'temporal_window': hp.quniform('temporal_window', 1, 20, 1),
        }

    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [True, False]), #111
        
    if ind == 0:
        trials = Trials() 
        
    else:
        if best_hyperparams['eval_fraction'] < min_eval_fraction:
            best_hyperparams['eval_fraction'] = min_eval_fraction
        
        trials = generate_trials_to_calculate([best_hyperparams])
        
    #trials = Trials() 
    if update_hyperparams:
        print('Search for hyperparameters')

        loss = np.inf
        tid = 0
        
        while loss > 60 and tid < 200:
    
            #optmimize hyperparameters. use all training data
            best_hyperparams = fmin(fn = objective_xgboost,
                            space = space,
                            algo = atpe.suggest,
                            trials = trials,
                            max_evals=99999999,
                            early_stop_fn=no_progress_loss(40)
                            )
            
            loss = trials.best_trial['result']['loss']
            tid = len(trials)
            
        for k in best_hyperparams.keys():
            if k in hyper_vals:
                hyper_vals[k].append(best_hyperparams[k])
        hyper_vals["losses"].append(trials.best_trial['result']['loss'])
        hyper_vals["num_matches"].append(sum(sel_name))
    
    print('Train')
    
    pars = {
        'max_depth': int(best_hyperparams['max_depth']),
        'min_split_loss': best_hyperparams['min_split_loss'],
        'reg_lambda': best_hyperparams['reg_lambda'],
        'reg_alpha': best_hyperparams['reg_alpha'],
        'min_child_weight': int(best_hyperparams['min_child_weight']),
        'learning_rate': best_hyperparams['learning_rate'],
        'subsample': best_hyperparams['subsample'],
        'colsample_bytree': best_hyperparams['colsample_bytree'],
        'colsample_bylevel': best_hyperparams['colsample_bylevel'],
        'colsample_bynode': best_hyperparams['colsample_bynode'],
        'max_delta_step': best_hyperparams['max_delta_step'],
        'grow_policy': grow_policy[best_hyperparams['grow_policy']],
        'max_leaves': int(best_hyperparams['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(best_hyperparams['max_bin']),
        'disable_default_eval_metric': 1
        }
        

    #get cross-valiated local predictions
    for X1, y1, X2, y2, mask1, mask2 in zip((cv_X, val_X), (cv_y, val_y), (val_X, cv_X), (val_y, cv_y), (cvs_mask, vals_mask), (vals_mask, cvs_mask)):

        # Get the 80% of the first matches every season...
        X1 = X1.reset_index(drop=True)
        
        #remove weaks that we don't need.
        # Define the threshold
        threshold = int(best_hyperparams['temporal_window'])
        
        # Filter the columns based on the defined function
        columns_to_keep = [col for col in X1.columns if should_keep_column(col, threshold)]
        objective_X = X1[columns_to_keep]      
          
        #remove features
        for feat in check_features:
            if feat in space.keys():
                #if remove
                if not space[feat]:     
                    columns_to_keep = []
                    for col in objective_X.columns:
                        if col == feat: # and col in do_remove_features:
                            continue
                        #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                        if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                            columns_to_keep.append(col)
                        
                    objective_X = objective_X[columns_to_keep]
                    
        # Get the 80% of the first matches every season...
        objective_copy = objective_X.copy()
        objective_copy = objective_copy.reset_index(drop=True)
        objective_copy['match_ind'] = pd.Series(match_ind[sel_name][mask1])

        
        # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
        season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                              .agg(lambda s: first_Xpct_unique(s.tolist(), 1-best_hyperparams['eval_fraction']))
                              .to_dict())
        
        # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
        # option 1: Unique across all seasons:
        fit_sample = list(set().union(*season_selection.values()))
            
        fits_mask =  pd.Series(match_ind_df[mask1]).isin(fit_sample)  # Mask for cross-validation sample
        evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
        
        fit_X = objective_X.iloc[fits_mask.values].copy()
        eval_X =  objective_X.loc[evals_mask.values].copy()
        fit_y =  y1.loc[fits_mask.values].copy()
        eval_y = y1.loc[evals_mask.values].copy()

        #make sure all categories in val_x is present in cv_x
        for column in eval_X.columns:
            if isinstance(eval_X[column].dtype, pd.CategoricalDtype):
                # Get the values in the current column of val_X
                val_values = eval_X[column]
                
                # Check which values are present in the corresponding column of cv_X
                mask = val_values.isin(fit_X[column])
                
                # Set values that are not present in cv_X[column] to NaN
                eval_X.loc[~mask, column] = np.nan
        
        dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
        deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)

        evals = [(dfit, 'train'), (deval, 'eval')]
        
        model = xgb.train(
        params=pars,
        num_boost_round=int(best_hyperparams['n_estimators']),
        early_stopping_rounds= int(best_hyperparams['early_stopping_rounds']),
        dtrain=dfit,
        evals=evals,
        custom_metric=custom_metric,
        obj=custom_objective,
        verbose_eval=False  # Set to True if you want to see detailed logging
            )
        

        objective_val_X = X2[columns_to_keep]
        dval_objective = xgb.DMatrix(data=objective_val_X, label=y2, enable_categorical=True)

        val_pred = model.predict(dval_objective)
        
        val_error = mean_squared_error(y2,  val_pred)
        #print('done1', val_error)
        
        mean_y = np.mean(y1)

        random_error = np.mean(np.abs((y2 - mean_y)**2))
        
        
        subset_idx = train_data.index[sel_name]        # index of selected rows
        rows_to_update = subset_idx[mask2]   
        
        original_data.loc[rows_to_update, 'local_predictions'] = val_pred

        #print(val_pred)        
        print('done', random_error-val_error)
        

     
    #store the model for the predictions...
    #for the players who are not part of current season we don't need the models...
    #if not current_season in np.unique(train_copy.season):
    if False:
            summary = {'model': [], 'train_features': [], 'hyperparams': best_hyperparams}#, 'all_rows': original_df}
            
            #pickle.dump(summary, open(model_path, 'wb'))
    else:
            
        # Get the 80% of the first matches every season...
            X1 = train_copy.reset_index(drop=True)
            
            #remove weaks that we don't need.
            # Define the threshold
            threshold = int(best_hyperparams['temporal_window'])
            
            # Filter the columns based on the defined function
            columns_to_keep = [col for col in X1.columns if should_keep_column(col, threshold)]
            objective_X = X1[columns_to_keep]      
              
            #remove features
            for feat in check_features:
                if feat in space.keys():
                    #if remove
                    if not space[feat]:     
                        columns_to_keep = []
                        for col in objective_X.columns:
                            if col == feat: # and col in do_remove_features:
                                continue
                            #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                            if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                                columns_to_keep.append(col)
                            
                        objective_X = objective_X[columns_to_keep]
                        
            # Get the 80% of the first matches every season...
            objective_copy = objective_X.copy()
            objective_copy = objective_copy.reset_index(drop=True)
            objective_copy['match_ind'] = pd.Series(match_ind[sel_name])
        
            
            # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
            season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                                  .agg(lambda s: first_Xpct_unique(s.tolist(), 1-best_hyperparams['eval_fraction']))
                                  .to_dict())
            
            # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
            # option 1: Unique across all seasons:
            fit_sample = list(set().union(*season_selection.values()))
                
            fits_mask =  pd.Series(match_ind_df).isin(fit_sample)  # Mask for cross-validation sample
            evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
            
            fit_X = objective_X.iloc[fits_mask.values].copy()
            eval_X =  objective_X.loc[evals_mask.values].copy()
            fit_y =  train_y[sel_name].loc[fits_mask.values].copy()
            eval_y = train_y[sel_name].loc[evals_mask.values].copy()
        
            #make sure all categories in val_x is present in cv_x
            for column in eval_X.columns:
                if isinstance(eval_X[column].dtype, pd.CategoricalDtype):
                    # Get the values in the current column of val_X
                    val_values = eval_X[column]
                    
                    # Check which values are present in the corresponding column of cv_X
                    mask = val_values.isin(fit_X[column])
                    
                    # Set values that are not present in cv_X[column] to NaN
                    eval_X.loc[~mask, column] = np.nan
            
            dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
            deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)
        
            evals = [(dfit, 'train'), (deval, 'eval')]
            
            model = xgb.train(
            params=pars,
            num_boost_round=int(best_hyperparams['n_estimators']),
            early_stopping_rounds= int(best_hyperparams['early_stopping_rounds']),
            dtrain=dfit,
            evals=evals,
            custom_metric=custom_metric,
            obj=custom_objective,
            verbose_eval=False  # Set to True if you want to see detailed logging
                )
            
            summary = {'model': model, 'train_features': objective_X, 'hyperparams': best_hyperparams}#, 'all_rows': original_df}
        
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


import os
import re
import pickle
import random
import math
import time

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

import xgboost as xgb


from hyperopt import STATUS_OK, Trials, fmin, hp, atpe, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate

from difflib import SequenceMatcher

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'


try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


method = 'xgboost'

season_dfs = []

# Function to correct string_team based on the majority
def correct_string_team(group):
    # Count occurrences of each string_team
    counts = group['string_team'].value_counts()
    majority_team = counts.idxmax()  # Get the majority string_team
    # Replace incorrect string_team with the majority_team
    group['string_team'] = majority_team
    return group

def sequence_matcher_similarity(s1, s2):
    similarity = SequenceMatcher(None, ' '.join(sorted(s1.split())), ' '.join(sorted(s2.split()))).ratio()
    first_name_similarity = SequenceMatcher(None, s1.split()[0], s2.split()[0]).ratio()
    if len(s1.split()) > 1 and len(s2.split()) > 1:
        second_name_similarity = SequenceMatcher(None, s1.split()[1], s2.split()[1]).ratio()
    else:
        second_name_similarity = np.nan

    return similarity, first_name_similarity, second_name_similarity


def clean_string(input_string):
    # Replace underscores with spaces
    cleaned_string = input_string.replace('_', ' ')
    cleaned_string = input_string.replace("'", "")
    # Remove all numbers
    cleaned_string = re.sub(r'\d+', '', cleaned_string)
    return cleaned_string.strip()  # Optional: strip leading/trailing spaces

def custom_metric(pred_y, dtrain):

    # Targets
    y = dtrain.get_label()

    mse = mean_squared_error(y, pred_y)

    return 'MSE60', mse

def custom_objective(pred_y, dtrain):
    #https://stackoverflow.com/questions/59683944/creating-a-custom-objective-function-in-for-xgboost-xgbregressor
    
    # Targets
    y = dtrain.get_label()

    errors = pred_y - y
    #grad = 0.5 * errors
    grad = 2 * errors
    hess = np.zeros_like(pred_y) + 2
    #hess = np.ones_like(pred_y)

    return grad, hess


def quantile_objective(pred_y, dtrain):

    y = dtrain.get_label()

    q = 0.5
    
    errors = pred_y - y
    
    #multiply by two to make it comparable to the custom objective (gradients are ~half of those)
    grad = 4 * (np.where(errors > 0, (1-q) * errors, q * errors))
    hess = np.zeros_like(pred_y) + 2
    
    return grad, hess


# Define a function to check if the column name meets the criteria
def should_keep_column(column_name, threshold):
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

#optimize hyperparameters
def objective_xgboost(space):
    
    
    
    #print(space)
    
    
    space["grow_policy"] = grow_policy[space["grow_policy"]]
    
    pars = {
        'max_depth': int(space['max_depth']),
        'min_split_loss': space['min_split_loss'],
        'reg_lambda': space['reg_lambda'],
        'reg_alpha': space['reg_alpha'],
        'min_child_weight': int(space['min_child_weight']),
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'colsample_bylevel': space['colsample_bylevel'],
        'colsample_bynode': space['colsample_bynode'],
        'max_delta_step': space['max_delta_step'],
        'grow_policy': space['grow_policy'],
        'max_leaves': int(space['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(space['max_bin']),
        'disable_default_eval_metric': 1
        }
    
    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(space['temporal_window'])
    
    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    objective_X = cv_X[columns_to_keep]    
    HO_X = val_X[columns_to_keep].copy()
        
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name][cvs_mask])

    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
    
      
    #remove features that are listed in space
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                HO_X = HO_X[columns_to_keep]
                
    #remove features that are unknown
    columns_to_keep = []
    for feat in objective_X.keys():
        keep = True
        for uk in unknown_features:
            if feat == uk:
                keep = False
        
        if keep:
            columns_to_keep.append(feat)
            
    objective_X = objective_X[columns_to_keep]
    HO_X = HO_X[columns_to_keep]
                
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y.loc[fits_mask.values].copy()
    eval_y = cv_y.loc[evals_mask.values].copy()

    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)
    dval_objective = xgb.DMatrix(data=HO_X, label=val_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]
    
    model = xgb.train(
    params=pars,
    num_boost_round=int(space['n_estimators']),
    early_stopping_rounds= int(space['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )
    

    val_pred = model.predict(dval_objective)
    
    val_error = mean_squared_error(val_y,  val_pred)
    #print('done1', val_error)

    cv_loss  = objective_xgboost_custom(space, val_X, val_y, cv_X, cv_y, vals_mask)
    cv_error = cv_loss['loss']
    #print('done2', cv_error)  
    

    total_error = np.mean([val_error, cv_error])
    
    #print(cv_loss["status"])
        
    #print(total_error, type(total_error))        

    return {'loss': total_error, 'status': STATUS_OK }

def objective_xgboost_custom(space, cv_X, cv_y, val_X, val_y, cvs_mask):
    
    
    pars = {
        'max_depth': int(space['max_depth']),
        'min_split_loss': space['min_split_loss'],
        'reg_lambda': space['reg_lambda'],
        'reg_alpha': space['reg_alpha'],
        'min_child_weight': int(space['min_child_weight']),
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'colsample_bylevel': space['colsample_bylevel'],
        'colsample_bynode': space['colsample_bynode'],
        'max_delta_step': space['max_delta_step'],
        'grow_policy': space['grow_policy'],
        'max_leaves': int(space['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(space['max_bin']),
        'disable_default_eval_metric': 1
        }
    
    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(space['temporal_window'])
    
    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    objective_X = cv_X[columns_to_keep]    
    HO_X = val_X[columns_to_keep].copy()
    
    
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name][cvs_mask])

    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
    
      
    #remove features that are listed in space
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                HO_X = HO_X[columns_to_keep]
                
    #remove features that are unknown
    columns_to_keep = []
    for feat in objective_X.keys():
        keep = True
        for uk in unknown_features:
            if feat == uk:
                keep = False
        
        if keep:
            columns_to_keep.append(feat)
            
    objective_X = objective_X[columns_to_keep]
    HO_X = HO_X[columns_to_keep]
                
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y.loc[fits_mask.values].copy()
    eval_y = cv_y.loc[evals_mask.values].copy()

    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)
    dval_objective = xgb.DMatrix(data=HO_X, label=val_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]

    model = xgb.train(
    params=pars,
    num_boost_round=int(space['n_estimators']),
    early_stopping_rounds= int(space['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )
    
    val_pred = model.predict(dval_objective)
    
    val_error = mean_squared_error(val_y,  val_pred)

    return {'loss': val_error, 'status': STATUS_OK }

               
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
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon']


#the non digit version of these features will be removed
unknown_features = ['minutes', 'ict_index', 'total_points', 'own_team_points', 'own_element_points', 'defcon']



train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)
                

# Identify categorical columns
categorical_columns = train_X.select_dtypes(['category']).columns

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()



#get one index for each match
match_ind = pd.factorize(
    train_X[['string_team', 'was_home', 'string_opp_team', 'kickoff_time']]
    .apply(lambda row: '-'.join(row.astype(str)), axis=1)
)[0]
   
#get 20% of the last matches for each season (to avoid leakage of data from points per game etc)
# Step 1: Get unique integers using a set
unique_integers = list(set(match_ind))

for ind in unique_integers:
    matches = np.where(match_ind == ind)[0]
    
    if len(matches)>0:
    
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
                
train_X = train_X.drop(['kickoff_time'], axis=1)

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()
    

# define a helper that returns the first 80% (by first-appearance order) of unique match_ind
def first_Xpct_unique(seq, X):
    # preserve order of first appearance
    seen = {}
    uniq_in_order = []
    for v in seq:
        if v not in seen:
            seen[v] = True
            uniq_in_order.append(v)
    n_keep = math.ceil(X * len(uniq_in_order))  # use ceil to keep at least one for small groups
    return set(uniq_in_order[:n_keep])


train_data['element_predictions'] = np.nan

trials = Trials() 

#train local models
for pos in range(1,5):
    
    print('Position:', pos)
    
    sel_name = train_X.element_type == pos  
    
    
    # Get the 80% of the first matches every season...
    train_copy = train_X.loc[sel_name].copy()
    train_copy = train_copy.reset_index(drop=True)
    train_copy['match_ind'] = pd.Series(match_ind[sel_name])
    match_ind_df = pd.Series(match_ind[sel_name]) 
   
    # Step 2: Calculate 20% of the unique integers
    
    # unique_integers = list(set(match_ind))
    
    # num_to_select = max(1, int(len(unique_integers) * 0.80))  # Ensure at least one is selected
    
    # Step 3: Randomly select 20% of the unique integers
    random.seed(42)
    
    cvs_mask = train_copy.split == 0  # Mask for cross-validation sample
    #vals_mask =train_copy.split == 1  # Mask for validation, simply the inverse of cvs_mask
    vals_mask = ~cvs_mask       
    
    cv_X = train_copy.loc[cvs_mask.values].copy()
    val_X =  train_copy.loc[vals_mask.values].copy()
    cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
    val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()
    
    #make sure all categories in val_x is present in cv_x
    for column in val_X.columns:
        if isinstance(val_X[column].dtype, pd.CategoricalDtype):
            # Get the values in the current column of val_X
            val_values = val_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(train_copy[column])
            
            # Set values that are not present in cv_X[column] to NaN
            val_X.loc[~mask, column] = np.nan
            
    #make sure all categories in val_x is present in cv_x
    for column in cv_X.columns:
        if isinstance(cv_X[column].dtype, pd.CategoricalDtype):
            # Get the values in the current column of val_X
            val_values = cv_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(train_copy[column])
            
            # Set values that are not present in cv_X[column] to NaN
            cv_X.loc[~mask, column] = np.nan
    
    cv_match_count = np.max(train_copy.loc[cvs_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
    val_match_count = np.max(train_copy.loc[vals_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
    min_eval_fraction = 1/np.min([cv_match_count, val_match_count])  
    
    grow_policy = ['depthwise', 'lossguide']
    
    #individual_features = ['season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_difficulty', 'ict_index', 'transfers_in', 'transfers_out', 'bps'] #, 'difficulty']
    
    element_features = ['ict_index', 'name', 'season', 'total_points', 'own_element_points', 'opp_element_points', 'own_difficulty', 'opp_difficulty', 'transfers_out', 'transfers_in', 'string_opp_team', 'name', 'points_per_played_game', 'points_per_game', 'local_predictions', 'local_const_predictions', 'minutes', 'was_home']

    # all_features = ['element_type', 'season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'ict_index',  'influence', 'threat', 'creativity', 'bps', 'expected_goals', 'expected_assists',
    #         'expected_goals_conceded', 'defcon', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'own_difficulty', 'transfers_in', 'transfers_out', 'local_predictions', 'local_const_predictions'] #, 'difficulty']


    check_features = element_features
    
    
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
    
    
        
    hyperparam_path = main_directory + f'\models\hyperparams{pos}.pkl'
    with open(hyperparam_path, 'rb') as f:
        old_trials = pickle.load(f)
    
    hyperparams = old_trials.best_trial['misc']['vals']
    #reformat the lists
    old_hyperparams = {}
    for field, val in hyperparams.items():
        old_hyperparams[field] = val[0]
        
    
    if old_hyperparams['eval_fraction'] < min_eval_fraction:
        old_hyperparams['eval_fraction'] = min_eval_fraction
    
    loss = objective_xgboost(old_hyperparams)
    old_loss = loss['loss']
    
    
    print('Old loss: ', old_loss)
    
    space={ 'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 4000, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.45),
            'learning_rate': hp.loguniform('learning_rate', 0, np.log(7)),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(175), 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 8200),
            'max_depth': hp.qloguniform("max_depth", 1, np.log(900), 1), 
            'max_leaves': hp.quniform('max_leaves', 0, 3000, 1),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 3000),
            'min_split_loss': hp.loguniform('min_split_loss', 0, np.log(200)), #log?
            'n_estimators': hp.quniform('n_estimators', 2, 65000, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 1500),
            'reg_lambda' : hp.uniform('reg_lambda', 0, 20000),            
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'temporal_window': hp.quniform('temporal_window', 1, 8, 1),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1  
            
        }

    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [True, False]), #111

       
    new_trials = Trials() 
            
    loss = np.inf
    tid = 0
    
    
    #temp_hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
    
    while loss > 60 and tid < 200:

        #optmimize hyperparameters. use all training data
        new_hyperparams = fmin(fn = objective_xgboost,
                        space = space,
                        algo = atpe.suggest,
                        trials = new_trials,
                        max_evals=99999999,
                        early_stop_fn=no_progress_loss(100)
                        )
        
        #pickle.dump(trials, open(temp_hyperparam_path, "wb"))
        
        loss = new_trials.best_trial['result']['loss']
        tid = len(new_trials)
        
    
    
    # with open(temp_hyperparam_path, 'rb') as f:
    #     new_trials = pickle.load(f)
        
    # hyperparams = new_trials.best_trial['misc']['vals']
    # #reformat the lists
    # new_hyperparams = {}
    for field, val in new_hyperparams.items():
        print(field, val)
        
    loss = objective_xgboost(new_hyperparams)
    new_loss = loss['loss']
    
    print('New loss: ', new_loss)
      
    
    if new_loss < old_loss:
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
        

    
    pars = {
        'max_depth': int(best_hyperparams['max_depth']),
        'min_split_loss': best_hyperparams['min_split_loss'],
        'reg_lambda': best_hyperparams['reg_lambda'],
        'reg_alpha': best_hyperparams['reg_alpha'],
        'min_child_weight': int(best_hyperparams['min_child_weight']),
        'learning_rate': best_hyperparams['learning_rate'],
        'subsample': best_hyperparams['subsample'],
        'colsample_bytree': best_hyperparams['colsample_bytree'],
        'colsample_bylevel': best_hyperparams['colsample_bylevel'],
        'colsample_bynode': best_hyperparams['colsample_bynode'],
        'max_delta_step': best_hyperparams['max_delta_step'],
        'grow_policy': grow_policy[best_hyperparams['grow_policy']],
        'max_leaves': int(best_hyperparams['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(best_hyperparams['max_bin']),
        'disable_default_eval_metric': 1
        }
    
    if best_hyperparams['eval_fraction'] < min_eval_fraction:
        best_hyperparams['eval_fraction'] = min_eval_fraction
    
    random_error = []
    local_const_error = []
    local_error = []
    element_error = []
    
    
    for X1, y1, X2, y2, mask1, mask2 in zip((cv_X, val_X), (cv_y, val_y), (val_X, cv_X), (val_y, cv_y), (cvs_mask, vals_mask), (vals_mask, cvs_mask)):

        # Get the 80% of the first matches every season...
        X1 = X1.reset_index(drop=True)
        
        #remove weaks that we don't need.
        # Define the threshold
        threshold = int(best_hyperparams['temporal_window'])
        
        # Filter the columns based on the defined function
        columns_to_keep = [col for col in X1.columns if should_keep_column(col, threshold)]
        objective_X = X1[columns_to_keep]      
          
        #remove features
        for feat in check_features:
            if feat in space.keys():
                #if remove
                if not space[feat]:     
                    columns_to_keep = []
                    for col in objective_X.columns:
                        if col == feat: # and col in do_remove_features:
                            continue
                        #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                        if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                            columns_to_keep.append(col)
                        
                    objective_X = objective_X[columns_to_keep]
                    
        # Get the 80% of the first matches every season...
        objective_copy = objective_X.copy()
        objective_copy = objective_copy.reset_index(drop=True)
        objective_copy['match_ind'] = pd.Series(match_ind[sel_name][mask1])

        
        # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
        season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                              .agg(lambda s: first_Xpct_unique(s.tolist(), 1-best_hyperparams['eval_fraction']))
                              .to_dict())
        
        # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
        # option 1: Unique across all seasons:
        fit_sample = list(set().union(*season_selection.values()))
            
        fits_mask =  pd.Series(match_ind_df[mask1]).isin(fit_sample)  # Mask for cross-validation sample
        evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
        
        fit_X = objective_X.iloc[fits_mask.values].copy()
        eval_X =  objective_X.loc[evals_mask.values].copy()
        fit_y =  y1.loc[fits_mask.values].copy()
        eval_y = y1.loc[evals_mask.values].copy()

        #make sure all categories in val_x is present in cv_x
        for column in eval_X.columns:
            if isinstance(eval_X[column].dtype, pd.CategoricalDtype):
                # Get the values in the current column of val_X
                val_values = eval_X[column]
                
                # Check which values are present in the corresponding column of cv_X
                mask = val_values.isin(fit_X[column])
                
                # Set values that are not present in cv_X[column] to NaN
                eval_X.loc[~mask, column] = np.nan
        
        dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
        deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)

        evals = [(dfit, 'train'), (deval, 'eval')]
        
        model = xgb.train(
        params=pars,
        num_boost_round=int(best_hyperparams['n_estimators']),
        early_stopping_rounds= int(best_hyperparams['early_stopping_rounds']),
        dtrain=dfit,
        evals=evals,
        custom_metric=custom_metric,
        obj=custom_objective,
        verbose_eval=False  # Set to True if you want to see detailed logging
            )
        
        # xgb.plot_importance(model, importance_type='gain',
        #                 max_num_features=20, show_values=False)
        # plt.show()

        objective_val_X = X2[columns_to_keep]
        dval_objective = xgb.DMatrix(data=objective_val_X, label=y2, enable_categorical=True)

        val_pred = model.predict(dval_objective)
        

        val_error = mean_squared_error(y2,  val_pred)
        #print('done1', val_error)
        
        mean_y = np.mean(y1)

        random_error.append(np.mean(np.abs((y2 - mean_y)**2)))
        
        
        subset_idx = train_data.index[sel_name]        # index of selected rows
        rows_to_update = subset_idx[mask2]   
        
        original_data.loc[rows_to_update, 'element_predictions'] = val_pred
        
        local_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_predictions'])**2))
        
        local_const_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_const_predictions'])**2))
        
        element_error.append(mean_squared_error(y2,  train_data.loc[rows_to_update, 'element_predictions']))

        
    print('Mean element error:', np.mean(random_error))
    print('Local constant error:', np.mean(local_const_error))
    print('Local error:', np.mean(local_error))
    print('ELement error:', np.mean(element_error))
    
    # Get the 80% of the first matches every season...
    X1 = train_copy.reset_index(drop=True)

    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(best_hyperparams['temporal_window'])

    # Filter the columns based on the defined function
    columns_to_keep = [col for col in X1.columns if should_keep_column(col, threshold)]
    objective_X = X1[columns_to_keep]      
      
    #remove features
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name])


    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-best_hyperparams['eval_fraction']))
                          .to_dict())

    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 

    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  train_y[sel_name].loc[fits_mask.values].copy()
    eval_y = train_y[sel_name].loc[evals_mask.values].copy()

    #make sure all categories in val_x is present in cv_x
    for column in eval_X.columns:
        if isinstance(eval_X[column].dtype, pd.CategoricalDtype):
            # Get the values in the current column of val_X
            val_values = eval_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(fit_X[column])
            
            # Set values that are not present in cv_X[column] to NaN
            eval_X.loc[~mask, column] = np.nan

    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]

    model = xgb.train(
    params=pars,
    num_boost_round=int(best_hyperparams['n_estimators']),
    early_stopping_rounds= int(best_hyperparams['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )

    xgb.plot_importance(model, importance_type='gain',
                    max_num_features=20, show_values=False)
    plt.show()


    summary = {'model': model, 'train_features': objective_X, 'hyperparameters': best_hyperparams}#, 'all_rows': original_df}
    
    model_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy"  + f'\element_model_{pos}.sav'
    
    pickle.dump(summary, open(model_path, 'wb'))
        
        
        
train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\model_data.pkl')  # Set index=False to not include row indices


import os
import re
import pickle
import random
import math
import time

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

import xgboost as xgb


from hyperopt import STATUS_OK, Trials, fmin, hp, atpe, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate

from difflib import SequenceMatcher

update_hyperparams = False

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"


try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


method = 'xgboost'

season_dfs = []

# Function to correct string_team based on the majority
def correct_string_team(group):
    # Count occurrences of each string_team
    counts = group['string_team'].value_counts()
    majority_team = counts.idxmax()  # Get the majority string_team
    # Replace incorrect string_team with the majority_team
    group['string_team'] = majority_team
    return group

def sequence_matcher_similarity(s1, s2):
    similarity = SequenceMatcher(None, ' '.join(sorted(s1.split())), ' '.join(sorted(s2.split()))).ratio()
    first_name_similarity = SequenceMatcher(None, s1.split()[0], s2.split()[0]).ratio()
    if len(s1.split()) > 1 and len(s2.split()) > 1:
        second_name_similarity = SequenceMatcher(None, s1.split()[1], s2.split()[1]).ratio()
    else:
        second_name_similarity = np.nan

    return similarity, first_name_similarity, second_name_similarity


def clean_string(input_string):
    # Replace underscores with spaces
    cleaned_string = input_string.replace('_', ' ')
    cleaned_string = input_string.replace("'", "")
    # Remove all numbers
    cleaned_string = re.sub(r'\d+', '', cleaned_string)
    return cleaned_string.strip()  # Optional: strip leading/trailing spaces

def custom_metric(pred_y, dtrain):

    # Targets
    y = dtrain.get_label()

    mse = mean_squared_error(y, pred_y)

    return 'MSE60', mse

def custom_objective(pred_y, dtrain):
    #https://stackoverflow.com/questions/59683944/creating-a-custom-objective-function-in-for-xgboost-xgbregressor
    
    # Targets
    y = dtrain.get_label()

    errors = pred_y - y
    #grad = 0.5 * errors
    grad = 2 * errors
    hess = np.zeros_like(pred_y) + 2
    #hess = np.ones_like(pred_y)

    return grad, hess


def quantile_objective(pred_y, dtrain):

    y = dtrain.get_label()

    q = 0.5
    
    errors = pred_y - y
    
    #multiply by two to make it comparable to the custom objective (gradients are ~half of those)
    grad = 4 * (np.where(errors > 0, (1-q) * errors, q * errors))
    hess = np.zeros_like(pred_y) + 2
    
    return grad, hess


# Define a function to check if the column name meets the criteria
def should_keep_column(column_name, threshold):
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

#optimize hyperparameters
def objective_xgboost(space):
    
    
    
    #print(space)
    
    
    space["grow_policy"] = grow_policy[space["grow_policy"]]
    
    pars = {
        'max_depth': int(space['max_depth']),
        'min_split_loss': space['min_split_loss'],
        'reg_lambda': space['reg_lambda'],
        'reg_alpha': space['reg_alpha'],
        'min_child_weight': int(space['min_child_weight']),
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'colsample_bylevel': space['colsample_bylevel'],
        'colsample_bynode': space['colsample_bynode'],
        'max_delta_step': space['max_delta_step'],
        'grow_policy': space['grow_policy'],
        'max_leaves': int(space['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(space['max_bin']),
        'disable_default_eval_metric': 1
        }
    
    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(space['temporal_window'])
    
    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    objective_X = cv_X[columns_to_keep]    
    HO_X = val_X[columns_to_keep].copy()
        
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name][cvs_mask])

    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
    
      
    #remove features that are listed in space
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                HO_X = HO_X[columns_to_keep]
                
    #remove features that are unknown
    columns_to_keep = []
    for feat in objective_X.keys():
        keep = True
        for uk in unknown_features:
            if feat == uk:
                keep = False
        
        if keep:
            columns_to_keep.append(feat)
            
    objective_X = objective_X[columns_to_keep]
    HO_X = HO_X[columns_to_keep]
                
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y.loc[fits_mask.values].copy()
    eval_y = cv_y.loc[evals_mask.values].copy()

    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)
    dval_objective = xgb.DMatrix(data=HO_X, label=val_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]
    
    model = xgb.train(
    params=pars,
    num_boost_round=int(space['n_estimators']),
    early_stopping_rounds= int(space['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )
    

    val_pred = model.predict(dval_objective)
    
    val_error = mean_squared_error(val_y,  val_pred)
    #print('done1', val_error)

    cv_loss  = objective_xgboost_custom(space, val_X, val_y, cv_X, cv_y, vals_mask)
    cv_error = cv_loss['loss']
    #print('done2', cv_error)  
    

    total_error = np.mean([val_error, cv_error])
    
    #print(cv_loss["status"])
        
    #print(total_error, type(total_error))        

    return {'loss': total_error, 'status': STATUS_OK }

def objective_xgboost_custom(space, cv_X, cv_y, val_X, val_y, cvs_mask):
    
    
    pars = {
        'max_depth': int(space['max_depth']),
        'min_split_loss': space['min_split_loss'],
        'reg_lambda': space['reg_lambda'],
        'reg_alpha': space['reg_alpha'],
        'min_child_weight': int(space['min_child_weight']),
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'colsample_bylevel': space['colsample_bylevel'],
        'colsample_bynode': space['colsample_bynode'],
        'max_delta_step': space['max_delta_step'],
        'grow_policy': space['grow_policy'],
        'max_leaves': int(space['max_leaves']),
        'tree_method': 'hist',
        'max_bin':  int(space['max_bin']),
        'disable_default_eval_metric': 1
        }
    
    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(space['temporal_window'])
    
    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    objective_X = cv_X[columns_to_keep]    
    HO_X = val_X[columns_to_keep].copy()
    
    
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name][cvs_mask])

    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
    
      
    #remove features that are listed in space
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                HO_X = HO_X[columns_to_keep]
                
    #remove features that are unknown
    columns_to_keep = []
    for feat in objective_X.keys():
        keep = True
        for uk in unknown_features:
            if feat == uk:
                keep = False
        
        if keep:
            columns_to_keep.append(feat)
            
    objective_X = objective_X[columns_to_keep]
    HO_X = HO_X[columns_to_keep]
                
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y.loc[fits_mask.values].copy()
    eval_y = cv_y.loc[evals_mask.values].copy()

    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)
    dval_objective = xgb.DMatrix(data=HO_X, label=val_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]

    model = xgb.train(
    params=pars,
    num_boost_round=int(space['n_estimators']),
    early_stopping_rounds= int(space['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )
    
    val_pred = model.predict(dval_objective)
    
    val_error = mean_squared_error(val_y,  val_pred)

    return {'loss': val_error, 'status': STATUS_OK }

               

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
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon']


#the non digit version of these features will be removed
unknown_features = ['minutes', 'ict_index', 'total_points', 'own_team_points', 'own_element_points', 'defcon']



train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)
                

# Identify categorical columns
categorical_columns = train_X.select_dtypes(['category']).columns

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()



#get one index for each match
match_ind = pd.factorize(
    train_X[['string_team', 'was_home', 'string_opp_team', 'kickoff_time']]
    .apply(lambda row: '-'.join(row.astype(str)), axis=1)
)[0]
   
#get 20% of the last matches for each season (to avoid leakage of data from points per game etc)
# Step 1: Get unique integers using a set
unique_integers = list(set(match_ind))

for ind in unique_integers:
    matches = np.where(match_ind == ind)[0]
    
    if len(matches)>0:
    
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
                
train_X = train_X.drop(['kickoff_time'], axis=1)

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()
    

# define a helper that returns the first 80% (by first-appearance order) of unique match_ind
def first_Xpct_unique(seq, X):
    # preserve order of first appearance
    seen = {}
    uniq_in_order = []
    for v in seq:
        if v not in seen:
            seen[v] = True
            uniq_in_order.append(v)
    n_keep = math.ceil(X * len(uniq_in_order))  # use ceil to keep at least one for small groups
    return set(uniq_in_order[:n_keep])

trials = Trials() 

#include all
sel_name = train_X.element_type == -1  
sel_name = ~sel_name
 
# Get the 80% of the first matches every season...
train_copy = train_X.loc[sel_name].copy()
train_copy = train_copy.reset_index(drop=True)
train_copy['match_ind'] = pd.Series(match_ind[sel_name])
match_ind_df = pd.Series(match_ind[sel_name]) 


# Step 3: Randomly select 20% of the unique integers
random.seed(42)

cvs_mask = train_copy.split == 0  # Mask for cross-validation sample
#vals_mask =train_copy.split == 1  # Mask for validation, simply the inverse of cvs_mask
vals_mask = ~cvs_mask  


cvs_match_integers = list(set(match_ind_df[cvs_mask]))
 
 
cv_X = train_copy.loc[cvs_mask.values].copy()
val_X =  train_copy.loc[vals_mask.values].copy()
cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()

#make sure all categories in val_x is present in cv_x
for column in val_X.columns:
    if isinstance(val_X[column].dtype, pd.CategoricalDtype):
        # Get the values in the current column of val_X
        val_values = val_X[column]
        
        # Check which values are present in the corresponding column of cv_X
        mask = val_values.isin(train_copy[column])
        
        # Set values that are not present in cv_X[column] to NaN
        val_X.loc[~mask, column] = np.nan
        
#make sure all categories in val_x is present in cv_x
for column in cv_X.columns:
    if isinstance(cv_X[column].dtype, pd.CategoricalDtype):
        # Get the values in the current column of val_X
        val_values = cv_X[column]
        
        # Check which values are present in the corresponding column of cv_X
        mask = val_values.isin(train_copy[column])
        
        # Set values that are not present in cv_X[column] to NaN
        cv_X.loc[~mask, column] = np.nan

cv_match_count = np.max(train_copy.loc[cvs_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
val_match_count = np.max(train_copy.loc[vals_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
min_eval_fraction = 1/np.min([cv_match_count, val_match_count])  

grow_policy = ['depthwise', 'lossguide']

#individual_features = ['season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_difficulty', 'ict_index', 'transfers_in', 'transfers_out', 'bps'] #, 'difficulty']

# element_features = ['season', 'string_team', 'points_per_played_game', 'points_per_game', 'transfers_in', 'transfers_out', 'minutes', 'was_home', 'season', 'ict_index', 'defcon', 'total_points', 'name',
 #         'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'string_opp_team', 'own_difficulty', 'opp_difficulty', 'local_predictions', 'local_const_predictions', 'influence', 'threat', 'creativity', 'bps'] #, 'difficulty']


all_features = ['element_type', 'season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'ict_index',  'influence', 'threat', 'creativity', 'bps', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'defcon', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'own_difficulty', 'transfers_in', 'transfers_out', 'local_predictions', 'local_const_predictions', 'element_predictions'] #, 'difficulty']

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
    

if old_hyperparams['eval_fraction'] < min_eval_fraction:
    old_hyperparams['eval_fraction'] = min_eval_fraction
 
loss = objective_xgboost(old_hyperparams)
old_loss = loss['loss']


print('Old loss: ', old_loss)

space={'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 3000, 1),
       'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(130), 1),
       'max_delta_step': hp.uniform('max_delta_step', 0, 2000),
       'max_depth': hp.qloguniform("max_depth", 1, np.log(225), 1),
       'min_child_weight' : hp.uniform('min_child_weight', 0, 800),
       'max_leaves': hp.quniform('max_leaves', 0, 300, 1),
        'min_split_loss': hp.loguniform('min_split_loss', 0, np.log(200)), #log?
        'n_estimators': hp.quniform('n_estimators', 2, 40000, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 1500),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 4000),
        'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.45),
        'learning_rate': hp.loguniform('learning_rate', 0, np.log(7)),
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
    space[feature] = hp.choice(feature, [True, False]), #111

   
trials = Trials() 
        
loss = np.inf
tid = 0


#temp_hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
if update_hyperparams:
    
    while loss > 60 and tid < 200:
    
        #optmimize hyperparameters. use all training data
        new_hyperparams = fmin(fn = objective_xgboost,
                        space = space,
                        algo = atpe.suggest,
                        trials = trials,
                        max_evals=99999999,
                        early_stop_fn=no_progress_loss(100)
                        )
        
        #pickle.dump(trials, open(temp_hyperparam_path, "wb"))
        
        loss = trials.best_trial['result']['loss']
        tid = len(trials)
        


     # with open(temp_hyperparam_path, 'rb') as f:
     #     new_trials = pickle.load(f)
         
     # hyperparams = new_trials.best_trial['misc']['vals']
     # #reformat the lists
     # new_hyperparams = {}
    for field, val in new_hyperparams.items():
         #new_hyperparams[field] = val[0]
         print(field, val)
         
    loss = objective_xgboost(new_hyperparams)
    new_loss = loss['loss']
     
    print('New loss: ', new_loss)

else:      
    new_loss = np.inf

if new_loss < old_loss:
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
    


pars = {
    'max_depth': int(best_hyperparams['max_depth']),
    'min_split_loss': best_hyperparams['min_split_loss'],
    'reg_lambda': best_hyperparams['reg_lambda'],
    'reg_alpha': best_hyperparams['reg_alpha'],
    'min_child_weight': int(best_hyperparams['min_child_weight']),
    'learning_rate': best_hyperparams['learning_rate'],
    'subsample': best_hyperparams['subsample'],
    'colsample_bytree': best_hyperparams['colsample_bytree'],
    'colsample_bylevel': best_hyperparams['colsample_bylevel'],
    'colsample_bynode': best_hyperparams['colsample_bynode'],
    'max_delta_step': best_hyperparams['max_delta_step'],
    'grow_policy': grow_policy[best_hyperparams['grow_policy']],
    'max_leaves': int(best_hyperparams['max_leaves']),
    'tree_method': 'hist',
    'max_bin':  int(best_hyperparams['max_bin']),
    'disable_default_eval_metric': 1
    }
 
if best_hyperparams['eval_fraction'] < min_eval_fraction:
    best_hyperparams['eval_fraction'] = min_eval_fraction

random_error = []
local_const_error = []
local_error = []
element_error = []
all_error = []

 
for X1, y1, X2, y2, mask1, mask2 in zip((cv_X, val_X), (cv_y, val_y), (val_X, cv_X), (val_y, cv_y), (cvs_mask, vals_mask), (vals_mask, cvs_mask)):

    # Get the 80% of the first matches every season...
    X1 = X1.reset_index(drop=True)
    
    #remove weaks that we don't need.
    # Define the threshold
    threshold = int(best_hyperparams['temporal_window'])
    
    # Filter the columns based on the defined function
    columns_to_keep = [col for col in X1.columns if should_keep_column(col, threshold)]
    objective_X = X1[columns_to_keep]      
      
    #remove features
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat: # and col in do_remove_features:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[sel_name][mask1])

    
    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-best_hyperparams['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
        
    fits_mask =  pd.Series(match_ind_df[mask1]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  y1.loc[fits_mask.values].copy()
    eval_y = y1.loc[evals_mask.values].copy()

    #make sure all categories in val_x is present in cv_x
    for column in eval_X.columns:
        if isinstance(eval_X[column].dtype, pd.CategoricalDtype):
            # Get the values in the current column of val_X
            val_values = eval_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(fit_X[column])
            
            # Set values that are not present in cv_X[column] to NaN
            eval_X.loc[~mask, column] = np.nan
    
    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)

    evals = [(dfit, 'train'), (deval, 'eval')]
    
    model = xgb.train(
    params=pars,
    num_boost_round=int(best_hyperparams['n_estimators']),
    early_stopping_rounds= int(best_hyperparams['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )
    
    objective_val_X = X2[columns_to_keep]
    dval_objective = xgb.DMatrix(data=objective_val_X, label=y2, enable_categorical=True)

    val_pred = model.predict(dval_objective)
    

    val_error = mean_squared_error(y2,  val_pred)
    #print('done1', val_error)
    
    mean_y = np.mean(y1)

    random_error.append(np.mean(np.abs((y2 - mean_y)**2)))
    
    
    subset_idx = train_data.index[sel_name]        # index of selected rows
    rows_to_update = subset_idx[mask2]   
    
    train_data.loc[rows_to_update, 'all_predictions'] = val_pred
    
    local_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_predictions'])**2))
    
    local_const_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_const_predictions'])**2))
    
    element_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'element_predictions'])**2))
    
    all_error.append(mean_squared_error(y2,  train_data.loc[rows_to_update, 'all_predictions']))
     
print('Mean error:', np.mean(random_error))
print('Local constant error:', np.mean(local_const_error))
print('Local error:', np.mean(local_error))
print('Element error:', np.mean(element_error))
print('All error:', np.mean(all_error))
        
        
        
#train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\all_data.pkl')  # Set index=False to not include row indices

# Get the 80% of the first matches every season...
X1 = train_copy.reset_index(drop=True)

#remove weaks that we don't need.
# Define the threshold
threshold = int(best_hyperparams['temporal_window'])

# Filter the columns based on the defined function
columns_to_keep = [col for col in X1.columns if should_keep_column(col, threshold)]
objective_X = X1[columns_to_keep]      
  
#remove features
for feat in check_features:
    if feat in space.keys():
        #if remove
        if not space[feat]:     
            columns_to_keep = []
            for col in objective_X.columns:
                if col == feat: # and col in do_remove_features:
                    continue
                #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                    columns_to_keep.append(col)
                
            objective_X = objective_X[columns_to_keep]
            
# Get the 80% of the first matches every season...
objective_copy = objective_X.copy()
objective_copy = objective_copy.reset_index(drop=True)
objective_copy['match_ind'] = pd.Series(match_ind[sel_name])


# groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                      .agg(lambda s: first_Xpct_unique(s.tolist(), 1-best_hyperparams['eval_fraction']))
                      .to_dict())

# If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
# option 1: Unique across all seasons:
fit_sample = list(set().union(*season_selection.values()))
    
fits_mask =  pd.Series(match_ind_df).isin(fit_sample)  # Mask for cross-validation sample
evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask 

fit_X = objective_X.iloc[fits_mask.values].copy()
eval_X =  objective_X.loc[evals_mask.values].copy()
fit_y =  train_y.loc[fits_mask.values].copy()
eval_y = train_y.loc[evals_mask.values].copy()

#make sure all categories in val_x is present in cv_x
for column in eval_X.columns:
    if isinstance(eval_X[column].dtype, pd.CategoricalDtype):
        # Get the values in the current column of val_X
        val_values = eval_X[column]
        
        # Check which values are present in the corresponding column of cv_X
        mask = val_values.isin(fit_X[column])
        
        # Set values that are not present in cv_X[column] to NaN
        eval_X.loc[~mask, column] = np.nan

dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)

evals = [(dfit, 'train'), (deval, 'eval')]

model = xgb.train(
params=pars,
num_boost_round=int(best_hyperparams['n_estimators']),
early_stopping_rounds= int(best_hyperparams['early_stopping_rounds']),
dtrain=dfit,
evals=evals,
custom_metric=custom_metric,
obj=custom_objective,
verbose_eval=False  # Set to True if you want to see detailed logging
    )

xgb.plot_importance(model, importance_type='gain',
                max_num_features=20, show_values=False)
plt.show()


summary = {'model': model, 'train_features': objective_X, 'hyperparameters': best_hyperparams}#, 'all_rows': original_df}

model_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\all_model.sav"

pickle.dump(summary, open(model_path, 'wb'))
    


import time

#time.sleep(60*60)


my_players = [
    {'web_name': 'Pickford', 'selling_price': 55, 'element_type': 1},
    {'web_name': 'Steele', 'selling_price': 40, 'element_type': 1},
    
    {'web_name': 'Virgil', 'selling_price': 65, 'element_type': 2},
    {'web_name': 'Guéhi', 'selling_price': 60, 'element_type': 2},
    {'web_name': 'Senesi', 'selling_price': 60, 'element_type': 2},
    {'web_name': 'Tarkowski', 'selling_price': 60, 'element_type': 2},
    {'web_name': "Matheus N.", 'selling_price': 60, 'element_type': 2},
    
    {'web_name': 'Doku', 'selling_price': 75, 'element_type': 3},
    {'web_name': 'Kroupi.Jr', 'selling_price': 75, 'element_type': 3},
    {'web_name': 'Rice', 'selling_price': 75, 'element_type': 3},
    {'web_name': 'Rogers', 'selling_price': 75, 'element_type': 3},
    {'web_name': 'Szoboszlai', 'selling_price': 70, 'element_type': 3},
    
    {'web_name': 'Thiago', 'selling_price': 80, 'element_type': 4},
    {'web_name': 'Gyökeres', 'selling_price': 75, 'element_type': 4},
    {'web_name': 'João Pedro', 'selling_price': 75, 'element_type': 4},
]






bank = 0 #in 10ths of M
free_transfers = 1
save_transfers_for_later = 5 #transfers left at end of last round (no need to put higher than 4)

forward_price_limit = -1 #in millions

minutes_thisyear_treshold = -1
form_treshold = -0.1
points_per_game_treshold = -0.1
running_minutes_threshold = -1

#'ARS', 'AVL', 'BHA', 'BOU', 'BRE', 'BUR', 'CHE', 'CRY', 'EVE',
       # 'FUL', 'LEE', 'LIV', 'MCI', 'MUN', 'NEW', 'NFO', 'SUN', 'TOT',
       # 'WHU', 'WOL'
exclude_team = []

exclude_players = []
#check james and saka and mateta
include_players = []

do_not_exclude_players = [] 


do_not_transfer_out = []
rounds_to_value = 5
#transfer to evaluate per week
trans_per_week = 3

jump_rounds = 0
#if you also want to evaluate players on the bench. in case of uncertain starters.
number_players_eval = 11

wildcard = True
benchboost = []
skip_gw = []

tripple_captain_gw = 100

iterations = 20

midfield_price_limit = -1

#assistant manager in 2024-25 season
assistant_manager_gw = 100
assistant_manager_team = 'CRY'
assistant_manager_price = 0.8 #in millions

addition_of_5_afcon_transfers = 100


force_90 = []

manual_pred = 1

#players
#
#afcon_players = ['Foster', 'Ouattara', 'Agbadou', 'M.Salah', 'Sarr', 'Doucouré', 'Ndіaye', 'Gueye', 'Iwobi', 'Bassey', 'Mbeumo', 'Mazraoui', 'Amad', 'Wissa', 'Aina', 'Boly', 'Sangaré', 'P.M.Sarr', 'Traoré', 'Diouf', 'Wan-Bissaka']
afcon_players = []
manual_blanks = {} #{29: ['Rice', 'Wilson']}

#GW               
manual_blank = {}#{34: {'BUR': ['MCI'], 'BHA': ['CHE'], 'ARS': ['NEW'], 'LIV': ['CRY']}}
#manual_double = {}
manual_double = {}#{33: {'BUR': ['MCI', 4, 2], 'BHA': ['CHE', 3, 3], 'ARS': ['NEW', 3, 5], 'LIV': ['CRY', 3, 4]}, }


season = '2026-27'
previous_season = '2025-26'

skip_free_hit_calc = False


import requests
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import random
import xgboost as xgb
import os
from pandas.api.types import CategoricalDtype
import re

num_jobs = 4


#insert string for team
#old PC

#info from vaastav
directory = os.path.join(r'C:\Users\jorgels\Github\Fantasy-Premier-League\data', season)
team_path = os.path.join(r'C:\Users\jorgels\Github\Fantasy-Premier-League\data', season, 'teams.csv')
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\all_model.sav"

try:
    df_teams = pd.read_csv(team_path)

except:
    #insert string for team
    directory = r'C:\Users\jorgels\Documents\GitHub\Fantasy-Premier-League\data' + '/' + season
    team_path = directory + "/teams.csv"
    

    df_teams = pd.read_csv(team_path)


string_names = df_teams['short_name'].values


am_num_team = np.where(string_names == assistant_manager_team)[0][0]

# #log in
# session = requests.session()
# url = 'https://users.premierleague.com/accounts/login/'
# payload = {
#  'password': 'jorgeN8#larseN(3',
#  'login': 'jorgen.sugar@gmail.com',
#  'redirect_uri': 'https://fantasy.premierleague.com',
#  'app': 'plfpl-web'
# }
# session.post(url, data=payload)

#get my team and money in the bank


#https://discord.gg/cjY37fv
def get_team():
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug  4 22:25:08 2025

    @author: jorgels
    """

    import base64
    import hashlib
    import os
    import re
    import secrets
    import uuid
    import requests

    URLS = {
        "auth": "https://account.premierleague.com/as/authorize",
        "start": "https://account.premierleague.com/davinci/policy/262ce4b01d19dd9d385d26bddb4297b6/start",
        "login": "https://account.premierleague.com/davinci/connections/0d8c928e4970386733ce110b9dda8412/capabilities/customHTMLTemplate",
        "resume": "https://account.premierleague.com/as/resume",
        "token": "https://account.premierleague.com/as/token",
        "me": "https://fantasy.premierleague.com/api/me/",
        "team": "https://fantasy.premierleague.com/api/my-team/3870053/"
    }


    def generate_code_verifier():
        return secrets.token_urlsafe(64)[:128]


    def generate_code_challenge(verifier):
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")


    code_verifier = generate_code_verifier()  # code_verifier for PKCE
    code_challenge = generate_code_challenge(code_verifier)  # code_challenge from the code_verifier
    initial_state = uuid.uuid4().hex  # random initial state for the OAuth flow

    session = requests.Session()

    # Step 1: Request authorization page
    params = {
        "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030",
        "redirect_uri": "https://fantasy.premierleague.com/",
        "response_type": "code",
        "scope": "openid profile email offline_access",
        "state": initial_state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_response = session.get(URLS["auth"], params=params)
    login_html = auth_response.text

    access_token = re.search(r'"accessToken":"([^"]+)"', login_html).group(1)
    # need to read state here for when we resume the OAuth flow later on
    new_state = re.search(r'<input[^>]+name="state"[^>]+value="([^"]+)"', login_html).group(1)


    # Step 2: Use accessToken to get interaction id and token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = session.post(URLS["start"], headers=headers).json()
    interaction_id = response["interactionId"]
    interaction_token = response["interactionToken"]


    # Step 3: log in with interaction tokens (requires 2 post requests)
    response = session.post(
        URLS["login"],
        headers={
            "interactionId": interaction_id,
            "interactionToken": interaction_token,
        },
        json={
            "id": response["id"],
            "eventName": "continue",
            "parameters": {"eventType": "polling"},
            "pollProps": {"status": "continue", "delayInMs": 10, "retriesAllowed": 1, "pollChallengeStatus": False},
        },
    )

    response = session.post(
        URLS["login"],
        headers={
            "interactionId": interaction_id,
            "interactionToken": interaction_token,
        },
        json={
            "id": response.json()["id"],
            "nextEvent": {
                "constructType": "skEvent",
                "eventName": "continue",
                "params": [],
                "eventType": "post",
                "postProcess": {},
            },
            "parameters": {
                "buttonType": "form-submit",
                "buttonValue": "SIGNON",
                "username": 'jorgen.sugar@gmail.com',
                "password": '3QdyXEGAP6t_9ad',
            },
            "eventName": "continue",
        },
    )
    dv_response = response.json()["dvResponse"]


    # Step 4: Resume the login using the dv_response and handle redirect
    response = session.post(
        URLS["resume"],
        data={
            "dvResponse": dv_response,
            "state": new_state,
        },
        allow_redirects=False,
    )

    location = response.headers["Location"]
    auth_code = re.search(r"[?&]code=([^&]+)", location).group(1)

    # Step 5: Exchange auth code for access token
    response = session.post(
        URLS["token"],
        data={
            "grant_type": "authorization_code",
            "redirect_uri": "https://fantasy.premierleague.com/",
            "code": auth_code,  # from the parsed redirect URL
            "code_verifier": code_verifier,  # the original code_verifier generated at the start
            "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030",
        },
    )

    access_token = response.json()["access_token"]
    response = session.get(URLS["team"], headers={"X-API-Authorization": f"Bearer {access_token}"})
    
    return response.json()

    
    
#my_team = get_team()

#my_players = pd.DataFrame(my_team['picks'])
#a = json.dumps(my_team)
#my_team_json = json.loads(a)
#transfers = my_team_json["transfers"]
#transfers['bank']

if wildcard: #or transfers['status'] == 'unlimited':
    free_transfers = 15
    unlimited_transfers = True
    print('Free transfers: ', 15)
else:
    unlimited_transfers = False
    #free_transfers = transfers["limit"] - transfers["made"]
    
    if free_transfers < 0:
        free_transfers = 0
        
    print('Free transfers: ', free_transfers)

#subtract 1 since we add one for each gw later
free_transfers -= 1

#get statistics of all players
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
statistics = r.json()

elements_df = pd.DataFrame(statistics['elements'])
#elements_types_df = pd.DataFrame(json['element_types'])
#teams_df = pd.DataFrame(json['teams'])

#lists all coloumns name
#elements_df.columns

#prune doen columns to thos that I need and players with 0 points
#include 79 some element treshold to remove players who are not updated in git data. remove/change later
#selected = np.logical_and(elements_df['total_points'] > 0, elements_df['minutes'] > 79)
slim_elements_df = elements_df[['transfers_in_event', 'transfers_out_event', 'yellow_cards', 'expected_goals', 'expected_goals_conceded', 'expected_assists', 'expected_goal_involvements', 'web_name', 'first_name', 'second_name', 'total_points', 'id', 'team', 'element_type', 'now_cost', 'minutes', 'points_per_game', 'chance_of_playing_next_round', 'form']].copy()
slim_elements_df["string_team"] = string_names[slim_elements_df["team"]-1]
slim_elements_df = slim_elements_df.reset_index()


#find out which gameweek
events_df = pd.DataFrame(statistics['events'])

# if not have_season_data:
#     df_gw.element = df_gw.new_year_element

i=0


while pd.to_datetime(events_df.deadline_time[i], format='%Y-%m-%dT%H:%M:%SZ') < datetime.now() - timedelta(hours=2):
    i = i + 1

current_gameweek = i + 1

print('previous:')





#get statistics for the past gameweeks
#df_past_games = pd.DataFrame(columns=['gameweek', 'team_h', 'team_a', 'difficulty_diff'])
for this_gw in range(1, current_gameweek):
    print(this_gw)
    #url = 'https://fantasy.premierleague.com/api/fixtures' + '?event=' + str(this_gw)
    #r = requests.get(url)
    #gw = r.json()

    #for game in gw:
        #add_frame = pd.DataFrame({'gameweek': this_gw, 'team_h': int(game['team_h']), 'team_a': int(game['team_a']), 'difficulty_diff': int(game['team_h_difficulty']) - int(game['team_a_difficulty'])}, index = [0])
        #df_past_games = pd.concat([df_past_games, add_frame])
        
    

print('current gameweek: ' + str(current_gameweek))

print('predicting:')
#get statistics for the x next gameweeks
df_future_games = pd.DataFrame(columns=['gameweek', 'team_h', 'team_a', 'difficulty_diff', 'kickoff_time'])

tripple_captain = []
assistant_manager = []
free_hit = []
add_afcon_transfers= []

if benchboost:
    benchboost_gws = []
else:
    benchboost_gws = [-1]
    
for i in range(jump_rounds, rounds_to_value+jump_rounds):
    this_gw = i + current_gameweek
    
    if this_gw in benchboost:
        benchboost_gws.append(this_gw)
        
    if tripple_captain_gw == this_gw:
        tripple_captain.append(True)
    else:
        tripple_captain.append(False)
        
    if assistant_manager_gw >= this_gw-2 and  assistant_manager_gw <= this_gw:
        assistant_manager.append(True)
    else:
        assistant_manager.append(False)
    
    #this will top up the transfers in this week
    if this_gw == addition_of_5_afcon_transfers:
        add_afcon_transfers.append(5)
    else:
        add_afcon_transfers.append(0)
        
        
    if this_gw in skip_gw:
        free_hit.append(True)
        skip_free_hit_calc = True
        continue
    else:
        print(this_gw)
        free_hit.append(False)
    

    url = 'https://fantasy.premierleague.com/api/fixtures' + '?event=' + str(this_gw)
    r = requests.get(url)
    gw = r.json()
    
    for game in gw:
        blank = False
        
        #check if blank
        if this_gw in manual_blank:

            for blank_game in manual_blank[this_gw]:
                
                #print(blank_game)
        
                home_team = blank_game
                away_team = manual_blank[this_gw][blank_game]
                
                home_ind = np.where(string_names == home_team)[0][0] + 1
                away_ind = np.where(string_names == away_team)[0][0] + 1
                
                if away_ind == int(game['team_a']) and home_ind == int(game['team_h']):
                    print('Manual blank:', this_gw, home_team, away_team)
                    blank = True
                    
        if not blank:                 
            timestamp = datetime.strptime(game['kickoff_time'], '%Y-%m-%dT%H:%M:%SZ')
            #positive home values == difficult games
            add_frame = pd.DataFrame({'gameweek': this_gw, 'gameweek_ind': i-jump_rounds, 'team_h': int(game['team_h']), 'team_a': int(game['team_a']), 'difficulty_diff': int(game['team_h_difficulty']) - int(game['team_a_difficulty']), 'difficulty_home': int(game['team_h_difficulty']), 'difficulty_away': int(game['team_a_difficulty']), 'kickoff_time': timestamp}, index = [0])
            if not df_future_games.empty:
                df_future_games = pd.concat([df_future_games, add_frame])
            else:
                df_future_games = add_frame              
            
    #add
    if this_gw in manual_double:
        timestamp = datetime.strptime(game['kickoff_time'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=24)            
            
        for double_game in manual_double[this_gw]:
    
            home_team = double_game
            away_team = manual_double[this_gw][double_game][0]
            home_diff = manual_double[this_gw][double_game][1]
            away_diff = manual_double[this_gw][double_game][2]
            
            home_ind = np.where(string_names == home_team)[0][0] + 1
            away_ind = np.where(string_names == away_team)[0][0] + 1
            
            add_frame = pd.DataFrame({'gameweek': this_gw, 'gameweek_ind': i-jump_rounds, 'team_h': int(home_ind), 'team_a': int(away_ind), 'difficulty_diff': int(home_diff) - int(away_diff), 'difficulty_home': home_diff, 'difficulty_away': away_diff, 'kickoff_time': timestamp}, index = [0])
            
            if not df_future_games.empty:
                df_future_games = pd.concat([df_future_games, add_frame])
            else:
                df_future_games = add_frame
            
            print('Manual double:', this_gw, home_team, away_team)
            
            
if len(benchboost_gws) == 0:
    print('No benchboost gws!')
    benchboost_gws = [-1]
        
slim_elements_df['form'] = slim_elements_df['form'].astype(float)

#exchange prices with own selling prices and calculate total cost
total_money = bank

points_per_game = slim_elements_df['points_per_game'].astype(float)
predicted_values = np.zeros((slim_elements_df.shape[0], rounds_to_value))
predicted_values_1st_gw = np.zeros((slim_elements_df.shape[0], 1))

#calculate value and add to database
total_points = slim_elements_df['total_points'].astype(float)
minutes_played = slim_elements_df['minutes'].astype(float)
form = slim_elements_df['form']

selected = points_per_game == 0
points_per_game[selected] = 0.1
games_played = np.round(total_points / points_per_game)


selected_players = (form < form_treshold) | (minutes_played < minutes_thisyear_treshold) | (points_per_game < points_per_game_treshold)

for club in exclude_team:
    ind = slim_elements_df['string_team'] == club
    selected_players.loc[ind] = True  
    if sum(ind) > 0:
        print('Exclude', slim_elements_df.loc[ind, 'string_team'].iloc[0])

for name in do_not_exclude_players:
    ind = slim_elements_df['web_name'] == name
    selected_players.loc[ind] = False

# points_per_game[selected_players] = 0

with open(model_path, 'rb') as f:
    all_model_summary = pickle.load(f)
    
predictions = []

all_model =  all_model_summary["model"]
hyperparameters =  all_model_summary["hyperparameters"]
all_temporal_window = int(hyperparameters["temporal_window"])

train_X = all_model_summary["train_features"]
#all_rows = summary["all_rows"]

with open(r'\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl', 'rb') as file:
    all_rows = pickle.load(file) 

#min_y = np.min(train_X['0total_points'])

predictions = []


# #add nan categories
# dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
#        'other_difficulty'] #'difficulty',

# int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon', 'SoT']

# float_variables = ['transfers_in', 'transfers_out', 'threat']

# #features that I don't have access to in advance.
# #opp_team_points included because it already calculate in model
# temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
#        'total_points', 'expected_goals', 'expected_assists',
#        'expected_goal_assists', 'expected_goals_conceded', 'own_team_points', 'own_element_points']
#        #'points_per_game', 'points_per_played_game']

# temporal_single_features = ['points_per_game', 'points_per_played_game']


# #total_points, minutes, kickoff time not for prediction
# fixed_features = ['element_type', 'string_team', 'season', 'name']







dynamic_features = ['string_opp_team', 'transfers_in', 'transfers_out',
        'was_home', 'own_difficulty', 'other_difficulty']#, 'difficulty']

#features that I don't have access to in advance.
#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points','defcon', 'opp_element_points', 'opp_team_points']
#included once
temporal_single_features = ['points_per_game', 'points_per_played_game']
#total_points, minutes, kickoff time not for prediction
#included once
fixed_features = ['kickoff_time', 'element_type', 'string_team', 'season', 'name']

#categories for dtype
categorical_variables = ['element_type', 'string_team', 'season', 'name']
#season_df[categorical_variables] = season_df[categorical_variables].astype('category')
#add nan categories
dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
        'other_difficulty'] #'difficulty',

int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon', 'opp_team_points']
#season_df[int_variables] = season_df[int_variables].astype('Int64')

float_variables = ['transfers_in', 'transfers_out', 'threat', 'own_element_points',  'expected_goals', 'expected_assists',
'expected_goals_conceded', 'creativity', 'ict_index', 'influence', 'opp_element_points']
#season_df[float_variables] = season_df[float_variables].astype('float')




keep_ind = []

element_models = []



#if rounds_to_value == 1 and wildcard:
for el in [1, 2, 3, 4]:
    
    #load element model
    
    element_model_path = f'\\\\platon.uio.no\\med-imb-u1\\jorgels\\fantasy\\element_model_{el}.sav'
    
    with open(element_model_path, 'rb') as f:
        element_models.append(pickle.load(f))
    
    
    selected = slim_elements_df.element_type == el
    min_keeper_price = np.min(slim_elements_df.loc[selected, 'now_cost'])
    
    selected_low_price = np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))
    
    for k in selected_low_price[0]:          
        keep_ind.append(k)
        selected_players.iloc[keep_ind[-1]] = False
    
    # if len(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0]) > 1 and el > 1:  
    #     keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0][1])
    
# else:
#     selected = slim_elements_df.element_type == 1
#     min_keeper_price = np.min(slim_elements_df.loc[selected, 'now_cost'])
    
#     keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == 1))[0][0])
              
    
# keep_ind = []
# sort_ind = np.argsort(slim_elements_df.now_cost)
# if rounds_to_value == 1 and wildcard:
#     for el in [1, 2, 3, 4]:
#         selected = slim_elements_df.iloc[sort_ind, slim_elements_df.keys() == 'element_type'] == el
        
#         keep_ind.append(sort_ind[selected['element_type']].iloc[0])
        
#         if el > 1:  
#             keep_ind.append(sort_ind[selected].iloc[1])
    
    
#predict future points
for df_name in slim_elements_df.iterrows():

    # if df_name[1].second_name == 'dos Santos Magalhães':
    #     print(df_name)
    # ind=4
    # df_name = (ind, slim_elements_df.iloc[ind])
    
    element_type = df_name[1].element_type 
    
    #with loose criteria all pass.
    if (not selected_players[df_name[0]]) and (element_type < 5):
        team = int(df_name[1].team)
        position = int(df_name[1].element_type)
        first_name = df_name[1].first_name
        second_name = df_name[1].second_name
        name = first_name + ' ' + second_name
        player_id =  df_name[1].id
        
        
        element_model = element_models[position-1]
        
        if all_temporal_window < element_model['hyperparameters']['temporal_window']:
            temporal_window = element_model['hyperparameters']['temporal_window']
        else:
            temporal_window = all_temporal_window

        # url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id)
        # downloaded = False
        # while not downloaded:
        #     try:
        #         r = requests.get(url)
        #         player = r.json()
        #         downloaded = True
        #     except:
        #         time.sleep(30)

        # player_games = pd.DataFrame(player['history'])

        # player_games['kickoff_time'] =  pd.to_datetime(player_games['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        # player_games = player_games.sort_values(by='kickoff_time')
        # player_games.set_index('kickoff_time', inplace=True)

        # fixtures = pd.DataFrame(player['fixtures'])
        # fixtures['kickoff_time'] =  pd.to_datetime(fixtures['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        # fixtures = fixtures.sort_values(by='kickoff_time')

        # should_have_trainingdata = True
        # should_have_database = False
        # past_history  = player["history_past"]
        # if past_history == []:
        #     should_have_trainingdata = False

        # else:
            # last_history = past_history[-1]['season_name']

            # if last_history[:4] == previous_season[:4]:
            #     should_have_database = True
                
        #build prediction_matrix
        #matches with team
        selected_matches = np.logical_or(df_future_games.team_h == team, df_future_games.team_a == team)
        gws = df_future_games[selected_matches]
                
        #cif there is no historical data for player. use data from slim
        #or if there are no matches
        if sum(all_rows.name == name) == 0 or len(gws)==0:

            selected_ind = np.where(elements_df.id == player_id)[0][-1]

            #at beginnig of season data contains season sums
            if sum(all_rows.name == name) == 0:
                print(name, ': seto to zero. Does not exist in game database. Have no historical data')
                
            is_estimated = True
            #just take some random data to make the script work
            predicting_df = all_rows.iloc[-(temporal_window+1+rounds_to_value):]
            
            game_idx = len(predicting_df)

        else:
            
            #load_player model
            player_model_path = rf"\\platon.uio.no\med-imb-u1\jorgels\fantasy\local_models\{name}.sav"
            
            with open(player_model_path, 'rb') as f:
                player_model = pickle.load(f)
                
            if temporal_window < player_model['hyperparams']['temporal_window']:
                temporal_window = player_model['hyperparams']['temporal_window']
        
            is_estimated = False
            selected = all_rows.name == name
            predicting_df = all_rows.loc[selected]
            predicting_df = predicting_df.iloc[-(temporal_window+1+rounds_to_value):]



        #low diff_difficulty = difficult games
        diff_difficulty = np.array(df_future_games.difficulty_diff[selected_matches])
        home_team = np.array(df_future_games.team_h.loc[selected_matches])
        away_team = np.array(df_future_games.team_a.loc[selected_matches])
        home_matches = home_team == team
        diff_difficulty[home_matches] = -diff_difficulty[home_matches]

        home_difficulty = np.array(df_future_games.difficulty_home[selected_matches])
        away_difficulty = np.array(df_future_games.difficulty_away[selected_matches])

        #correct for fixtures
        pred_score = np.zeros(rounds_to_value)
        total_matches = 0

        gws = gws.reset_index()

        minutes = np.nanmean(predicting_df.iloc[-2:]['minutes'])
        
        last_known_row = predicting_df.iloc[-1].copy()

        for game in gws.iterrows():

            #add empty row
            new_row = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in predicting_df.dtypes.items()})

            #add fixed
            #only if we want to rotate the features
            new_row.loc[0, fixed_features] = predicting_df[fixed_features].iloc[-1]
            new_row.loc[0, 'points_per_game'] = predicting_df['points_per_game'].iloc[-1]
            new_row.loc[0, 'points_per_played_game'] = predicting_df['points_per_played_game'].iloc[-1]
            #new_row.loc[0] = predicting_df.iloc[-1]

            #add dynamic

            game_idx = game[0]
            gw_idx = int(game[1].gameweek_ind)
            gw = game[1].gameweek

            if home_matches[game_idx]:

                new_row['own_difficulty'] = home_difficulty[game_idx]
                new_row['other_difficulty'] = away_difficulty[game_idx]
                new_row['string_opp_team'] = string_names[game[1].team_a-1]
                new_row['was_home'] = 1
                #necessary because of unavailable players
                new_row['string_team'] = string_names[game[1].team_h-1]

            else:

                new_row['own_difficulty'] = away_difficulty[game_idx]
                new_row['other_difficulty'] = home_difficulty[game_idx]
                new_row['string_opp_team'] = string_names[game[1].team_h-1]
                new_row['was_home'] = 0

                new_row['string_team'] = string_names[game[1].team_a-1]
                    

            #new_row['difficulty'] = diff_difficulty[game_idx]
            new_row['kickoff_time'] = game[1]['kickoff_time']

            sum_transfers = sum(elements_df.transfers_in_event)
            if gw_idx == 0 and sum_transfers > 0:

                new_row['transfers_in'] = elements_df.iloc[df_name[0]].transfers_in_event/sum_transfers
                new_row['transfers_out'] = elements_df.iloc[df_name[0]].transfers_out_event/sum_transfers
            else:
                new_row['transfers_in'] = np.nan
                new_row['transfers_out'] = np.nan


            predicting_df = pd.concat([predicting_df, new_row], ignore_index = True, axis=0)

        #add temporal features
        #for each week iteration
        category_names  = [fixed_features]
        
        #since we will not add more data we can loop the columns!
        for match_row in predicting_df.iterrows():
            for feat in predicting_df.keys():
                
                if match_row[0] > 0:
                
                    #split the feat in integer and string
                    m = re.match(r"^(\d+)(.*)$", feat)
                    if m:
                        num = int(m.group(1))   # 20
                        rest = m.group(2)
                        
                        if rest == 'opp_team_points' or rest == 'opp_element_points' or num >= temporal_window:
                            continue
                        
                        #all temporal features is the last one + 1
                        if num == 0:
                            last_match_feat  = rest     
                        else:
                            last_match_feat  = str(num-1) + rest
                        
                        #print(feat, 'Change', predicting_df.loc[match_row[0], feat], 'with', predicting_df.loc[match_row[0]-1, last_match_feat])
                        
                        #the match we look at is the previous match and a column with one less digit as name.
                        #gives nan for matches that are not yet played
                        #predicting_df.loc[match_row[0], feat] = predicting_df.loc[match_row[0]-1, last_match_feat]
                        
                        #if we want the last played match to be used as the starting point
                        predicting_df.loc[match_row[0], feat] = last_known_row[last_match_feat]
                        
                             
                
            
            
            
            
    
        
        # if temporal_window > 0:
        #     for k in range(int(temporal_window)):
    
    
        #         temporal_names = [str(k) + s for s in temporal_features]
    
        #         if k==0:
                    
        #             dynamic_names = [s for s in dynamic_features]        
        #             temporal_single_names = [s for s in temporal_single_features]
                    
        #             col_names = temporal_single_names + dynamic_names + temporal_names 

        #         else:
        #             dynamic_names = [str(k-1) + s for s in dynamic_features] 
                    
        #             col_names = dynamic_names + temporal_names
                    
        #         #add in empty data
        #         temp_train = pd.DataFrame(index=predicting_df.index, columns=col_names)
                
        #         #loop all data
                
        #         if k==0:
        #             temporal_single_data = predicting_df[temporal_single_features].shift(k+1)
        #             temp_train[temporal_single_names] = temporal_single_data.values
                   
                
    
        #         temporal_data = predicting_df[temporal_features].shift(k+1)
        #         dynamic_data = predicting_df[dynamic_features].shift(k)
    
        #         temp_train.loc[temporal_names] = temporal_data.values
        #         temp_train.loc[dynamic_names] = dynamic_data.values
    
        #         #set dtype
        #         for col in temp_train.columns:
    
        #             col_stem = ''.join([char for char in col if not char.isdigit()])
    
        #             if col_stem in dynamic_categorical_variables:
        #                 temp_train[col] = temp_train[col].astype('category')
        #             elif col_stem in int_variables:
        #                 temp_train[col] = temp_train[col].astype('Int64')
        #             elif col_stem in temporal_features or col_stem in float_variables or col_stem in temporal_single_features:
        #                 temp_train[col] = temp_train[col].astype('float')
        #             else:
        #                 print('CHECK', col)
    
        #         predicting_df = pd.concat([predicting_df, temp_train], axis=1)
        
        if temporal_window > 0:        
            #add in data about the opponent   
            opponent_point_names = [str(k) + 'opp_team_points' for k in range(temporal_window)]  
            opponent_element_names = [str(k) + 'opp_element_points' for k in range(temporal_window)]  
            
            for ind, game in enumerate(predicting_df.iloc[-(game_idx+1):].iterrows()):
            
                index = [game[0]]
        
                temp_train = pd.DataFrame(index=index, columns=opponent_point_names + opponent_element_names)
                     
                opponent_club =  game[1]['string_opp_team']
                    
                opp_selected = all_rows.string_opp_team == opponent_club
                
                kick_off = game[1]['kickoff_time']
                    
                #find all matches of the opponent before the current match
                opp_match_selected =  opp_selected #& (all_rows['kickoff_time'] < kickoff)
                 
                #find the unique kickoff times
                first_indices = all_rows.loc[opp_match_selected].drop_duplicates(subset='kickoff_time', keep='first').index
                
                full_ooop = pd.Series([pd.NA] * len(opponent_point_names), dtype="Int64")
                
                opponents_of_opponents_points = all_rows.loc[first_indices[-temporal_window:], "own_team_points"]
                opponents_of_opponents_points = opponents_of_opponents_points.shift(-ind)
                if len(opponents_of_opponents_points):
                    full_ooop[-len(opponents_of_opponents_points):] = opponents_of_opponents_points.values
    
                temp_train.loc[index, opponent_point_names] = full_ooop[::-1].to_list()
                
                opp_elem_selected =  opp_selected & (all_rows['element_type'] == position)
                      
                first_indices = all_rows.loc[opp_elem_selected].drop_duplicates(subset='kickoff_time', keep='first').index
                
                full_oooep = [np.nan] * len(opponent_element_names)
                
                opponents_of_opponents_elements = all_rows.loc[first_indices[-temporal_window:], "own_element_points"]
                opponents_of_opponents_elements = opponents_of_opponents_elements.shift(-ind)
                if len(opponents_of_opponents_points):
                    full_oooep[-len(opponents_of_opponents_elements):] = opponents_of_opponents_elements
                
                temp_train.loc[index, opponent_element_names] = full_oooep[::-1]
        
        
                #set dtype
                for col in opponent_point_names:
                    temp_train[col] = temp_train[col].astype('Int64')
                    
                for col in opponent_element_names:
                    temp_train[col] = temp_train[col].astype('float')
                    
                predicting_df.loc[game[0], temp_train.keys()] = temp_train.values[0].copy()
                    
                
            #set dtype
            for col in predicting_df.columns:
    
                col_stem = ''.join([char for char in col if not char.isdigit()])
    
                if col_stem in dynamic_categorical_variables:
                    cat_series = predicting_df[col].astype('category')
                    predicting_df = predicting_df.drop(columns=[col])
                    predicting_df[col] = cat_series
                    #predicting_df.loc[:, col] = predicting_df[col].astype('category')
                elif col_stem in int_variables:
                    predicting_df.loc[:, col] = predicting_df[col].astype('Int64')
                elif col_stem in temporal_features or col_stem in float_variables or col_stem in temporal_single_features:
                    predicting_df.loc[:, col] = predicting_df[col].astype('float')
                # else:
                #     print('CHECK', col)
                
                
        
        #include also train_X to maintain categories. use inner to not get too many columns
        #predicting_df = pd.concat([train_X, predicting_df], ignore_index = True, join='inner')
        common_columns = train_X.columns.intersection(predicting_df.columns)
        predicting_df = predicting_df[common_columns]


        #total_points, minutes, kickoff time not for prediction
        #pick the last rows
        predicting_df = predicting_df.iloc[-(game_idx+1):]

        #keep_rows = predicting_df.shape[0]



        #predicting_df[fixed_features] = predicting_df[fixed_features].astype('category')

        for cat in predicting_df.keys():
            if isinstance(train_X[cat].dtype, pd.CategoricalDtype):
                #get_categories
                train_cats = train_X[cat].cat.categories
                cats = CategoricalDtype(categories=train_cats, ordered=False)
                predicting_df[cat] = predicting_df[cat].astype(cats)

        #remove train_X
        #predicting_df = predicting_df.iloc[-keep_rows:]

        predicting_df = predicting_df.reset_index(drop=True)
        
        
        #make sure all categories in pred is present in train. to avoid predictions outside of feature space
        for column in predicting_df.columns:
            if isinstance(predicting_df[column].dtype, pd.CategoricalDtype):
                # Get the values in the current column of val_X
                val_values = predicting_df[column]
                
                # Check which values are present in the corresponding column of cv_X
                mask = val_values.isin(train_X[column])
                
                if sum(~mask) > 0:                
                    # Set values that are not present in cv_X[column] to NaN
                    predicting_df.loc[~mask, column] = np.nan
                    
                    if not column == 'string_opp_team':
                        print(str(val_values[~mask]) + ': does not exist in training data. Set to nan')




        #prediciting one by one:
        for game in gws.iterrows():

            game_idx = game[0]
            gw_idx = int(game[1].gameweek_ind)
            
            gw = game[1].gameweek

            Dgame = xgb.DMatrix(data=predicting_df.iloc[[game_idx]], enable_categorical=True)
            
            local_prediction = player_model['model'].predict(Dgame)[0]
            

            estimated = all_model.predict(Dgame)[0]
            
            #estimated = (10**estimated) - 1 + min_y

            # #insert value intor future matches
            # s=0
            # for future_game in range((game_idx+1), (gws.shape[0])):
            #      string_name = str(s)+'total_points'
            #      predicting_df.loc[future_game, string_name] = estimated.copy()
            #      s += 1

            for name_inc in include_players:
                if df_name[1]['first_name'] in name_inc and df_name[1]['second_name'] in name_inc:
                    estimated = 100*random.random()
                    
            if df_name[1]['web_name'] in include_players:
                estimated = 100*random.random()
                    
            #exclude the cheap forwards
            if df_name[1]['element_type'] == 4 and df_name[1]['now_cost'] <= forward_price_limit*10:
                estimated = 0
            elif  df_name[1]['element_type'] == 3 and df_name[1]['now_cost'] <= midfield_price_limit*10:
                estimated = 0
            
            elif (gw in manual_blanks.keys()) and (df_name[1]['web_name'] in manual_blanks[gw]):
                estimated = 0
            
            
            
            elif df_name[1]['web_name'] not in do_not_exclude_players:       
                
                if minutes < running_minutes_threshold  or np.isnan(minutes):
                    estimated = 0
    
                #remove if unlikely to play: game_idx for game. gw_idx for gw
                if gw_idx==0 and gw_idx+jump_rounds == 0 and df_name[1]['chance_of_playing_next_round'] < 75:
                    estimated = 0
    
                # if sum(all_rows.name == name) == 0 and (game_idx == 0):
                #     if should_have_trainingdata:
                #         print(name + ': does not exist in training data. Shoul dbe predicted without name')
                #     #estimated = 0
                if game_idx == 0:
                    #check that categorical is the same!
                    # Identify categorical columns
                    categorical_columns = predicting_df.select_dtypes(['category']).columns
    
                    # Reset categories for each categorical column
                    for column in categorical_columns:
    
                        are_identical = set(train_X[column].cat.categories) == set(predicting_df[column].cat.categories)
                        if not are_identical:
                            print("ERROR CATEGORIES", df_name[0], column)

                if df_name[1]['web_name'] in exclude_players:
                    estimated = 0
    
                for exclude_name in exclude_players:
                    if df_name[1]['first_name'] in exclude_name and df_name[1]['second_name'] in exclude_name:
                        estimated = 0
                        
                if is_estimated and df_name[1]['web_name'] not in do_not_exclude_players:
                    estimated = 0
                    

            pred_score[gw_idx] = pred_score[gw_idx] + estimated
            total_matches = total_matches + 1
          
        #keep a budget player
        if df_name[0] in keep_ind:
            for p_ind, p in enumerate(pred_score):
                if p < 0.1:
                    pred_score[p_ind] = 0.1
                
            print('Including because of low price:', df_name[1].web_name)


        first_gw = pred_score[0]

        #predicted_points = pred_score/total_matches - (4 / rounds_to_reset)
        predicted_points = pred_score  #- (4 / rounds_to_reset)
        predicted_values[df_name[0]] = predicted_points
        predicted_values_1st_gw[df_name[0]] = first_gw
        predictions.append(pred_score)

    else:
        predictions.append(np.zeros(rounds_to_value).astype(float))



del all_rows



slim_elements_df['points_1st_gw'] = predicted_values_1st_gw

# #set what to use for evaluation. can be points_per_game
# prediction = np.copy(predicted_values)




slim_elements_df['prediction'] = predictions

#turn to numpy array
all_gws_predictions = np.array(predictions)

#start out with blank team (none are picked)
slim_elements_df['picked'] = False
slim_elements_df['original_player'] = False

#initiate variables counting number of players in each position/team
num_position = np.zeros([4, 1])
num_team = np.zeros([20, 1])

#insert element into my_players
my_players = pd.DataFrame(my_players)
my_element = []
for k in my_players.iterrows():
    s = (slim_elements_df.web_name == k[1].web_name) & (slim_elements_df.element_type == k[1].element_type)
    if not sum(s) == 1:
        print('More than one possible my player')
        
    my_element.append(slim_elements_df.loc[s, 'id'].values[0])
    
my_players['element'] = my_element
      

# decorrect own players
my_players_df = pd.DataFrame()
for i in range(15):
    id = my_players.iloc[i]['element']
    selling_price = my_players.iloc[i]['selling_price']

    selected = slim_elements_df['id'] == id

    slim_elements_df.loc[selected, 'now_cost'] = selling_price

    slim_elements_df.loc[selected, 'original_player'] = True
    slim_elements_df.loc[selected, "picked"] = True

    my_players_df = pd.concat([my_players_df, slim_elements_df.loc[selected]], ignore_index=True)

    total_money = total_money + selling_price

    print(list(slim_elements_df.web_name[selected])[0] + ' ' + str(np.round(sum((predicted_values[selected])), decimals=1)))


#total_money = 1003

original_players = my_players_df

now_cost = slim_elements_df['now_cost'].astype(float)
value = slim_elements_df['prediction'].apply(sum) / now_cost
slim_elements_df['value'] = value

        
        

#find points for each match or a series of matches (depends on len of prediction)
def find_team_points(team_positions, gw_prediction, benchboost, tc):
    
    if tc:
        captain_return = 3
    else:
        captain_return = 2
    
    if benchboost:
        captain_ind = np.argmax(gw_prediction)
        
        #cannot play benchboost and tripple catpain in same round
        gw_prediction[captain_ind] = gw_prediction[captain_ind]*2

        return sum(gw_prediction)

    else:

        pred_points = []

        order = np.argsort(gw_prediction)
        ordered_points = np.sort(gw_prediction)
        ordered_positions = team_positions[order]

        #pick the 11 best players of the team
        for i in range(number_players_eval):

            #force pick from some positions
            if i == 0:
                selected = ordered_positions == 1

            elif i == 1 or i == 2 or i == 3:
                selected = ordered_positions == 2

            elif i == 4 or i == 5:
                selected = ordered_positions == 3

            elif i == 6:
                selected = ordered_positions == 4
            #do not repick a keeper
            else:
                selected = ordered_positions > 1

            selected_index = np.where(selected)[0][-1]

            pred_points.append(ordered_points[selected_index])

            ordered_points = np.delete(ordered_points, selected_index)
            ordered_positions =  np.delete(ordered_positions, selected_index)

        captain_ind = np.argmax(pred_points)

        pred_points[captain_ind] = pred_points[captain_ind]*captain_return

        return sum(pred_points)




#calculate points for a given set of transfers
def objective(check_transfers, unlimited_transfers, free_transfers):

    #print(check_transfers)

    team = slim_elements_df['picked'].values.copy()

    # print(params)

    if unlimited_transfers:
        gw_iteration = 1
        
        #force first to be true if wildcard
        if any(assistant_manager):
            assistant_manager[0] = True
        
    else:
        gw_iteration = rounds_to_value

    max_price = 0
    deduct_points = 0

    #loop through the transfers and check if they are possible
    for gw in range(gw_iteration):
        
        num_team = np.zeros((20))
        if assistant_manager[gw]:
            #convert from M to 100k
            am_price = assistant_manager_price*10
            #do not add 1 since it is indexed earlier.
            num_team[am_num_team] = 1
            
        else:
            am_price = 0           
        
        if not unlimited_transfers:

            k=0

            for gw_trans in range(trans_per_week):
                transfer = check_transfers[gw*trans_per_week + gw_trans]
                k += 1

                if not np.isnan(transfer[0]):
                    #check if players are already transfered
                    if team[transfer[0]] == False or team[transfer[1]] == True:
                        print('I think this never happens 1', check_transfers, gw*trans_per_week + gw_trans, transfer)
                        #return np.nan, np.nan, np.nan

                    team[transfer[0]] = False
                    team[transfer[1]] = True

        else:
            #check all transfers before moving on
            for transfer in check_transfers:
                if not np.isnan(transfer[0]):
                    #print('I think this never happens 2')
                    team[transfer[0]] = False
                    team[transfer[1]] = True

        #if too expensive or too many players from club
        total_price =  sum(slim_elements_df.loc[team, 'now_cost'])
        
        #get the max price for each of the gws
        if max_price < total_price:
            max_price = total_price

        #count_clubs
        for team_ind in slim_elements_df.loc[team, 'team']:
            num_team[team_ind-1] += 1

        if (total_money-am_price) < total_price or np.max(num_team) > 3 or sum(team) != 15:
            # if total_money < total_price:
            #     print('money')
            # if np.max(num_team) > 3:
            #     print('team')
            # if sum(team) != 15:
            #     print('overlap')
            #print('a')
            return [np.nan], [np.nan], [np.nan]

    team = slim_elements_df['picked'].values.copy()

    team_points = []

    all_points = []

    #loop through the transfers and count points
    for gw in range(gw_iteration):
        
        if free_hit[gw]:
            team_points.append(0)
            all_points.append(0)
            continue
        
        #add transfers
        if add_afcon_transfers[gw] > 0:
            #subtract one since 1 will be added soon
            free_transfers = add_afcon_transfers[gw] - 1 

        if not unlimited_transfers:

            #if all pred is zero skip week (=free hit)
            if sum(predictions[:, gw]) == 0:
                estimated_points = 0

                all_points.append(0)
            else:
                #add one for the gw
                free_transfers +=1
                
                #print('GW:', gw, free_transfers + 5)

                k=0
                for gw_trans in range(trans_per_week):
                    transfer = check_transfers[gw*trans_per_week + gw_trans]
                    k += 1
                    
                    #subtract a free transfer if there is a transfer
                    if not np.isnan(transfer[0]):
                        team[transfer[0]] = False
                        team[transfer[1]] = True
                        #subtract one for the transfer
                        free_transfers -=1
                    
                        #pay if negative
                        if free_transfers < 0:
                            deduct_points += -transfer_cost
                            free_transfers += 1
                    
                    #ceil the possible number of transfers. 4 since we add one before next round
                    if free_transfers > 4:
                        free_transfers = 4

                gw_prediction = predictions[team, gw]
                team_positions = slim_elements_df.loc[team, 'element_type'].values

                estimated_points = find_team_points(team_positions, gw_prediction, benchboost[gw], tripple_captain[gw])
                
                captain_bonus = np.max(predictions[team, gw])
                
                all_points.append(np.sum(predictions[team, gw])+captain_bonus)

            team_points.append(estimated_points)

        else:
            #loop all transfers before calculating the points.
            for transfer in check_transfers:
                if not np.isnan(transfer[0]):
                    team[transfer[0]] = False
                    team[transfer[1]] = True

            #for gws in range(rounds_to_value):
            predictions_shape = predictions.shape
            if len(predictions_shape) == 2:
                iterate = predictions_shape[1]
            else:
                iterate = 1
            for gws in range(iterate):
                
                if len(predictions_shape) == 2:
                    gw_prediction = predictions[team, gws]
                else:
                    gw_prediction = predictions[team]
                team_positions = slim_elements_df.loc[team, 'element_type'].values

                estimated_points = find_team_points(team_positions, gw_prediction, benchboost[gws], tripple_captain[gws])

                team_points.append(estimated_points)

                captain_bonus = np.max(gw_prediction)
                
                all_points.append(np.sum(gw_prediction)+captain_bonus)


        #print(sum(team_points))
    
    #subtract points if we haven't saved transfers
    if free_transfers < save_transfers_for_later:
        deduct_transfers = save_transfers_for_later - free_transfers
        deduct_points += deduct_transfers*-transfer_cost
        
    team_points.append(deduct_points)
    all_points.append(deduct_points)
        
    return team_points, max_price, all_points



def check_random_transfers(i, unlimited_transfers, free_transfers):
    
    rng = np.random.default_rng(seed=i)

    random_evaluated_transfers = []
    random_points = []
    random_prices = []

    random_all_points = []

    random_counts = np.zeros((len(point_diff), len(probabilities[0])), dtype='uint32')
    random_sum_points = np.zeros((len(point_diff), len(probabilities[0])))

    for j in range(batch_size):
        
        #print(j)

        #loop to get a transfer combination
        random_transfer_ind = []
        random_putative_transfers = []
        #add one and one transfer
        for i in range(len(point_diff)):
            random_trans_ind = rng.choice(np.arange(prob.shape[0]), 1, p=prob[:, i])[0]
            random_trans = transfers[random_trans_ind]

            #redo to nan if player is allready transfered in/out
            if (not random_trans[0] == np.nan) and (i > 0):
                #loop thropugh the already recorded transfers
                for t in random_putative_transfers:
                    if t[0] == random_trans[0] or t[1] == random_trans[1]:
                        #skip every third transfer
                        random_trans_ind = prob.shape[0]-1
                        break

            random_transfer_ind.append(random_trans_ind)
            random_trans = transfers[random_trans_ind]
            random_putative_transfers.append(random_trans)

        # random_transfer_ind = []
        # random_putative_transfers = []
        # for i in best_transfer:
        #     trans = transfers[i]
        #     random_putative_transfers.append(trans)
        #     random_transfer_ind.append(i)


        random_point, random_price, random_all_point = objective(random_putative_transfers, unlimited_transfers, free_transfers)
            
            
        random_points.append(random_point)
        random_prices.append(random_price)
        random_all_points.append(random_all_point)
        random_evaluated_transfers.append(random_transfer_ind)

        for week, transfer in enumerate(random_transfer_ind):
            if not any([np.isnan(p) for p in random_point]):
                random_sum_points[week, transfer] = random_sum_points[week, transfer] + (sum(random_point)-np.sum(baseline_point))
                random_counts[week, transfer] += 1
            #punish also nan teams
            else:
                random_counts[week, transfer] += 1
                
                
    if not all(np.isnan([np.sum(inner_list) for inner_list in random_points])):
        random_max_value = np.nanmax([np.sum(inner_list) for inner_list in random_points])

        random_indices_with_max_value = [i for i, value in enumerate(random_points) if np.sum(value) == random_max_value]
        random_min_value_other_list = min(random_prices[i] for i in random_indices_with_max_value)
        random_best_ind = next(i for i in random_indices_with_max_value if random_prices[i] == random_min_value_other_list)

        random_best_point = np.sum(random_points[random_best_ind])
        random_best_price = random_prices[random_best_ind]
        random_best_all_point = np.sum(random_all_points[random_best_ind])
        random_best_transfer = random_evaluated_transfers[random_best_ind]


                                    
        #print(best_point, best_price)


        check_guided = True
        while check_guided:
            check_guided = False

            random_order = list(range(prob.shape[1]))
            random.shuffle(random_order)

           #print('New')
            #guided part. exhange one transfer
            for k in random_order:
                
                #if there are more than one transfer to choose from
                if sum(prob[:, k] > 0) < 2:
                    continue
                
                guided_points, guided_prices, guided_all_points, guided_evaluated_transfers, guided_sum_points, guided_counts = check_guided_transfers(k, random_best_transfer, random_best_point, unlimited_transfers, free_transfers)

                random_points = random_points +  guided_points
                random_prices = random_prices + guided_prices
                random_all_points = random_all_points + guided_all_points
                random_evaluated_transfers = random_evaluated_transfers + guided_evaluated_transfers
                random_sum_points += guided_sum_points
                random_counts += guided_counts

                #max points
                #random variables now includes both
                guided_max_value = np.nanmax([np.sum(inner_list) for inner_list in random_points])
                #lowest price
                guided_indices_with_max_value = [i for i, value in enumerate(random_points) if np.sum(value) == guided_max_value]
                guided_min_value_other_list = min(random_prices[i] for i in guided_indices_with_max_value)
                guided_best_ind = next(i for i in guided_indices_with_max_value if random_prices[i] == guided_min_value_other_list)

                guided_best_price = random_prices[guided_best_ind]

                #highest total points
                guided_best_point = 0
                for i in range(len(random_all_points)):
                    if np.sum(random_points[i]) == guided_max_value and random_prices[i] == guided_best_price and np.sum(random_all_points[i]) > guided_best_point:
                        guided_best_point = np.sum(random_all_points[i])
                        guided_best_ind = i

                #print(k)
                if guided_max_value > random_best_point or (guided_max_value == random_best_point and guided_best_price < random_best_price) or (guided_max_value == random_best_point and guided_best_price == random_best_price and  guided_best_point > random_best_all_point):
                            
                    check_guided = True
                    random_best_point = sum(random_points[guided_best_ind])
                    random_best_price = guided_best_price
                    random_best_all_point = guided_best_point.copy()
                    random_best_transfer = random_evaluated_transfers[guided_best_ind].copy()
                    


                    #print(random_best_point, random_best_price, random_best_all_point)
                  
            
            
            
        #DELAY transfers as much as possible        
        delayed_trans_ind = random_best_transfer.copy()
        
        if not unlimited_transfers:

            for t in range(len(random_best_transfer)-1):
                
                #print(t)
                
                #do not check if nan:
                if delayed_trans_ind[t] == len(transfers)-1:
                    continue
                
                #print('Move from', t)
                
                #loop tp the next transfers
                for potential_move_to in range(t+1, len(delayed_trans_ind)):
                    
                    #print(t, potential_move_to)
                    
                    delayed_trans_ind = random_best_transfer.copy()
                    
                    if delayed_trans_ind[potential_move_to] == len(transfers)-1:
                        
                        #print('Check move to', potential_move_to)

                        #switch transfer
                        delayed_trans_ind[potential_move_to] = delayed_trans_ind[t]
                        delayed_trans_ind[t] = len(transfers)-1
                        
                        delayed_transfers = []
                        for k in range(len(delayed_trans_ind)):
                            delayed_transfers.append(transfers[delayed_trans_ind[k]])
                        
                        delayed_point, delayed_price, delayed_all_point = objective(delayed_transfers, unlimited_transfers, free_transfers)
                        
                        random_points = random_points + [delayed_point]
                        random_prices = random_prices + [delayed_price]
                        random_all_points = random_all_points + [delayed_all_point]
                        random_evaluated_transfers = random_evaluated_transfers + [delayed_trans_ind]
                        
                        if np.sum(delayed_point) >= random_best_point:
                            #print('Move', t, 'to', potential_move_to)
                            
                            random_best_point = sum(delayed_point)
                            random_best_all_point = sum(delayed_all_point)
                            random_best_transfer = delayed_trans_ind.copy()
                            
                            break
                        
                            #else:
                                #print('Did not move', t, 'to', potential_move_to)
                            
    
    return [random_points, random_prices, random_all_points, random_evaluated_transfers, random_sum_points, random_counts]

def check_guided_transfers(i, random_best_transfer, random_reference_point, unlimited_transfers, free_transfers):
   
    

    guided_evaluated_transfers = []
    guided_points = []
    guided_prices = []
    guided_all_points = []

    guided_counts = np.zeros((len(point_diff), len(probabilities[0])), dtype='uint32')
    guided_sum_points = np.zeros((len(point_diff), len(probabilities[0])))

    #loop to get the transfer combination
    guided_transfer_ind = []
    guided_putative_transfers = []
    for j in random_best_transfer:
        guided_transfer_ind.append(j)
        guided_putative_transfers.append(transfers[j])

    random_ordered_transfers = list(range(len(transfers)))
    random.shuffle(random_ordered_transfers)

    # guided_original_transfer = np.array(guided_putative_transfers).copy()

    # guided_original_team_ind = np.where(slim_elements_df['picked'].values)

    #exhange one of the transfers
    for j in random_ordered_transfers:
        if prob[j, i] > 0:
            
            

            guided_transfer_ind[i] = j
            incomming_transfer = transfers[guided_transfer_ind[i]]

            guided_putative_transfers[i] = incomming_transfer                   


            #chack that only one of the incoming/outgoing players are in the team
            if j == len(transfers)-1 or (sum(incomming_transfer[1] ==  np.array(guided_putative_transfers)[:, 1]) == 1 and  sum(incomming_transfer[0] ==  np.array(guided_putative_transfers)[:, 0]) == 1):
                #check
                guided_point, guided_price, guided_all_point = objective(guided_putative_transfers, unlimited_transfers, free_transfers)
                guided_points.append(guided_point)
                guided_prices.append(guided_price)
                guided_all_points.append(guided_all_point)
                guided_evaluated_transfers.append(guided_transfer_ind.copy())

                if not np.isnan(np.sum(guided_point)):
                    #print(j, i)
                    guided_sum_points[i, guided_transfer_ind[i]] += (np.sum(guided_point)-random_reference_point)
                    guided_counts[i, guided_transfer_ind[i]] += 1

                #punish also nan teams
                else:
                    guided_counts[i, guided_transfer_ind[i]] += 1
                    

    return guided_points, guided_prices, guided_all_points, guided_evaluated_transfers, guided_sum_points, guided_counts


#et free hit team
transfer_cost = 0
player_iteration = 15
gw_iteration = rounds_to_value


#initiate probabilities based on predictions.
#start out by putting some to nan and other to it's predicitio


#loop players
#loop gws
free_hit_points = []

if True: #unlimited_transfers:
    for i in range(gw_iteration):
        free_hit_points.append(0)
else:
    
    for i in range(gw_iteration):
        
        if free_hit[i] or skip_free_hit_calc:
            free_hit_points.append(0)
            continue
        
        point_diff = []
        
        
        for j in range(player_iteration):

    
            transfers = []
            probability = []
            
            #this counts the number of picked players we have assessed
            ind_next = 0
    
            #loop transfers
            for player_out in slim_elements_df.iterrows():
                # ind = 543
                # player_out = (ind, slim_elements_df.iloc[ind])
                
                #check if picked
                if player_out[1]['picked']:
    
                    for player_in in slim_elements_df.iterrows():
                        # ind = 289
                        # player_in = (ind, slim_elements_df.iloc[ind])
    
                        #check if not picked, not same the other player, any predictions >0 and same element
                        if (not player_in[1]['picked']) and sum(player_in[1].prediction) > 0 and (any(player_in[1].prediction > player_out[1].prediction) or player_in[1].now_cost < player_out[1].now_cost) and player_in[1].element_type == player_out[1].element_type:                        
                            
                            if not player_in[1].element_type == player_out[1].element_type:
                                print('Different position should not happen')
                                a=djdjdjdj
                                
                            transfers.append([player_out[0], player_in[0]])
                            
                            #print(j, ind_next)
                            #the ind_next makes sure that in each column only one player is transfered out
                            if unlimited_transfers and j is not ind_next:
                                # if player_out[0] == 543:
                                #     if player_out[0] == 543:                                
                                #         print(player_in[0], j, ind_next)
                                probability.append(np.nan)
                                continue
                            
                            #if lower prediction and higher cost.
                            if player_in[1].prediction[i] <= player_out[1].prediction[i] and (player_in[1].now_cost >= player_out[1].now_cost):
                                probability.append(np.nan)
                                continue
                            
                            preds = np.cumsum((all_gws_predictions[player_in[0]] - all_gws_predictions[player_out[0]])[::-1])[::-1]
    
                            probability.append(preds[i])
    
                            
                    #add one for each player out
                    ind_next += 1
    
    
    
    
            #add no transfer
            probability.append(4)
            transfers.append([np.nan, np.nan])
    
            point_diff.append(probability)
            
        #get free hit team
        probabilities = np.array(point_diff)
    
        counts = np.ones((1, len(probabilities[0])), dtype='uint32')
        p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
        prob = (p.T) / np.nansum((p.T), axis=0)
        selected = np.isnan(prob)
        prob[selected] = 0
        
        counter = 1
        batch_size = 1000
        baseline_point = 0
        predictions = all_gws_predictions[:, i]
        #need threading for parallel because of subprocess module not found
        
        parallel_results = Parallel(n_jobs=num_jobs)(delayed(check_random_transfers)(i, True, free_transfers) for i in range(counter, counter+num_jobs))
        
        best_points = -np.inf
        best_price = np.inf
        best_all_points = -np.inf
        #store data for later
        #organize_output    
        for par in parallel_results:           
                
            #to get the last most positive
            sum_points = [np.sum(inner_list) for inner_list in par[0]]
            
            max_points = np.nanmax(sum_points)
            
            max_indices = np.where(sum_points == max_points)[0]
            
            #loop_those in reverse order (because of delayed transfer)
            for ind_max in max_indices[::-1]:
                if max_points > best_points or (max_points == best_points and par[1][ind_max] < best_price) or (max_points == best_points and par[1][ind_max] == best_price and sum(par[2][ind_max]) >  best_all_points):
                    best_points =  max_points
                    best_transfer = par[3][ind_max]
                    best_price = par[1][ind_max]
                    best_all_points = sum(par[2][ind_max])
                    
    
        print('\nFree hit team GW', i+1)
            
        print('Points:', np.round(best_points, decimals=1))
        
        selected = slim_elements_df.picked == True
        
        transfered_out = []
        
        for ind, k in enumerate(best_transfer):
            trans = transfers[k]
            if not np.isnan(trans[1]):
                print(slim_elements_df.iloc[trans[0]]['web_name'], 'for', slim_elements_df.iloc[trans[1]]['web_name'])
                transfered_out.append(slim_elements_df.iloc[trans[0]]['web_name'])
                
        for p in slim_elements_df.loc[selected].iterrows():
            if not p[1].web_name in transfered_out:
                print(p[1].web_name)
                
        free_hit_points.append(best_points)

predictions = all_gws_predictions

if unlimited_transfers:
    transfer_cost = 0
    gw_iteration = 1
    player_iteration = 15
else:
    transfer_cost = 4
    gw_iteration = rounds_to_value
    player_iteration = 1



point_diff = []

#initiate probabilities based on predictions.
#start out by putting some to nan and other to it's predicitio

#loop players
for j in range(player_iteration):

    #loop gws
    for i in range(gw_iteration):
        transfers = []
        probability = []
        probability_hit = []
        
        #this counts the number of picked players we have assessed
        ind_next = 0

        #loop transfers
        for player_out in slim_elements_df.iterrows():
            # ind = 543
            # player_out = (ind, slim_elements_df.iloc[ind])
            
            #check if picked
            if player_out[1]['picked']:

                if player_out[1]['web_name'] in do_not_transfer_out:
                    continue

                for player_in in slim_elements_df.iterrows():
                    # ind = 289
                    # player_in = (ind, slim_elements_df.iloc[ind])

                    #check if not picked, not same the other player, any predictions >0 and same element
                    if (not player_in[1]['picked']) and sum(player_in[1].prediction) > 0 and (any(player_in[1].prediction > player_out[1].prediction) or player_in[1].now_cost < player_out[1].now_cost) and  player_in[1].element_type == player_out[1].element_type:                        
                        
                        transfers.append([player_out[0], player_in[0]])
                        
                        #skip if free hit
                        if free_hit[i]:
                            probability.append(np.nan)
                            probability_hit.append(np.nan)
                            continue
                        
                        #print(j, ind_next)
                        #the ind_next makes sure that in each column only one player is transfered out
                        if unlimited_transfers and j is not ind_next:
                            # if player_out[0] == 543:
                            #     if player_out[0] == 543:                                
                            #         print(player_in[0], j, ind_next)
                            probability.append(np.nan)
                            probability_hit.append(np.nan)
                            continue
                        
                        # if player_out[0] == 543:
                        #     if player_in[0] == 289:                                
                        #         print(player_in[0], j, ind_next)

                        #if more expensive and less gain
                        if not unlimited_transfers:
                            if player_in[1].prediction[i] <= player_out[1].prediction[i] and (player_in[1].now_cost >= player_out[1].now_cost):
                                probability.append(np.nan)
                                probability_hit.append(np.nan)
                                continue
                        else:
                            if sum(player_in[1].prediction > player_out[1].prediction) == 0 and (player_in[1].now_cost >= player_out[1].now_cost):
                                probability.append(np.nan)
                                probability_hit.append(np.nan)
                                continue 


                        preds = np.cumsum((predictions[player_in[0]] - predictions[player_out[0]])[::-1])[::-1]

                        probability.append(preds[i])


                        #for hit we cannot accept lower score and we need a cumulative 4 point increase at somepoint during the run
                        if not unlimited_transfers:                            
                            if (player_in[1].prediction[i] < player_out[1].prediction[i]):
                                probability_hit.append(np.nan)
                            else:
                                probability_hit.append(preds[i])
                        else:
                            if sum(player_in[1].prediction > player_out[1].prediction) == 0:
                                probability_hit.append(np.nan)
                            else:
                                probability_hit.append(preds[i])
                        
                #add one for each player out
                ind_next += 1


        #add no transfer
        probability.append(4)
        probability_hit.append(4)
        transfers.append([np.nan, np.nan])
        
        #for each player-gw: add the probability into the initating variables. 3 transfers per round.
        if unlimited_transfers:
            point_diff.append(probability)
        #if all are nan for hits (no hots possible) and not wild card
        else:

            #n transfers
            for k in range(trans_per_week):
                #add a transfer per round
                #first transfer must be a gain
                if k==0:
                    point_diff.append(probability_hit)
                else:
                    #these can also be lower price and less gain to accomodate the first
                    point_diff.append(probability)
                    

probabilities = np.array(point_diff.copy())

#get baseline
no_transfers = []
for i in range(len(point_diff)):
    no_transfers.append([np.nan, np.nan])

    
for benchboost_gw in benchboost_gws:
    
    benchboost = []
    
    for i in range(jump_rounds, rounds_to_value+jump_rounds):
        this_gw = i + current_gameweek
        
        if benchboost_gw == this_gw:
            benchboost.append(True)
            print('Benchboost', benchboost_gw)
        else:
            benchboost.append(False)


    print('Check baseline')
    
    
    #check current team
    baseline_point, baseline_price, baseline_all_point = objective(no_transfers, unlimited_transfers, free_transfers)
    
    
    best_points = sum(baseline_point)
    best_all_points = sum(baseline_all_point)
    best_price = baseline_price
    counts = np.ones((len(no_transfers), len(probabilities[0])), dtype='uint32')
    best_transfer = [len(transfers)-1 for _ in range(15)]
    
    best_pitch = baseline_point.copy()
    best_bench = [a - b for a, b in zip(baseline_all_point, baseline_point)]





    try:
        print('Check saved transfers')
        #load saved_transfers
        with open(r'\\platon.uio.no\med-imb-u1\jorgels\best_transfers.pkl', 'rb') as file:
            saved_transfers = pickle.load(file)
            
        check_transfers = []
        for k in saved_transfers:
            check_transfers.append(transfers[k])
                                
        saved_point, saved_price, saved_all_points = objective(check_transfers, unlimited_transfers, free_transfers)
        
        if np.sum(saved_point) > best_points or (np.sum(saved_point) == best_points and saved_price < best_price) or (np.sum(saved_point) == best_points and saved_price == best_price and np.sum(saved_all_point) > best_all_points):
            best_point = saved_point 
            best_price = saved_price
            best_all_points = saved_all_points
            best_transfer = saved_transfers
            
            best_pitch = best_point.copy()
            best_bench = [a - b for a, b in zip(best_all_points, best_point)]
    except:
        print('Saved transfers did not work')
                
    
    all_evaluated_transfers = [no_transfers]
    
    p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
    prob = (p.T) / np.nansum((p.T), axis=0)
    selected = np.isnan(prob)
    prob[selected] = 0
    
    print('Check guided transfers')
    
    check_guided = True
    
    try:
        while check_guided:
            check_guided = False
            #do guided search on 
            random_order = list(range(prob.shape[1]))
            random.shuffle(random_order)
            
            #guided part. exhange one transfer
            for k in random_order:
                
                #if there are more than one transfer to choose from
                if sum(prob[:, k] > 0) < 2:
                    continue
                
                guided_points, guided_prices, guided_all_points, guided_evaluated_transfers, guided_sum_points, guided_counts = check_guided_transfers(k, best_transfer, best_points, unlimited_transfers, free_transfers)
            
                #max points
                #random variables now includes both
                guided_max_value = np.nanmax([np.nansum(inner_list) for inner_list in guided_points])
                #lowest price
                guided_indices_with_max_value = [i for i, value in enumerate(guided_points) if np.nansum(value) == guided_max_value]
                guided_min_value_other_list = min(guided_prices[i] for i in guided_indices_with_max_value)
                guided_best_ind = next(i for i in guided_indices_with_max_value if guided_prices[i] == guided_min_value_other_list)
            
                guided_best_price = guided_prices[guided_best_ind]
            
                #highest total points
                guided_best_point = 0
                for i in range(len(guided_all_points)):
                    if np.nansum(guided_points[i]) == guided_max_value and guided_prices[i] == guided_best_price and np.nansum(guided_all_points[i]) > guided_best_point:
                        guided_best_point = sum(guided_all_points[i])
                        guided_best_ind = i
                
                if guided_max_value > best_points or (guided_max_value == best_points and guided_best_price < best_price) or (guided_max_value == best_points and guided_best_price == best_price and  guided_best_point > best_all_points):
                    
                    check_guided = True
                    best_points = sum(guided_points[guided_best_ind])
                    best_price = guided_best_price
                    best_all_points = guided_best_point.copy()
                    best_transfer = guided_evaluated_transfers[guided_best_ind].copy()
                    
                    best_pitch = guided_points[guided_best_ind].copy()
                    best_bench = [a - b for a, b in zip(guided_all_points[guided_best_ind], guided_points[guided_best_ind])]
    except:
        print('Not able to guide best transfers')
    
    
    counter = 0
    old_num_teams = 0
    
    
    if rounds_to_value == 1:
        batch_size = 1
    else:
        batch_size = 100000
    
    import time
    
    while counter < iterations:
    
        all_evaluated_transfers = []
    
        if counter > 0:
            print('Start random selections', counter)
        
            p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
            prob = (p.T) / np.nansum((p.T), axis=0)
            selected = np.isnan(prob)
            prob[selected] = 0
        
            #guessing part. try random combination followed up by a targeted selection
            print('Getting  teams')
            t1_start = time.time()
            
            parallel_results = Parallel(n_jobs=num_jobs)(delayed(check_random_transfers)(i, unlimited_transfers, free_transfers) for i in range(counter, counter+num_jobs))
            t1_stop = time.time()
            print("Elapsed time:", t1_stop - t1_start)
            print('Interpreting results')
        
        
            #store data for later
            #organize_output                   
            for par in parallel_results:
                                   
                        
                #to get the last most positive
                sum_points = [np.sum(inner_list) for inner_list in par[0]]
                
                max_points = np.nanmax(sum_points)
                
                max_indices = np.where(sum_points == max_points)[0]
                
                #loop_those in reverse order (because of delayed transfer)
                for ind_max in max_indices[::-1]:
                    if max_points > best_points or (max_points == best_points and par[1][ind_max] < best_price) or (max_points == best_points and par[1][ind_max] == best_price and sum(par[2][ind_max]) >  best_all_points):
                        best_points =  max_points
                        best_transfer = par[3][ind_max]
                        best_price = par[1][ind_max]
                        best_all_points = sum(par[2][ind_max])
                        
                        best_pitch = par[0][ind_max].copy()
                        best_bench = [a - b for a, b in zip(par[2][ind_max], best_pitch)]
                        
                        counter = 0
                        
                        
                        #save transfers
                        with open(r'\\platon.uio.no\med-imb-u1\jorgels\best_transfers.pkl', 'wb') as file:
                            pickle.dump(best_transfer, file)
                        
                    
        
                all_evaluated_transfers = all_evaluated_transfers + par[3]
                
                #the first prob of each week is different than the others
                k = 0
                for w in range(rounds_to_value):
                    for t_res in range(trans_per_week):
                        index = w*trans_per_week + t_res
        
                        probabilities[index, :] += par[4][index, :]
                        counts[index, :] += par[5][index, :]
    
    
            
    
        # print('Checked', len(all_evaluated_transfers)-old_num_teams, 'teams')
        # old_num_teams = len(all_evaluated_transfers)
    
        # # Convert each list to a tuple
        # unique_tuples = set(tuple(x) for x in all_evaluated_transfers)
        # # Convert the tuples back to lists
        # all_evaluated_transfers = [list(x) for x in unique_tuples]
    
        # print(len(all_evaluated_transfers), 'unique teams')
    
        counter += 1
        
        if len(best_transfer) == 0:
            print('No acceptable teams')
            continue
    
        #print results
        price = []
        last_gw = 0
        try:
            with open(r"\\platon.uio.no\med-imb-u1\jorgels\best_transfers.pkl" + str(benchboost_gw) + ".txt", 'w') as file:
            
                print('gw', 'free_hit', 'bench', file=file)
                print('gw', 'free_hit', 'bench')
                
                for gw_ind, transfer_ind in enumerate(best_transfer):
            
                    transfer = transfers[transfer_ind]
                    
                    gw = int(1+gw_ind/trans_per_week)
                    
                    if not gw == last_gw and not unlimited_transfers:
                        #print('\n')
                        print('GW', gw, np.round(free_hit_points[gw-1] - best_pitch[gw-1], decimals=1), np.round(best_bench[gw-1], decimals=1), file=file)
                        print('GW', gw, np.round(free_hit_points[gw-1] - best_pitch[gw-1], decimals=1), np.round(best_bench[gw-1], decimals=1))
                        
                        last_gw = gw
                        if np.round(best_bench[gw-1], decimals=1) < 0:
                            a = hfhfhfff
            
                    if not transfer == [np.nan, np.nan]:
                        price.append(slim_elements_df.loc[transfer[1], 'now_cost'])
            
                        if not unlimited_transfers:
                            print( slim_elements_df.loc[transfer[0], 'web_name'], 'for', slim_elements_df.loc[transfer[1], 'web_name'], np.round(prob[transfer_ind, gw_ind], 4), file=file)
                            print( np.round(predictions[transfer[0], :], decimals=1), file=file)
                            print( np.round(predictions[transfer[1], :], decimals=1), file=file)
                            #print(prob[transfer_ind, gw_ind])
                        else:
                            print(int(gw_ind), slim_elements_df.loc[transfer[1], 'web_name'], np.round(predictions[transfer[1], :], 1),  np.round(prob[transfer_ind, gw_ind], 4), file=file)
                        
                        if not unlimited_transfers:
                            print( slim_elements_df.loc[transfer[0], 'web_name'], 'for', slim_elements_df.loc[transfer[1], 'web_name'], np.round(prob[transfer_ind, gw_ind], 4))
                            print( np.round(predictions[transfer[0], :], decimals=1))
                            print( np.round(predictions[transfer[1], :], decimals=1))
                            #print(prob[transfer_ind, gw_ind])
                        else:
                            print(int(gw_ind), slim_elements_df.loc[transfer[1], 'web_name'], np.round(predictions[transfer[1], :], 1),  np.round(prob[transfer_ind, gw_ind], 4))
            
            
                    else:
                        if unlimited_transfers:
                            try:
                                max_ind = np.nanargmax(p[gw_ind, :-1])
                                transfer = transfers[max_ind]
                                print(int(gw_ind), slim_elements_df.loc[transfer[0], 'web_name'], np.round(predictions[transfer[0], :], 1), np.round(prob[transfer_ind, gw_ind], 4))
                                print(int(gw_ind), slim_elements_df.loc[transfer[0], 'web_name'], np.round(predictions[transfer[0], :], 1), np.round(prob[transfer_ind, gw_ind], 4), file=file)
                                price.append(slim_elements_df.loc[transfer[0], 'now_cost'])
                            except:
                                print('Not able to print')
                                print('Not able to print', file=file)
                                    
                        
                print('points: ', np.round(sum(best_pitch), decimals=1), '. diff: ',  np.round(best_points-sum(baseline_point), decimals=1), '. price: ', sum(price), file=file)
                print('points: ', np.round(sum(best_pitch), decimals=1), '. diff: ',  np.round(best_points-sum(baseline_point), decimals=1), '. price: ', sum(price))
                print('\n')
        except:
            print('Not able to open file')
        
        
        
        


