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

               


with open(r"\\platon.uio.no\med-imb-u1\jorgels\model_data.pkl", 'rb') as file:
    train_data = pickle.load(file)                



selected = train_data["minutes"] >= 60
train_data = train_data.loc[selected]

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
            'expected_goals_conceded', 'defcon', 'own_element_points']
    
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
    do_remove_features= ['names', 'points_per_game', 'points_per_played_game', 'season']   
    

    
    space={'max_depth': hp.qloguniform("max_depth", 1, np.log(100), 1), 
            'min_split_loss': hp.loguniform('min_split_loss', 0, np.log(40)), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 250),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 70),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 70),
            'learning_rate': hp.loguniform('learning_rate', 0, np.log(7)),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 400, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.51),
            'n_estimators': hp.quniform('n_estimators', 2, 10000, 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 400),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1
            'max_leaves': hp.quniform('max_leaves', 0, 200, 1),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(130), 1),
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
        
        train_data.loc[rows_to_update, 'local_predictions'] = val_pred

        #print(val_pred)        
        print('done', random_error-val_error)
        
        

for k in hyper_vals:
    plt.figure()
    #print(k, np.max(np.array(hyper_vals[k])[selected]))
    plt.hist(np.array(hyper_vals[k]))
    plt.title(k)
    
plt.show()

local_error = np.nanmean((train_y - train_data['local_predictions'])**2)
print('Local error:', local_error)

        
train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\model_local_data.pkl')  # Set index=False to not include row indices