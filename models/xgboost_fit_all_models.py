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

               


with open(r"\\platon.uio.no\med-imb-u1\jorgels\\element_data.pkl", 'rb') as file:
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
        
        
        
train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\all_data.pkl')  # Set index=False to not include row indices

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


model_path = r"\\platon.uio.no\med-imb-u1\jorgels\all_model.sav"

pickle.dump(summary, open(model_path, 'wb'))
    