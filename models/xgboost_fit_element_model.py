import os
import re
import pickle
import random
import math

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


optimize = True
continue_optimize = False

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

    q = 0.4
    
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
    objective_X = cv_X.loc[cv_selected, columns_to_keep]   
        
    # interaction_constraints = get_interaction_constraints(objective_X.columns)
    # pars['interaction_constraints'] = str(interaction_constraints)       
    # Step 2: Calculate 20% of the unique integers
    
    # Step 2: Calculate X% of the unique integers
    # eval_num_to_select = max(1, int(len(cvs_match_integers) * space['eval_fraction']))  # Ensure at least one is selected
    
    # random.seed(44)
    
    # eval_sample = random.sample(cvs_match_integers, eval_num_to_select)
    

    
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[cvs_mask][cv_selected.values])

    
    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_matches = list(set().union(*season_selection.values()))
    
    
    fits_mask =  pd.Series(match_ind_df[cvs_mask][cv_selected.values]).isin(fit_matches)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask
    
    #remove features
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                
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
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y[cv_selected].loc[fits_mask.values].copy()
    eval_y = cv_y[cv_selected].loc[evals_mask.values].copy()

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
    num_boost_round=int(space['n_estimators']),
    early_stopping_rounds= int(space['early_stopping_rounds']),
    dtrain=dfit,
    evals=evals,
    custom_metric=custom_metric,
    obj=custom_objective,
    verbose_eval=False  # Set to True if you want to see detailed logging
        )
    
    if not model.get_score():
        val_error = np.inf
    else:

        objective_val_X = val_X.loc[val_selected, columns_to_keep]
        dval_objective = xgb.DMatrix(data= objective_val_X, label=val_y[val_selected], enable_categorical=True)
    
        val_pred = model.predict(dval_objective)
        
        val_error = mean_squared_error(val_y[val_selected],  val_pred)
        #val_error = mean_squared_error(val_y,  (10**val_pred) - 1 + min_y)
        
    
        

    return {'loss': val_error, 'status': STATUS_OK }



#optimize hyperparameters
def plot_objective_xgboost(space):
    
    # print(space)
    
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
    objective_X = cv_X.loc[cv_selected, columns_to_keep]   
        
    # interaction_constraints = get_interaction_constraints(objective_X.columns)
    # pars['interaction_constraints'] = str(interaction_constraints)       
    # Step 2: Calculate 20% of the unique integers
    
    # Step 2: Calculate X% of the unique integers
    # eval_num_to_select = max(1, int(len(cvs_match_integers) * space['eval_fraction']))  # Ensure at least one is selected
    
    # random.seed(44)
    
    # eval_sample = random.sample(cvs_match_integers, eval_num_to_select)
    

    
    # Get the 80% of the first matches every season...
    objective_copy = objective_X.copy()
    objective_copy = objective_copy.reset_index(drop=True)
    objective_copy['match_ind'] = pd.Series(match_ind[cvs_mask][cv_selected.values])

    
    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
    
    
    fits_mask =  pd.Series(match_ind_df[cvs_mask][cv_selected.values]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask
    
    #remove features
    for feat in check_features:
        if feat in space.keys():
            #if remove
            if not space[feat]:     
                columns_to_keep = []
                for col in objective_X.columns:
                    if col == feat:
                        continue
                    #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
                    if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
                        columns_to_keep.append(col)
                    
                objective_X = objective_X[columns_to_keep]
                
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
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y[cv_selected].loc[fits_mask.values].copy()
    eval_y = cv_y[cv_selected].loc[evals_mask.values].copy()

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
    
    
    num_features = fit_X.shape[1]
    
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
    
    if not model.get_score():
        val_error = np.inf
    else:    
        objective_val_X = val_X.loc[val_selected, columns_to_keep]
        dval_objective = xgb.DMatrix(data= objective_val_X, label=val_y[val_selected], enable_categorical=True)
    
        val_pred = model.predict(dval_objective)
        
        val_error = mean_squared_error(val_y[val_selected],  val_pred)
        #val_error = mean_squared_error(val_y,  (10**val_pred) - 1 + min_y)
        
    
        
        try:
            xgb.plot_importance(model, importance_type='gain',
                            max_num_features=np.min([20, num_features]), show_values=False)
            plt.show()
        except:
            print('Not able to plot')

    return {'loss': val_error, 'status': STATUS_OK }



def get_interaction_constraints(features):
    #set up interaction_constraints
    interaction_constraints = []
    
    global_features = ['element_type', 'string_team', 'season', 'names', 'points_per_game', 'points_per_played_game']
    #current_features = ['string_opp_team', 'transfers_in', 'transfers_out', 'was_home', 'own_difficulty', 'other_difficulty']    
    
    global_group = []
    current_group = []
    
    week_group = []
    type_group = []
    
    week = 0
    
    for feat_ind, feat in enumerate(features):
        digits = ''.join(re.findall(r'\d', feat))
        letters = ''.join(re.findall(r'[A-Za-z_]', feat))
        
        if digits == '':
            
            if letters in global_features:
                global_group.append(feat_ind)
            else:
                current_group.append(feat_ind)
            
        else:
            if not int(digits) == week:
                week = int(digits)
                interaction_constraints.append(global_group + week_group.copy())
                week_group = []
            
            week_group.append(feat_ind)
            
            # #set up feature type interactions: one category for each feature independent of week
            # if int(digits) == 0:
            #     type_group.append(global_group+[feat_ind])
            # else:
            #     type_group[len(week_group)-1].append(feat_ind)
                
    # #add last week
    # interaction_constraints.append(global_group + week_group.copy())
    # interaction_constraints.append(global_group + current_group)
        #interaction_constraints = interaction_constraints + type_group
                
        
            #all except the dynamic features go into a group.
            if not letters in current_features:
                type_group.append(feat_ind)
        
    
    #add last week
    interaction_constraints.append(global_group + week_group.copy())
    interaction_constraints.append(global_group + current_group)
    interaction_constraints = interaction_constraints + [global_group + current_group + type_group]
    
    return interaction_constraints
                


def objective_linear_reg(space):

    if space['reg'] == 'lasso':
        model = Lasso(alpha=space['alpha'])
    else:
        model = Ridge(alpha=space['alpha'])

    model.fit(scaled_cv_X, log_cv_y)

    val_pred = model.predict(scaled_val_X)
    
    val_normal = np.exp(val_pred) + min_val - 1
    
    val_error = mean_squared_error(val_y,  val_normal)

    return {'loss': val_error, 'status': STATUS_OK }

def objective_svr(space):
    #print(space)

    model = SVR(**space['pars'])
    
    
    model.fit(scaled_cv_X, log_cv_y)

    val_pred = model.predict(scaled_val_X)
    
    val_normal = np.exp(val_pred) + min_val - 1
    
    val_error = mean_squared_error(val_y,  val_normal)

    return {'loss': val_error, 'status': STATUS_OK }

def objective_linear_svr(space):

    #print(space)

    model = LinearSVR(**space, dual="auto")
    
    model.fit(scaled_cv_X, log_cv_y)

    val_pred = model.predict(scaled_val_X)
    
    val_normal = np.exp(val_pred) + min_val - 1
    
    val_error = mean_squared_error(val_y,  val_normal)

    return {'loss': val_error, 'status': STATUS_OK }


with open(r"\\platon.uio.no\med-imb-u1\jorgels\model_local_data.pkl", 'rb') as file:
    train_data = pickle.load(file)                




selected = train_data["minutes"] >= 60
train_data = train_data.loc[selected]

#remove players with few matches
unique_names = train_data.name.unique()

n_tresh = 2

for unique_ind, name in enumerate(unique_names):
    selected = (train_data.name == name)

    if sum(selected) < n_tresh:
        train_data.loc[selected, 'name'] = np.nan


#included for all windows, but not current
# temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
#         'total_points', 'expected_goals', 'expected_assists',
#         'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon']
temporal_features = ['influence', 'threat', 'creativity', 'bps',
        'expected_goals', 'expected_assists',
        'expected_goals_conceded']

train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)
                

            

# Identify categorical columns
categorical_columns = train_X.select_dtypes(['category']).columns

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()

# # Define the number of quantiles/bins
# num_bins = 100

# # Calculate the quantile boundaries of the outcome variable
# centiles = pd.qcut(train_y, q=100, duplicates="drop", retbins=True)[1]
# centiles[0] = -np.inf
# # Discretize the outcome variable using the quantile boundaries
# stratify = pd.cut(train_y, bins=centiles, labels=False)

# min_y = np.min(train_y)
# train_y = np.log10(train_y-min_y+1)

# cw = compute_class_weight('balanced', classes=np.unique(stratify), y=stratify)

# for k in np.unique(stratify):
#     selected = stratify == k
#     sample_weights[selected] = sample_weights[selected]*cw[k]


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
    
    



# Get the 80% of the first matches every season...
train_copy = train_X.copy()
train_copy = train_copy.reset_index(drop=True)
train_copy['match_ind'] = pd.Series(match_ind)
match_ind_df = pd.Series(match_ind) 

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

# groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
season_selection = (train_copy.groupby('season', observed=False)['match_ind']
                      .agg(lambda s: first_Xpct_unique(s.tolist(), 0.8))
                      .to_dict())

# If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
# option 1: Unique across all seasons:
train_sample = list(set().union(*season_selection.values()))


# Step 3: Randomly select 20% of the unique integers
if optimize:
    #9.38
    random.seed(0)
else:
    random.seed(1)



cvs_mask = pd.Series(match_ind_df).isin(train_sample)  # Mask for cross-validation sample
vals_mask = ~cvs_mask  # Mask for validation, simply the inverse of cvs_mask

cvs_match_integers = list(set(match_ind_df[cvs_mask]))


cv_X = train_X.loc[cvs_mask.values].copy()
val_X =  train_X.loc[vals_mask.values].copy()
cv_y =  train_y.loc[cvs_mask.values].copy()
val_y = train_y.loc[vals_mask.values].copy()


#make sure all categories in val_x is present in cv_x
for column in val_X.columns:
    if isinstance(val_X[column].dtype, pd.CategoricalDtype):
        # Get the values in the current column of val_X
        val_values = val_X[column]
        
        # Check which values are present in the corresponding column of cv_X
        mask = val_values.isin(cv_X[column])
        
        # Set values that are not present in cv_X[column] to NaN
        val_X.loc[~mask, column] = np.nan
        
   
    
   

    

train_pos_selected = []
#loop elements
for i in range(1,5):
    
    cv_selected = cv_X.element_type == i
    val_selected = val_X.element_type == i
    train_selected = train_X.element_type == i
    
    train_pos_selected.append([cv_selected, val_selected, train_selected])
    

grow_policy = ['depthwise', 'lossguide']
#include feature search in the hyperparams

#these features define which are kept for analysis
check_features = ['string_team', 'transfers_in', 'transfers_out', 'minutes', 'was_home', 'season', 'ict_index', 'defcon', 'total_points', 'name',
        'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'string_opp_team', 'own_difficulty', 'opp_difficulty', 'local_predictions', 'local_const_predictions'] #, 'difficulty']



#the non digit version of these features will be removed
unknown_features = ['minutes', 'ict_index', 'total_points', 'own_team_points', 'own_element_points', 'defcon', 'opp_team_points', 'opp_element_points']

#remove all features which do not contain any of the check features.
keys = cv_X.keys()
for f in cv_X.keys():
    drop_col = True
    for k in check_features:
        if k in f:
            drop_col = False
            
    if drop_col:       
        cv_X = cv_X.drop(columns = [f])
        val_X = val_X.drop(columns = [f])
        train_X = train_X.drop(columns = [f])



# check_features = ['transfers_in', 'transfers_out', 'minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
#         'total_points', 'expected_goals', 'expected_assists', 'points_per_played_game', 'was_home', 'season',
#         'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon', 'name', 'points_per_game', 'string_opp_team', 'own_difficulty', 'opp_difficulty'] #, 'difficulty']

# individual_features = ['season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_difficulty', 'ict_index', 'transfers_in', 'transfers_out'] #, 'difficulty']   
   
#remove all features which do not contain any of the check features.
keys = cv_X.keys()
for f in cv_X.keys():
    drop_col = True
    for k in check_features:
        if k in f:
            drop_col = False
            
    if drop_col:       
        cv_X = cv_X.drop(columns = [f])
        val_X = val_X.drop(columns = [f])
        train_X = train_X.drop(columns = [f])

        
pos_weighted_error = 0

#loop elements
for pos in range(1,5):
    
    print('Field position', pos)
    
    cv_selected = train_pos_selected[pos-1][0]
    val_selected = train_pos_selected[pos-1][1]
    
    hyperparam_path = main_directory + f'\models\hyperparams{pos}.pkl'
    with open(hyperparam_path, 'rb') as f:
        old_trials = pickle.load(f)
    
    hyperparams = old_trials.best_trial['misc']['vals']
    #reformat the lists
    old_hyperparams = {}
    for field, val in hyperparams.items():
        old_hyperparams[field] = val[0]
        
    
    loss = plot_objective_xgboost(old_hyperparams)
    old_loss = loss['loss']
    
    
    print('Old loss: ', old_loss)
        
    
    
    
    match_count_df = cv_X.loc[cv_selected].copy()        
    match_count_df = match_count_df.reset_index(drop=True)
    match_count_df['match_ind'] = pd.Series(match_ind[cvs_mask][cv_selected.values])

    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    match_counts = (match_count_df.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: len(np.unique(s.tolist()))))


    min_eval_fraction = 1/np.max(match_counts)
    
    
    space={'max_depth': hp.qloguniform("max_depth", 1, np.log(150), 1), 
            'min_split_loss': hp.loguniform('min_split_loss', 0, np.log(250)), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 200),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 350),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 400),
            'learning_rate': hp.uniform('learning_rate', 0, 3),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 3000, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.35),
            'n_estimators': hp.quniform('n_estimators', 2, 12000, 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 100),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1
            'max_leaves': hp.quniform('max_leaves', 0, 60, 1),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(300), 1),
            'temporal_window': hp.quniform('temporal_window', 1, 12, 1),
        }
    
    
    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [True, False]), #111
    
        
    #optimize and iteratively get hyperparamaters
    batch_size = 100
    if optimize:
        max_evals = 500000
        
    temp_hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
    
    if continue_optimize:        
        with open(temp_hyperparam_path, 'rb') as f:
            trials = pickle.load(f)
    else:
        trials = Trials()
    
    if optimize:
        
        loss = np.inf
    
        for i in range(len(trials.trials)+batch_size, max_evals + 1, batch_size):
    
            # Save the trials object every 'batch_size' iterations. Can save with any method you prefer
    
            #optmimize hyperparameters. use all training data
            best_hyperparams = fmin(fn = objective_xgboost,
                            space = space,
                            algo = atpe.suggest,
                            max_evals = i,
                            trials = trials)
    
            
            
            #hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
            pickle.dump(trials, open(temp_hyperparam_path, "wb"))
            
            if trials.best_trial['result']['loss'] == loss:
                break
            else:
                loss = trials.best_trial['result']['loss']
     
                
        
    with open(temp_hyperparam_path, 'rb') as f:
        new_trials = pickle.load(f)
        
    hyperparams = new_trials.best_trial['misc']['vals']
    #reformat the lists
    new_hyperparams = {}
    for field, val in hyperparams.items():
        new_hyperparams[field] = val[0]
        print(field, val[0])
        
    #new_hyperparams["grow_policy"] = grow_policy[new_hyperparams["grow_policy"]]
                       
        
    # new_trials = generate_trials_to_calculate([new_hyperparams])

    # new_hyperparams = fmin(fn = objective_xgboost,
    #                 space = space,
    #                 algo = tpe.suggest,
    #                 max_evals = 1,
    #                 trials = new_trials)    

    # new_loss =  new_trials.best_trial["result"]["loss"]

    loss = plot_objective_xgboost(new_hyperparams)
    new_loss = loss['loss']
    
    print('New loss: ', new_loss)
    
    mean_y = np.mean(cv_y)
    const_error = np.mean((np.abs(val_y[val_selected] - mean_y))**2)
    
    mean_y = np.mean(cv_y[cv_selected])
    const_element_error = np.mean((np.abs(val_y[val_selected] - mean_y))**2)
    
    #revert. shouldn't be any nans
    local_const_error = np.nanmean(np.abs(val_X.loc[val_selected, 'local_const_predictions'] - val_y[val_selected])**2)
    #mean_squared_error(val_X.loc[val_selected, 'local_const_predictions'], val_y[val_selected])
    
    local_error = np.nanmean(np.abs(val_X.loc[val_selected, 'local_predictions'] - val_y[val_selected])**2)
    
    
    print('Constant error: ', const_error)
    print('Constant element error: ', const_element_error)
    print('Local constant error: ', local_const_error)
    print('Local error: ', local_error)
    
    
    if new_loss < old_loss:
        print('Element error: ', new_loss)
        
        print('Overwriting old loss')
        pickle.dump(new_trials, open(hyperparam_path, "wb"))
        trials = new_trials
        
        mse = new_loss
        
        
    else:
        
        print('Element error: ', old_loss)
        
        print('Keep old loss')
        trials = old_trials
        
        mse = old_loss
        
    pos_weighted_error += mse*sum(train_pos_selected[pos-1][1])
    
print('Element weighted error:', pos_weighted_error/len(train_pos_selected[pos-1][1]))

    
    # losses = []
    # for i in range(len(trials.trials)):

    #     if trials.trials[i]['result'] == {'status': 'new'}:
    #         losses.append(9999)
    #         print('Miss result')
    #     else:
    #         losses.append(trials.trials[i]['result']['loss'])

    # sorted_losses = np.argsort(losses)

    
    # best_best_ind = 0

    # #train with all data
    # best_cv_trial =  sorted_losses[best_best_ind]
    # print('Original loss:', losses[best_cv_trial])

    # hyperparams = trials.trials[best_cv_trial]['misc']['vals']
    # #print(hyperparams)

    # space = {}
    # for field, val in hyperparams.items():
    #     space[field] = val[0]

    # pars = {
    #     'max_depth': int(space['max_depth']),
    #     'min_split_loss': space['min_split_loss'],
    #     'reg_lambda': space['reg_lambda'],
    #     'reg_alpha': space['reg_alpha'],
    #     'min_child_weight': int(space['min_child_weight']),
    #     'learning_rate': space['learning_rate'],
    #     'subsample': space['subsample'],
    #     'colsample_bytree': space['colsample_bytree'],
    #     'colsample_bylevel': space['colsample_bylevel'],
    #     'colsample_bynode': space['colsample_bynode'],
    #     'max_delta_step': space['max_delta_step'],
    #     'grow_policy': grow_policy[space['grow_policy']],
    #     'max_leaves': int(space['max_leaves']),
    #     'tree_method': 'hist',
    #     'max_bin':  int(space['max_bin']),
    #     'disable_default_eval_metric': 1
    #     }

    # #remove weaks that we don't need.
    # # Define the threshold
    # threshold = int(space['temporal_window'])

    # # Filter the columns based on the defined function
    # columns_to_keep = [col for col in train_X.columns if should_keep_column(col, threshold)]
    
    
    
    
    # pos_selected = train_pos_selected[pos-1][2]
    
    # objective_X = train_X.loc[pos_selected, columns_to_keep]
    
    # # Get the 80% of the first matches every season...
    # objective_copy = objective_X.copy()
    # objective_copy = objective_copy.reset_index(drop=True)
    # objective_copy['match_ind'] = pd.Series(match_ind[pos_selected])

    
    # # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    # season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
    #                       .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
    #                       .to_dict())
    
    # # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # # option 1: Unique across all seasons:
    # fit_sample = list(set().union(*season_selection.values()))
    
    
    # fits_mask =  pd.Series(match_ind_df[pos_selected.values]).isin(fit_sample)  # Mask for cross-validation sample
    # evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask
    
    # #remove features
    # for feat in check_features:
    #     if feat in space.keys():
    #         #if remove
    #         if not space[feat]:     
    #             columns_to_keep = []
    #             for col in objective_X.columns:
    #                 if col == feat:
    #                     continue
    #                 #keep if it foes not have a number in front or first is not a digit (i.e. the fixed features)
    #                 if (not feat == re.sub(r'\d+', '', col) or not col[0].isdigit()):
    #                     columns_to_keep.append(col)
                    
    #             objective_X = objective_X[columns_to_keep]
                
    
    
    # fit_X = objective_X.iloc[fits_mask.values].copy()
    # eval_X =  objective_X.loc[evals_mask.values].copy()
    # fit_y =  train_y[pos_selected].loc[fits_mask.values].copy()
    # eval_y = train_y[pos_selected].loc[evals_mask.values].copy()
    

    # #make sure all categories in val_x is present in cv_x
    # for column in eval_X.columns:
    #     if isinstance(eval_X[column].dtype, pd.CategoricalDtype):
    #         # Get the values in the current column of val_X
    #         val_values = eval_X[column]
            
    #         # Check which values are present in the corresponding column of cv_X
    #         mask = val_values.isin(fit_X[column])
            
    #         # Set values that are not present in cv_X[column] to NaN
    #         eval_X.loc[~mask, column] = np.nan
    
    # dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True)
    # deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True)

    # evals = [(dfit, 'train'), (deval, 'eval')]

    # model = xgb.train(
    # params=pars,
    # num_boost_round=int(space['n_estimators']),
    # early_stopping_rounds= int(space['early_stopping_rounds']),
    # dtrain=dfit,
    # evals=evals,
    # custom_metric=custom_metric,
    # obj=custom_objective,
    # verbose_eval=False  # Set to True if you want to see detailed logging
    #     )
    

    # summary = {'model': model, 'train_features': objective_X, 'hyperparameters': space}#, 'all_rows': original_df}
    
    # model_path = r'\\platon.uio.no\med-imb-u1\jorgels\model' + f'{pos}.sav'
    
    # pickle.dump(summary, open(model_path, 'wb'))
    
    # try:
    #     xgb.plot_importance(model, importance_type='gain',
    #                     max_num_features=20, show_values=False)
    #     plt.show()
    # except:
    #     print('Not able to plot')
    
    # data =  model.get_score()

    # # Dictionary to hold summed values and counts
    # summed_values = {}
    # count_values = {}

    # for key, value in data.items():
    #     # Extract the part of the string after the digits
    #     new_key = ''.join(filter(lambda x: not x.isdigit(), key))  # or use re.sub(r'^\d+', '', key)
        
    #     # Sum the values and count the occurrences for the same new_key
    #     if new_key in summed_values:
    #         summed_values[new_key] += value
    #         count_values[new_key] += 1
    #     else:
    #         summed_values[new_key] = value
    #         count_values[new_key] = 1

    # # Calculate mean for each key
    # mean_values = {k: summed_values[k] / count_values[k] for k in summed_values}

    # # Sort the mean values by their values
    # sorted_mean_values = dict(sorted(mean_values.items(), key=lambda item: item[1]))

    # #print(sorted_mean_values)  # Output will be sorted by mean values

    # # Plotting the sorted mean values
    # plt.figure(figsize=(10, 6))
    # plt.bar(sorted_mean_values.keys(), sorted_mean_values.values(), color='skyblue')
    # plt.xlabel('Labels')
    # plt.ylabel('Mean Values')
    # plt.title('Mean Values of Points Sorted')
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.tight_layout()  # Adjust layout to prevent clipping of labels

    # # Show the plot
    # plt.show()
    
    # train_data = xgb.DMatrix(data=objective_X, label=train_y[pos_selected], enable_categorical=True)
    # pred = model.predict(train_data)
    
    # train_error = np.mean(np.abs((train_y[pos_selected] - pred)**2))
    
    # print('Train error:', train_error)