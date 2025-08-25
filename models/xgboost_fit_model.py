import os
import re
import pickle
import requests
import random

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

import xgboost as xgb
import statsmodels.api as sm

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss

from pandas.api.types import CategoricalDtype
import time

import difflib
from difflib import SequenceMatcher

directories = r'C:\Users\jorgels\Documents\GitHub\Fantasy-Premier-League\data'
try:
    folders = os.listdir(directories)
    model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"
    main_directory = r'C:\Users\jorgels\Documents\GitHub\Fantasy-Premier-League'
except:
    directories = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data'
    folders = os.listdir(directories)
    model_path = r"M:\model.sav"
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


optimize = True
continue_optimize = False

#add 2. one because threshold is bounded upwards. and one because last week is only partly encoded (dynamic features)
#+1. e.g 28 here means 29 later.
temporal_window = 30



method = 'xgboost'

season_dfs = []

season_count = 0

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
    
    #remove features
    for feat in check_features:
        if feat in space.keys(:)
            if not space[feat][0]:        
                columns_to_keep = [col for col in objective_X.columns if not feat == re.sub(r'\d+', '', col)]
                objective_X = objective_X[columns_to_keep]  
    
    # interaction_constraints = get_interaction_constraints(objective_X.columns)
    # pars['interaction_constraints'] = str(interaction_constraints)       
    # Step 2: Calculate 20% of the unique integers
    
    # Step 2: Calculate 20% of the unique integers
    eval_num_to_select = max(1, int(len(cvs_match_integers) * space['eval_fraction']))  # Ensure at least one is selected
    
    random.seed(44)
    
    eval_sample = random.sample(cvs_match_integers, eval_num_to_select)
            
    evals_mask = pd.Series(match_ind_df[cvs_mask]).isin(eval_sample)  # Mask for cross-validation sample
    fits_mask = ~evals_mask  # Mask for validation, simply the inverse of cvs_mask
    
    fit_X = objective_X.iloc[fits_mask.values].copy()
    eval_X =  objective_X.loc[evals_mask.values].copy()
    fit_y =  cv_y.loc[fits_mask.values].copy()
    eval_y = cv_y.loc[evals_mask.values].copy()

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

    objective_val_X = val_X[columns_to_keep]
    dval_objective = xgb.DMatrix(data= objective_val_X, label=val_y, enable_categorical=True)

    val_pred = model.predict(dval_objective)
    
    val_error = mean_squared_error(val_y,  val_pred)
    #val_error = mean_squared_error(val_y,  (10**val_pred) - 1 + min_y)

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


with open(r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\model_data.pkl', 'rb') as file:
    train_data = pickle.load(file)                


selected = train_data["minutes"] >= 60
train_data = train_data.loc[selected]
train_data = train_data.drop(['minutes'], axis=1)                


#remove players with few matches
unique_names = train_data.name.unique()

n_tresh = 3

for unique_ind, name in enumerate(unique_names):
    selected = (train_data.name == name)

    if sum(selected) < n_tresh:
        train_data.loc[selected, 'name'] = np.nan
                

                
                
    # #weight samples for time
    # # last_year = season_df['kickoff_time'].iloc[-1] - season_df['kickoff_time']
    # # selected = last_year < timedelta(365)
    # # sample_weights = np.ones(selected.shape)
    # # sample_weights[selected] = 4
    
    # #season_df.replace(to_replace=[None], value=np.nan, inplace=True)
    
    # #train model. no changes of catgeories in train_X after this point!
    # train_X = train.drop(['total_points'], axis=1)
    # train_y = train['total_points'].astype(int)
    
    # # Identify categorical columns
    # #categorical_columns = train_X.select_dtypes(['category']).columns
    # #categories for dtype
    
    # categorical_variables = ['element_type', 'string_team', 'season', 'name']
    # dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
    #         'other_difficulty'] #'difficulty',
    
    # season_df[categorical_variables] = season_df[categorical_variables].astype('category')
    # # #add nan categories
    # # dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
    # #         'other_difficulty'] #'difficulty',
    
    # # int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon', 'SoT']
    # # season_df[int_variables] = season_df[int_variables].astype('Int64')
    
    # # float_variables = ['transfers_in', 'transfers_out', 'threat', 'own_element_points',  'expected_goals', 'expected_assists',
    # # 'expected_goal_assists', 'expected_goals_conceded', 'creativity', 'ict_index', 'influence']
    # # season_df[float_variables] = season_df[float_variables].astype('float')
    
    # # Reset categories for each categorical column
    # season_df[categorical_variables] = season_df[categorical_variables].astype('category')
    
    # # Reset categories for each categorical column
    # for column in categorical_columns:
    #     train_X[column] = train_X[column].cat.remove_unused_categories()
        
        
    # for column_X in train_X.keys():
    #     for column_cat in dynamic_categorical_variables:
    #         if column_cat in column_X:
    #             print('Set to categorical', column_X)
    #             train_X[column_X] = train_X[column_X].astype('category')
    #             train_X[column_X] = train_X[column_X].cat.remove_unused_categories()

#weight samples for time
# last_year = season_df['kickoff_time'].iloc[-1] - season_df['kickoff_time']
# selected = last_year < timedelta(365)
# sample_weights = np.ones(selected.shape)
# sample_weights[selected] = 4

#season_df.replace(to_replace=[None], value=np.nan, inplace=True)

#train model. no changes of catgeories in train_X after this point!
train_X = train_data.drop(['total_points'], axis=1)
train_y = train_data['total_points'].astype(int)

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
   
#get 20% of those matches
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
    
    
unique_integers = list(set(match_ind))

# Step 2: Calculate 20% of the unique integers
num_to_select = max(1, int(len(unique_integers) * 0.80))  # Ensure at least one is selected

# Step 3: Randomly select 20% of the unique integers
if optimize:
    #9.38
    random.seed(42)
else:
    random.seed(43)

train_sample = random.sample(unique_integers, num_to_select)

match_ind_df = pd.Series(match_ind) 

# vals = [x not in train_sample for x in match_ind_df]
# cvs = [x in train_sample for x in match_ind_df]

cvs_mask = pd.Series(match_ind_df).isin(train_sample)  # Mask for cross-validation sample
vals_mask = ~cvs_mask  # Mask for validation, simply the inverse of cvs_mask

cvs_match_integers = list(set(match_ind_df[cvs_mask]))

cv_X = train_X.loc[cvs_mask.values].copy()
val_X =  train_X.loc[vals_mask.values].copy()
cv_y =  train_y.loc[cvs_mask.values].copy()
val_y = train_y.loc[vals_mask.values].copy()

#transform y to normal distribution
min_val = np.min(cv_y)
log_cv_y = np.log(cv_y - min_val + 1)   

if method == 'linear_reg':
    #9.04 with hyperparams tested on val data. temp win = 0
    #9.02 with hyperparams tested on val data. temp win = 1
    #9.04 with hyperparams tested on val data. temp win = 2
    
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    
    #do not keep historical data
    threshold = 2

    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    
    objective_X = cv_X[columns_to_keep]
    val_X = val_X[columns_to_keep]  
    
    objective_X = objective_X.drop('season', axis=1)
    val_X = val_X.drop('season', axis=1)
    
    for c in objective_X.columns:
        if 'own_difficulty' in c or 'other_difficulty' in c:
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        elif objective_X[c].dtype == 'Int64':
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        
   
    df_cv_one_hot = pd.get_dummies(objective_X, columns=['element_type', 'names'])
    df_val_one_hot = pd.get_dummies(val_X, columns=['element_type', 'names'])
    
    for c in objective_X.columns:
        if 'string_team' in c or 'string_opp_team' in c:
            df_cv_one_hot = pd.get_dummies(df_cv_one_hot, columns=[c])
            df_val_one_hot = pd.get_dummies(df_val_one_hot, columns=[c])
            
    cv_filled_mean = df_cv_one_hot.fillna(df_cv_one_hot.mean(numeric_only=True))
    val_filled_mean = df_val_one_hot.fillna(df_val_one_hot.mean(numeric_only=True))
    
    scaler = StandardScaler()
    scaled_cv_X = scaler.fit_transform(cv_filled_mean)
    
    scaled_cv_X = pd.DataFrame(scaled_cv_X, columns=cv_filled_mean.columns)
    scaled_val_X =  pd.DataFrame(scaler.transform(val_filled_mean), columns=cv_filled_mean.columns)

    regularization = ['lasso', 'ridge']

    space={'alpha': hp.loguniform('alpha', -3, np.log(1000)),
           'reg': hp.choice('reg', regularization),
        }

    trials = Trials()

    best_hyperparams = fmin(fn = objective_linear_reg,
                    space = space,
                    algo = tpe.suggest,
                    early_stop_fn=no_progress_loss(500),
                    trials = trials)

    # model.fit(cv_filled_mean, cv_y)

    # selected = val_X['running_minutes'] > 60

    # val_pred = model.predict(val_filled_mean[selected])
    # val_error = mean_squared_error(val_y[selected], val_pred)

    # plt.scatter(val_y[selected], np.abs((val_y[selected]-val_pred)))

elif method == 'svr':
    
    #takes too long to fit...

    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    
    #do not keep historical data
    threshold = 0

    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    
    objective_X = cv_X[columns_to_keep]
    val_X = val_X[columns_to_keep]  
    
    objective_X = objective_X.drop('season', axis=1)
    val_X = val_X.drop('season', axis=1)
    
    for c in objective_X.columns:
        if 'own_difficulty' in c or 'other_difficulty' in c:
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        elif objective_X[c].dtype == 'Int64':
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        
   
    df_cv_one_hot = pd.get_dummies(objective_X, columns=['element_type', 'names'])
    df_val_one_hot = pd.get_dummies(val_X, columns=['element_type', 'names'])
    
    for c in objective_X.columns:
        if 'string_team' in c or 'string_opp_team' in c:
            df_cv_one_hot = pd.get_dummies(df_cv_one_hot, columns=[c])
            df_val_one_hot = pd.get_dummies(df_val_one_hot, columns=[c])
            
    cv_filled_mean = df_cv_one_hot.fillna(df_cv_one_hot.mean(numeric_only=True))
    val_filled_mean = df_val_one_hot.fillna(df_val_one_hot.mean(numeric_only=True))
    
    scaler = StandardScaler()
    scaled_cv_X = scaler.fit_transform(cv_filled_mean)
    
    scaled_cv_X = pd.DataFrame(scaled_cv_X, columns=cv_filled_mean.columns)
    scaled_val_X =  pd.DataFrame(scaler.transform(val_filled_mean), columns=cv_filled_mean.columns)
    
    #transform y to normal distribution
    min_val = np.min(cv_y)
    log_cv_y = np.log(cv_y - min_val + 1)   
    
    #get an validation set for fitting
    #cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(df_one_hot, train_y, sample_weights, stratify, test_size=0.25, stratify=stratify, random_state=42)

    space = {
        'pars': hp.choice('kernel_shape', [
            {'kernel': 'linear', 'C': hp.loguniform('C_linear', -3, 3), 'epsilon': hp.loguniform('epsilon_linear', -2, 2)},
            {'kernel': 'poly', 'degree': hp.quniform('degree', 2, 5, 1), 'gamma': hp.loguniform('gamma_poly', -4, 2), 'C': hp.loguniform('C_poly', -3, 3), 'epsilon': hp.loguniform('epsilon_poly', -2, 2)},
            {'kernel': 'rbf', 'gamma': hp.loguniform('gamma_rbf', -4, 2), 'C': hp.loguniform('C_rbf', -3, 3), 'epsilon': hp.loguniform('epsilon_rbf', -2, 2)},
            {'kernel': 'sigmoid', 'gamma': hp.loguniform('gamma_sigmoid', -4, 2), 'C': hp.loguniform('C_sigmoid', -3, 3), 'epsilon': hp.loguniform('epsilon_sigmoid', -2, 2)}
        ])
    }


    trials = Trials()

    best_hyperparams = fmin(fn = objective_svr,
                    space = space,
                    algo = tpe.suggest,
                    early_stop_fn=no_progress_loss(1000),
                    max_evals = 10000,
                    trials = trials)


    model.fit(cv_filled_mean, cv_y)

    selected = val_X['running_minutes'] > 60

    val_pred = model.predict(val_filled_mean[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)

    plt.scatter(val_y[selected], np.abs((val_y[selected]-val_pred)))


elif method == 'linear_svr':
    
    #window 0: 9.07
    #windiw 1: 9.06

    from sklearn.svm import LinearSVR
    from sklearn.preprocessing import StandardScaler
    
    
    #do not keep historical data
    threshold = 1

    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    
    objective_X = cv_X[columns_to_keep]
    val_X = val_X[columns_to_keep]  
    
    objective_X = objective_X.drop('season', axis=1)
    val_X = val_X.drop('season', axis=1)
    
    for c in objective_X.columns:
        if 'own_difficulty' in c or 'other_difficulty' in c:
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        elif objective_X[c].dtype == 'Int64':
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        
   
    df_cv_one_hot = pd.get_dummies(objective_X, columns=['element_type', 'names'])
    df_val_one_hot = pd.get_dummies(val_X, columns=['element_type', 'names'])
    
    for c in objective_X.columns:
        if 'string_team' in c or 'string_opp_team' in c:
            df_cv_one_hot = pd.get_dummies(df_cv_one_hot, columns=[c])
            df_val_one_hot = pd.get_dummies(df_val_one_hot, columns=[c])
            
    cv_filled_mean = df_cv_one_hot.fillna(df_cv_one_hot.mean(numeric_only=True))
    val_filled_mean = df_val_one_hot.fillna(df_val_one_hot.mean(numeric_only=True))
    
    scaler = StandardScaler()
    scaled_cv_X = scaler.fit_transform(cv_filled_mean)
    
    scaled_cv_X = pd.DataFrame(scaled_cv_X, columns=cv_filled_mean.columns)
    scaled_val_X =  pd.DataFrame(scaler.transform(val_filled_mean), columns=cv_filled_mean.columns)
    


    space = {'C': hp.loguniform('C_linear', -3, 3),
             'epsilon': hp.loguniform('epsilon_linear', -2, 2),
             'loss': hp.choice('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            }


    trials = Trials()

    best_hyperparams = fmin(fn = objective_linear_svr,
                    space = space,
                    algo = tpe.suggest,
                    early_stop_fn=no_progress_loss(500),
                    max_evals = 10000,
                    trials = trials)


    model.fit(cv_filled_mean, cv_y)

    selected = val_X['running_minutes'] > 60

    val_pred = model.predict(val_filled_mean[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)

    plt.scatter(val_y[selected], np.abs((val_y[selected]-val_pred)))


elif method == 'cnn':

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from sklearn.preprocessing import StandardScaler
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ReduceLROnPlateau
    from keras.regularizers import l1, l2, l1_l2
    
    #do not keep historical data
    threshold = 0

    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    
    objective_X = cv_X[columns_to_keep]
    val_X = val_X[columns_to_keep]  
    
    objective_X = objective_X.drop('season', axis=1)
    val_X = val_X.drop('season', axis=1)
    
    for c in objective_X.columns:
        if 'own_difficulty' in c or 'other_difficulty' in c:
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        elif objective_X[c].dtype == 'Int64':
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        
   
    df_cv_one_hot = pd.get_dummies(objective_X, columns=['element_type', 'names'])
    df_val_one_hot = pd.get_dummies(val_X, columns=['element_type', 'names'])
    
    for c in objective_X.columns:
        if 'string_team' in c or 'string_opp_team' in c:
            df_cv_one_hot = pd.get_dummies(df_cv_one_hot, columns=[c])
            df_val_one_hot = pd.get_dummies(df_val_one_hot, columns=[c])
            
            
    fit_X, eval_X, fit_y, eval_y = train_test_split(df_cv_one_hot, cv_y, test_size=0.25, random_state=42)
    
    cv_filled_mean = df_cv_one_hot.fillna(df_cv_one_hot.median(numeric_only=True))
    val_filled_mean = df_val_one_hot.fillna(df_cv_one_hot.median(numeric_only=True))
    
    eval_X_filled_mean = eval_X.fillna(fit_X.nanmedian(numeric_only=True))
    fit_X_filled_mean = fit_X.fillna(fit_X.median(numeric_only=True))   

    scaler = StandardScaler()
    scaled_fit_X = scaler.fit_transform(fit_X_filled_mean)
    scaled_eval_X = scaler.transform(eval_X_filled_mean)

    scaler = StandardScaler()
    scaled_cv_X = scaler.fit_transform(cv_filled_mean)
    scaled_val_X = scaler.transform(val_filled_mean)

    alpha_l1 = 0.001
    alpha_l2 = 0.001

    # Define the model
    model = Sequential()

    model.add(Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=alpha_l1, l2=alpha_l2)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))  # Change to 'sigmoid' if binary classification

    # Define the ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Metric to monitor
    factor=0.1,          # Factor by which to reduce the learning rate
    patience=1,         # Number of epochs with no improvement after which learning rate is reduced
    min_lr=0.000001,      # Lower bound on the learning rate
    verbose=1            # Verbosity mode; 1 for printing messages, 0 for silent
    )

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')  # Choose 'binary_crossentropy' for binary classification

    # Train the model
    model.fit(scaled_fit_X.astype(float), fit_y, epochs=100, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=2), reduce_lr], validation_data=(scaled_eval_X.astype(float), eval_y))

    # Evaluate the model
    #selected = val_X['running_minutes'] > 60

    val_pred = model.predict(scaled_val_X)
    val_error = mean_squared_error(val_y, val_pred)
    print(val_error)

elif method == 'mixedLM':
    #2025: with random effect: 8.80, window 0
    #2025: with random effect: 8.79, window 1
    #2025: with random effect: 8.80, window 2
    
    cv_X = train_X.iloc[cvs].copy()
    val_X =  train_X.loc[vals].copy()
    cv_y =  train_y.loc[cvs].copy()
    val_y = train_y.loc[vals].copy()


    #do not keep historical data
    threshold = 2

    # Filter the columns based on the defined function
    columns_to_keep = [col for col in cv_X.columns if should_keep_column(col, threshold)]
    
    objective_X = cv_X[columns_to_keep]
    val_X = val_X[columns_to_keep]  
    
    objective_X = objective_X.drop('season', axis=1)
    val_X = val_X.drop('season', axis=1)
    
    #objective_X = objective_X.dropna(how='any')
    
    for c in objective_X.columns:
        if 'own_difficulty' in c or 'other_difficulty' in c:
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)
        elif objective_X[c].dtype == 'Int64':
            objective_X[c] = objective_X[c].astype(float)
            val_X[c] = val_X[c].astype(float)   
        elif objective_X[c].dtype == 'category':
            objective_X[c] = objective_X[c].cat.add_categories('unknown')  # Make sure 'NaN' is a category
            objective_X[c].fillna('unknown', inplace=True)
            
            val_X[c] = val_X[c].cat.add_categories('unknown')  # Make sure 'NaN' is a category
            val_X[c].fillna('unknown', inplace=True)
            
            objective_X[c] = objective_X[c].astype(str)
            val_X[c] = val_X[c].astype(str)  
            

    cv_filled_mean = objective_X.fillna(objective_X.median(numeric_only=True))
    val_filled_mean = val_X.fillna(objective_X.median(numeric_only=True))
    
    
    #rename columns
    new_columns = []
    # Create a mapping of digits to capital letters
    digit_to_letter = {
        '0': 'A',
        '1': 'B',
        '2': 'C',
        '3': 'D',
        '4': 'E',
        '5': 'F',
        '6': 'G',
        '7': 'H',
        '8': 'I',
        '9': 'J'
    }


    for col in cv_filled_mean:
        if col[0].isdigit():  # Check if the first character is a digit
            new_col = digit_to_letter[col[0]] + col[1:]  # Move the first character to the end
            new_columns.append(new_col)
        else:
            new_columns.append(col)  # Keep the original name

    # Renaming the columns
    cv_filled_mean.columns = new_columns
    val_filled_mean.columns = new_columns
    
    
    #make fit_tring
    fit_string = "total_points ~ "
    for ind, c in enumerate(cv_filled_mean.keys()):
        
        if ind > 100:
            continue
        
        if c == 'names' or c =='total_points':
            continue
        
        if cv_filled_mean[c].dtype == 'category':
            c_string = "C(" + c + ")"
        else:
            c_string = c
        
        if ind > 0:
            fit_string = fit_string + " + " + c_string
        else:
            fit_string = fit_string + c_string
        
    #transform y to normal distribution
    min_val = np.min(cv_y)
    log_cv_y = np.log(cv_y - min_val + 1)  
    
    cv_filled_mean['total_points'] = log_cv_y
    cv_filled_mean['names'] = cv_filled_mean['names'].astype(str)
    
    #necessary to avoid singular matrix
    # Remove rows that contain 'unknown' in any column
    #cv_filled_mean = cv_filled_mean[~cv_filled_mean.isin(['unknown']).any(axis=1)]

    model = sm.MixedLM.from_formula(fit_string, groups='names', data=cv_filled_mean)
    result = model.fit()

    prediction = result.predict(val_filled_mean)
    val_normal = np.exp(prediction) + min_val - 1    
    val_error = mean_squared_error(val_y,  val_normal)
    print('without random effect', val_error)
    

    rand_e = result.random_effects
    name_list = list(rand_e.keys())

    for row in val_filled_mean.iterrows():
        if row[1]["names"] in name_list:
            random_effect = rand_e[row[1]["names"]].iloc[0]
        else:
            random_effect = 0

        # if row[1]["element_type"] in name_list:
        #     random_effect = re[row[1]["element_type"]].iloc[0]
        # else:
        #     random_effect = 0

        prediction[row[0]] = prediction[row[0]] + random_effect

    val_normal = np.exp(prediction) + min_val - 1    
    val_error = mean_squared_error(val_y,  val_normal)
    print('with random effect', val_error)


elif method == 'xgboost':

    #make sure all categories in val_x is present in cv_x
    for column in val_X.columns:
        if isinstance(val_X[column].dtype, pd.CategoricalDtype):
            # Get the values in the current column of val_X
            val_values = val_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(cv_X[column])
            
            # Set values that are not present in cv_X[column] to NaN
            val_X.loc[~mask, column] = np.nan
    
    grow_policy = ['depthwise', 'lossguide']
    
    
    # #make sure that there will be data left for evaluation in the final model
    # cv_season =  cv_X.iloc[-1].season
    # selected_cv =  cv_X.season == cv_season
    # cv_fraction = sum(selected_cv) / cv_X.shape[0]   
    
    # current_season =  train_X.iloc[-1].season
    # selected_test =  train_X.season == current_season
    # current_fraction = sum(selected_test) / train_X.shape[0]   
    
    #max_eval_fraction = np.min([cv_fraction, current_fraction])
    
    
    min_eval_fraction = 1/(len(unique_integers) * 0.80)#len(np.unique(cv_stratify))/cv_X.shape[0]

    space={'max_depth': hp.quniform("max_depth", 1, 1500, 1),
            'min_split_loss': hp.uniform('min_split_loss', 0, 175), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 275),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 400),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 700),
            'learning_rate': hp.uniform('learning_rate', 0, 0.05),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 10, 2500, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.2),
            'n_estimators': hp.quniform('n_estimators', 2, 20000, 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 40),
            'grow_policy': hp.choice('grow_policy', grow_policy), #111
            'max_leaves': hp.quniform('max_leaves', 0, 1400, 1),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(125), 1),
            'temporal_window': hp.quniform('temporal_window', 0, temporal_window+1, 1),
        }
    
    #include feature search in the hyperparams
    check_features = ['transfers_in', 'transfers_out', 'minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
            'total_points', 'expected_goals', 'expected_assists', 'points_per_played_game', 'was_home',
            'expected_goal_assists', 'expected_goals_conceded', 'own_team_points', 'own_element_points', 'SoT', 'defcon', 'name', 'points_per_game']#, 'difficulty']

    
    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [True, False]), #111

    mean_cv = np.mean(cv_y)
    train_error = np.mean(np.abs((cv_y - mean_cv)**2))
    validation_error = np.mean(np.abs((val_y - mean_cv)**2))
    
    print('Train random error: ', train_error)
    print('Validation random error: ', validation_error)   
    
    hyperparam_path = main_directory + '\models\hyperparams.pkl'
    with open(hyperparam_path, 'rb') as f:
        old_trials = pickle.load(f)

    hyperparams = old_trials.best_trial['misc']['vals']
    #reformat the lists
    old_hyperparams = {}
    for field, val in hyperparams.items():
        old_hyperparams[field] = val[0]
        
    # old_trials = generate_trials_to_calculate([old_hyperparams])

    # old_hyperparams = fmin(fn = objective_xgboost,
    #                 space = space,
    #                 algo = tpe.suggest,
    #                 max_evals = 1,
    #                 trials = old_trials)    
        
    # old_loss = old_trials.best_trial["result"]["loss"]
    
    old_hyperparams["grow_policy"] = grow_policy[old_hyperparams["grow_policy"]]
    
    #loss = objective_xgboost(old_hyperparams)
    #old_loss = loss['loss']
    old_loss = 1
    
    print('Old loss: ', old_loss)
        
    #optimize and iteratively get hyperparamaters
    batch_size = 100
    if optimize:
        max_evals = 500000
    
    if continue_optimize:
        hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
        with open(hyperparam_path, 'rb') as f:
            trials = pickle.load(f)
    else:
        trials = Trials()

    if optimize:

        for i in range(len(trials.trials)+batch_size, max_evals + 1, batch_size):

            # Save the trials object every 'batch_size' iterations. Can save with any method you prefer

            #optmimize hyperparameters. use all training data
            best_hyperparams = fmin(fn = objective_xgboost,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = i,
                            trials = trials)

            print(best_hyperparams)
            
            #hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
            hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
            pickle.dump(trials, open(hyperparam_path, "wb"))
            
    else:      

        
        hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
        with open(hyperparam_path, 'rb') as f:
            new_trials = pickle.load(f)
            
        hyperparams = new_trials.best_trial['misc']['vals']
        #reformat the lists
        new_hyperparams = {}
        for field, val in hyperparams.items():
            new_hyperparams[field] = val[0]
            
        new_hyperparams["grow_policy"] = grow_policy[new_hyperparams["grow_policy"]]
            
        # new_trials = generate_trials_to_calculate([new_hyperparams])

        # new_hyperparams = fmin(fn = objective_xgboost,
        #                 space = space,
        #                 algo = tpe.suggest,
        #                 max_evals = 1,
        #                 trials = new_trials)    

        # new_loss =  new_trials.best_trial["result"]["loss"]
        
        loss = objective_xgboost(new_hyperparams)
        new_loss = loss['loss']
        
        print('New loss: ', new_loss)
        
        if new_loss < old_loss:
            print('Overwriting old loss')
            hyperparam_path = main_directory + '\models\hyperparams.pkl'
            pickle.dump(new_trials, open(hyperparam_path, "wb"))
            trials = new_trials
            
            print(new_hyperparams)
        else:
            trials = old_trials
            print(old_hyperparams)
        
        losses = []
        for i in range(len(trials.trials)):
    
            if trials.trials[i]['result'] == {'status': 'new'}:
                losses.append(9999)
                print('Miss result')
            else:
                losses.append(trials.trials[i]['result']['loss'])
    
        sorted_losses = np.argsort(losses)
    
        
        best_best_ind = 0
    
        #train with all data
        best_cv_trial =  sorted_losses[best_best_ind]
        print('Original loss:', losses[best_cv_trial])
    
        hyperparams = trials.trials[best_cv_trial]['misc']['vals']
        #print(hyperparams)
    
        space = {}
        for field, val in hyperparams.items():
            space[field] = val[0]
    
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
            'grow_policy': grow_policy[space['grow_policy']],
            'max_leaves': int(space['max_leaves']),
            'tree_method': 'hist',
            'max_bin':  int(space['max_bin']),
            'disable_default_eval_metric': 1
            }
    
        #remove weaks that we don't need.
        # Define the threshold
        threshold = int(space['temporal_window'])
    
        # Filter the columns based on the defined function
        columns_to_keep = [col for col in train_X.columns if should_keep_column(col, threshold)]
        objective_X = train_X[columns_to_keep]
        
        #remove features
        for feat in check_features:
            if feat in space.keys():
                if not space[feat][0]:        
                    columns_to_keep = [col for col in objective_X.columns if not feat == re.sub(r'\d+', '', col)]
                    objective_X = objective_X[columns_to_keep]  
    
        #fit_X, eval_X, fit_y, eval_y, fit_sample_weights, eval_sample_weights = train_test_split(objective_X, train_y, sample_weights, test_size=space['eval_fraction'], stratify=stratify, random_state=42)
        
        match_ind = pd.factorize(
            objective_X[['string_team', 'was_home', 'string_opp_team', 'season']]
            .apply(lambda row: '-'.join(row.astype(str)), axis=1)
        )[0]
            
        #get 20% of those matches
        # Step 1: Get unique integers using a set
        unique_integers = list(set(match_ind))
        
        # Step 2: Calculate 20% of the unique integers
        num_to_select = int(len(unique_integers) * space['eval_fraction'])
        
        
        
        # Step 3: Randomly select 20% of the unique integers
        eval_sample = random.sample(unique_integers, num_to_select)
        
        match_ind_df = pd.Series(match_ind) 
        
        evals_mask = match_ind_df.isin(eval_sample)  # Mask for cross-validation sample
        fits_mask = ~evals_mask  # Mask for validation, simply the inverse of cvs_mask
        
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
        num_boost_round=int(space['n_estimators']),
        early_stopping_rounds= int(space['early_stopping_rounds']),
        dtrain=dfit,
        evals=evals,
        custom_metric=custom_metric,
        obj=quantile_objective,
        verbose_eval=False  # Set to True if you want to see detailed logging
            )
    
        summary = {'model': model, 'train_features': objective_X, 'hyperparameters': space}#, 'all_rows': original_df}
    
        pickle.dump(summary, open(model_path, 'wb'))
    
        xgb.plot_importance(model, importance_type='gain',
                        max_num_features=20, show_values=False)
        plt.show()
        
        data =  model.get_score()

        # Dictionary to hold summed values and counts
        summed_values = {}
        count_values = {}

        for key, value in data.items():
            # Extract the part of the string after the digits
            new_key = ''.join(filter(lambda x: not x.isdigit(), key))  # or use re.sub(r'^\d+', '', key)
            
            # Sum the values and count the occurrences for the same new_key
            if new_key in summed_values:
                summed_values[new_key] += value
                count_values[new_key] += 1
            else:
                summed_values[new_key] = value
                count_values[new_key] = 1

        # Calculate mean for each key
        mean_values = {k: summed_values[k] / count_values[k] for k in summed_values}

        # Sort the mean values by their values
        sorted_mean_values = dict(sorted(mean_values.items(), key=lambda item: item[1]))

        #print(sorted_mean_values)  # Output will be sorted by mean values

        # Plotting the sorted mean values
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_mean_values.keys(), sorted_mean_values.values(), color='skyblue')
        plt.xlabel('Labels')
        plt.ylabel('Mean Values')
        plt.title('Mean Values of Points Sorted')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of labels

        # Show the plot
        plt.show()
        
        train_data = xgb.DMatrix(data=objective_X, label=train_y, enable_categorical=True)
        pred = model.predict(train_data)
        
        train_error = np.mean(np.abs((train_y - pred)**2))
        
        print('Train error:', train_error)