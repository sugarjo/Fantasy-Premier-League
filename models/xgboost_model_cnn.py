import os
import pandas as pd
import pickle
import statsmodels.api as sm
import numpy as np
import difflib
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, train_test_split
from datetime import timedelta, datetime
import multiprocessing

from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from matplotlib import pyplot as plt
from hyperopt.pyll import scope
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate

directories = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data'

optimize = True
continue_optimize = False

method = 'linear_reg' #xgboost

season_dfs = []

season_count = 0


#get each season
for folder in os.listdir(directories):
    
    directory = directories + '/' + folder
    fixture_data = directory + "/fixtures.csv"
    gws_data = directory + '/gws'
    team_path = directory + "/teams.csv"

    if os.path.isfile(fixture_data) and  os.path.isdir(gws_data):

        #check that it is not a file
        if folder[-4] != '.':

            print(folder)

            #get id so it can be matched with position
            player_path = directory + '/players_raw.csv'
            df_player = pd.read_csv(player_path)
            
            #rename befor merge
            df_player = df_player.rename(columns={"id": "element"})  
            
            #insert string for team
            df_teams = pd.read_csv(team_path)
            string_names = df_teams['short_name'].values
            df_player["string_team"] = string_names[df_player["team"]-1]
            
            dfs_gw = []
            
            #open each gw and get data for players
            for gw_csv in os.listdir(directory + '/gws'):
                if gw_csv[0] == 'g':
                    
                    gw_path = directory + '/gws' + '/' + gw_csv
                    print(gw_csv)
                    
                    if folder == '2018-19' or folder == '2016-17':
                        dfs_gw.append(pd.read_csv(gw_path, encoding='latin1'))
                    else:
                        dfs_gw.append(pd.read_csv(gw_path))

            df_gw = pd.concat(dfs_gw)
            
            df_gw['kickoff_time'] =  pd.to_datetime(df_gw['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
            df_gw = df_gw.sort_values(by='kickoff_time')
            
            df_gw.reset_index(inplace=True)
            
            df_gw['form'] = np.nan
            df_gw['running_xG'] = np.nan
            df_gw['running_xA'] = np.nan
            df_gw['running_xGI'] = np.nan
            df_gw['running_xGC'] = np.nan
            df_gw['running_xP'] = np.nan
            df_gw['points_per_game'] = np.nan
            df_gw['points_per_played_game'] = np.nan
            df_gw['running_ict'] = np.nan
            df_gw['running_influence'] = np.nan
            df_gw['running_threat'] = np.nan
            df_gw['running_creativity'] = np.nan
            df_gw['running_bps'] = np.nan
            df_gw['transfer_in'] = np.nan
            df_gw['transfer_out'] = np.nan
            df_gw['string_opp_team'] = np.nan
            df_gw['running_minutes'] = np.nan
                
            # Calculate rolling values not including the observaiton
            for player in df_gw['element'].unique():
                   
                selected_ind = df_gw['element'] == player                    
                player_df = df_gw[selected_ind]
                player_df.set_index('kickoff_time', inplace=True)
                
                
                #add column if they don't exist
                if folder == '2018-19' or folder == '2019-20':
                    xG = np.empty((len(player_df)))
                    xG[:] = np.nan
                    
                    xP = np.empty((len(player_df)))
                    xP[:] = np.nan
                    
                    xA = np.empty((len(player_df)))
                    xA[:] = np.nan
                    
                    xGI = np.empty((len(player_df)))
                    xGI[:] = np.nan
                    
                    xGC = np.empty((len(player_df)))
                    xGC[:] = np.nan
                elif folder == '2020-21' or folder == '2021-22':
                    xG = np.empty((len(player_df)))
                    xG[:] = np.nan
                    
                    xP = player_df['xP'].shift(1).rolling('30D').mean().values
                    
                    xA = np.empty((len(player_df)))
                    xA[:] = np.nan
                    
                    xGI = np.empty((len(player_df)))
                    xGI[:] = np.nan
                    
                    xGC = np.empty((len(player_df)))
                    xGC[:] = np.nan
                    
                else:  
                    xG = player_df['expected_goals'].shift(1).rolling('30D').mean().values
                    xP = player_df['xP'].shift(1).rolling('30D').mean().values
                    xA = player_df['expected_assists'].shift(1).rolling('30D').mean().values
                    xGI = player_df['expected_goal_involvements'].shift(1).rolling('30D').mean().values
                    xGC = player_df['expected_goals_conceded'].shift(1).rolling('30D').mean().values


                form = player_df['total_points'].shift(1).rolling('30D').mean()
                points_per_game =  player_df['total_points'].cumsum().shift(1) / (player_df['round']).shift(1)
                ict = player_df['ict_index'].shift(1).rolling('30D').mean()
                influence = player_df['influence'].shift(1).rolling('30D').mean()
                threat = player_df['threat'].shift(1).rolling('30D').mean()
                creativity = player_df['creativity'].shift(1).rolling('30D').mean()
                bps = player_df['bps'].shift(1).rolling('30D').mean()
                transfer_in = player_df['transfers_in'].values
                transfer_out = player_df['transfers_out'].values
                minutes = player_df['minutes'].shift(1).rolling('30D').mean()
                
                #points per played game
                result = np.zeros(len(player_df['total_points'])+1)  # initialize result array
                last_games = 0  # initialize last_vplayer_df['total_points']alue to 0
                last_point = 0
                
                for i in range(len(player_df['total_points'])):
                    
                    if player_df['minutes'][i] >= 60:
                        last_point += player_df['total_points'][i]
                        last_games += 1
                    
                    if last_games > 0:
                        result[i+1] = last_point/last_games
                
                df_gw.loc[selected_ind, 'running_ict'] = ict.values
                df_gw.loc[selected_ind, 'running_influence'] = influence.values
                df_gw.loc[selected_ind, 'running_threat'] = threat.values
                df_gw.loc[selected_ind, 'running_creativity'] = creativity.values
                df_gw.loc[selected_ind, 'running_bps'] = bps.values
                df_gw.loc[selected_ind, 'form'] = form.values
                df_gw.loc[selected_ind, 'running_xG'] = xG
                df_gw.loc[selected_ind, 'running_xA'] = xA
                df_gw.loc[selected_ind, 'running_xGI'] = xGI
                df_gw.loc[selected_ind, 'running_xGC'] = xGC   
                df_gw.loc[selected_ind, 'running_xP'] = xP
                df_gw.loc[selected_ind, 'points_per_game'] = points_per_game.values
                df_gw.loc[selected_ind, 'points_per_played_game'] = result[:-1]
                df_gw.loc[selected_ind, 'transfer_in'] = transfer_in
                df_gw.loc[selected_ind, 'transfer_out'] = transfer_out
                df_gw.loc[selected_ind, 'running_minutes'] = minutes.values
                
                opp_team = []
                for team in player_df['opponent_team'].astype(int).values-1:
                    opp_team.append(string_names[team])
                       
                df_gw.loc[selected_ind, 'string_opp_team'] = opp_team
                
            season_df = df_gw[['running_minutes', 'string_opp_team', 'transfer_in', 'transfer_out', 'running_ict', 'running_influence', 'running_threat', 'running_creativity', 'running_bps', 'element', 'fixture', 'minutes', 'total_points', 'round', 'was_home', 'kickoff_time', 'running_xP', 'running_xG', 'running_xA', 'running_xGI', 'running_xGC', 'form', 'points_per_game', 'points_per_played_game']]
            
            #get fixture difficulty difference for each datapoint
            #open fixtures data
            fixture_data_path = directory + "/fixtures.csv"
            
            fixture_df = pd.read_csv(fixture_data_path)
            
            #rename befor merge
            fixture_df = fixture_df.rename(columns={"id": "fixture"})
            season_df = pd.merge(season_df, fixture_df[["team_a_difficulty", "team_h_difficulty", "fixture"]], on='fixture')
            
            season_df = pd.merge(season_df, df_player[["element_type", "first_name", "second_name", "web_name", "string_team", "element"]], on="element")
            
            season_df["season"] = folder
            
            season_dfs.append(season_df)
                     
season_df = pd.concat(season_dfs)
season_df['transfer_in'] = season_df['transfer_in'].astype(float)
season_df['transfer_out'] = season_df['transfer_out'].astype(float)
season_df['points_per_game'] = season_df['points_per_game'].astype(float)

season_df = season_df.reset_index()

min_val = np.min(season_df['total_points'])-1

#season_df['total_points'] = np.log10(season_df['total_points'].astype(float)-min_val)

season_df['names'] = season_df['first_name'] + season_df['second_name']

#MATCH NAMES
name_list = season_df['names'].unique()

#not that dangerous to merge previous players, but avoid to merge into current player
#loop through the most recent players first
for name_ind, name in reversed(list(enumerate(name_list))):
    matched_names = difflib.get_close_matches(name, name_list[0:name_ind+1], cutoff=0.84)
    
    #if any matched
    if len(matched_names) > 1:
        
        original_player_ind = season_df['names'] == matched_names[0]
        original_position = season_df['element_type'][original_player_ind].unique()
        
        #if same position
        for position in original_position:
            for change_name_ind in range(1,len(matched_names)):
                #find the other player
                change_player_ind = np.logical_and(season_df['names'] == matched_names[change_name_ind], season_df['element_type'] == position)
                
                if any(change_player_ind): 
                    #make the other player into current player
                    season_df.loc[change_player_ind, 'names'] = matched_names[0]
                    print(matched_names[change_name_ind] + ' changed with ' + matched_names[0])

original_season_df = season_df

#remove players that don't play
selected = season_df["minutes"] > 0
season_df = season_df.loc[selected]

#different events have different impacts on different player types
selected = np.logical_or(season_df['element_type'] == 1, season_df['element_type'] == 2)
season_df.loc[selected, 'running_xG'] = season_df.loc[selected, 'running_xG']*6
season_df.loc[selected, 'running_xGC'] = season_df.loc[selected, 'running_xGC']*4

selected = season_df['element_type'] == 3
season_df.loc[selected, 'running_xG'] = season_df.loc[selected, 'running_xG']*5

selected = season_df['element_type'] == 4
season_df.loc[selected, 'running_xG'] = season_df.loc[selected, 'running_xG']*4
season_df.loc[selected, 'running_xGC'] = 0

season_df.loc[:, 'running_xA'] = season_df.loc[:, 'running_xA']*3
            
home_diff = season_df["team_h_difficulty"].copy()
away_diff = season_df["team_a_difficulty"].copy()

difficulty_diff = (home_diff - away_diff)

season_df.loc[:, 'difficulty'] = difficulty_diff

home = season_df['was_home'] == 1
season_df.loc[home, 'difficulty'] = -season_df.loc[home, 'difficulty']

#for all away matches
season_df['own_difficulty'] = season_df["team_a_difficulty"].copy()
season_df['other_difficulty'] = season_df["team_h_difficulty"].copy()
#correct home matches
season_df.loc[home, 'own_difficulty'] = season_df.loc[home, "team_h_difficulty"]
season_df.loc[home, 'other_difficulty'] = season_df.loc[home, "team_a_difficulty"]

season_df['was_home'] = season_df['was_home'].astype(int)

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
        'early_stopping_rounds': int(space['early_stopping_rounds']),
        'n_estimators': int(space['n_estimators']),
        'max_delta_step': space['max_delta_step'],
        'grow_policy': space['grow_policy'],
        'max_leaves': int(space['max_leaves']),
        }        
        
    #get som data for evaluation       
    fit_X, eval_X, fit_y, eval_y, fit_sample_weights, _ =  train_test_split(cv_X, cv_y, cv_sample_weights, test_size=space['eval_fraction'], stratify=cv_stratify, random_state=42)

    model = xgb.XGBRegressor(**pars, tree_method="hist", enable_categorical=True, max_bin=30)
    model.fit(fit_X, fit_y, verbose=False,
        eval_set=[(fit_X, fit_y), (eval_X, eval_y)], sample_weight=fit_sample_weights) 
    
    #selected = val_sample_weights >= 1
    selected = val_X['running_minutes'] > 60
    
    val_pred = model.predict(val_X[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)
    
        
    return {'loss': val_error, 'status': STATUS_OK }


def objective_linear_reg(space):
    
    if space['reg'] == 'lasso':
        model = Lasso(alpha=space['alpha'])
    else:
        model = Ridge(alpha=space['alpha'])
        
    model.fit(cv_filled_mean, cv_y)
    
    #selected = val_sample_weights >= 1
    selected = val_X['running_minutes'] > 60
    
    val_pred = model.predict(val_filled_mean[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)
    
    return {'loss': val_error, 'status': STATUS_OK }

def objective_svr(space):
    print(space)
    
    model = SVR(**space['pars'])
        
    model.fit(cv_filled_mean, cv_y)
    
    #selected = val_sample_weights >= 1
    selected = val_X['running_minutes'] > 60
    
    val_pred = model.predict(val_filled_mean[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)
    
    return {'loss': val_error, 'status': STATUS_OK }

def objective_linear_svr(space):
    
    model = LinearSVR(**space, fit_intercept=False, dual="auto", max_iter=100000)
        
    model.fit(cv_filled_mean, cv_y)
    
    #selected = val_sample_weights >= 1
    selected = val_X['running_minutes'] > 60
    
    val_pred = model.predict(val_filled_mean[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)
    
    return {'loss': val_error, 'status': STATUS_OK }
      
      

#remove players with few matches
unique_names, unique_counts = np.unique(season_df.names, return_counts=True)

for unique_ind, unique_n in enumerate(unique_counts):
    if unique_n < 3:
        name = unique_names[unique_ind]
        selected = (season_df.names == name)
        season_df.names.loc[selected] = 'unknown'        

season_df['element_type'] = season_df['element_type'].astype('category')
season_df['names'] = season_df['names'].astype('category')
season_df['difficulty'] = season_df['difficulty'].astype('category')
season_df['own_difficulty'] = season_df['own_difficulty'].astype('category')
season_df['other_difficulty'] = season_df['other_difficulty'].astype('category')
season_df['string_team'] = season_df['string_team'].astype('category')
season_df['string_opp_team'] = season_df['string_opp_team'].astype('category')

keep_features = ['running_minutes', 'transfer_in', 'transfer_out', 'running_ict', 'running_influence', 'running_threat', 'running_creativity', 'running_bps', 'string_opp_team', 'string_team', 'names', 'element_type', 'was_home', 'running_xP', 'running_xG', 'running_xA', 'running_xGI', 'running_xGC', 'form',
                 'points_per_game', 'points_per_played_game', 'other_difficulty', 'own_difficulty', 'total_points']

#weight samples for time
last_year = season_df['kickoff_time'].iloc[-1] - season_df['kickoff_time']
selected = last_year < timedelta(365)
sample_weights = np.ones(selected.shape)
sample_weights[selected] = 4

# time_since = season_df['kickoff_time'] - season_df['kickoff_time'].iloc[0] + timedelta(365)
# sample_weights = (time_since/timedelta(365)).values**4
#give small weight for the uninteresting samples
# selected = season_df['minutes'] < 60
# sample_weights[selected] = 0.25

season_df = season_df[keep_features]       

cpu_count = multiprocessing.cpu_count()

#train final model
train_X = season_df.drop(['total_points'], axis=1)
train_y = season_df['total_points'].astype(int)

#optimize and iteratively get hyperparamaters
batch_size = 100
max_evals = 500000

with open(r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\hyperparams.pkl', 'rb') as f:
    old_trials = pickle.load(f)
    
old_hyperparams = old_trials.best_trial['misc']['vals']
#reformat the lists
test_hyperparams = {}
for field, val in old_hyperparams.items():
    test_hyperparams[field] = val[0]
    
if not continue_optimize:
    trials = generate_trials_to_calculate([test_hyperparams])
else:
    trials = old_trials

# Define the number of quantiles/bins
num_bins = 100
    
# Calculate the quantile boundaries of the outcome variable
centiles = pd.qcut(train_y, q=num_bins, duplicates="drop", retbins=True)[1]
centiles[0] = -np.inf
# Discretize the outcome variable using the quantile boundaries
stratify = pd.cut(train_y, bins=centiles, labels=False)


if method == 'linear_reg':
    #8.94
    
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    df_one_hot = pd.get_dummies(train_X, columns=['string_team', 'string_opp_team', 'element_type', 'names'])
    
    #df_one_hot = df_one_hot.drop('names', axis=1)
    #get an validation set for fitting
    cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(df_one_hot, train_y, sample_weights, stratify, test_size=0.25, stratify=stratify, random_state=42)

    scaled_cv_X = scaler.fit_transform(cv_X)
    scaled_val_X = scaler.transform(val_X)  

    cv_filled_mean = cv_X.fillna(cv_X.mean(numeric_only=True))
    val_filled_mean = val_X.fillna(cv_X.mean(numeric_only=True))
    
    regularization = ['lasso', 'ridge']

    space={'alpha': hp.loguniform('alpha', -3, np.log(1000)),
           'reg': hp.choice('reg', regularization),
        }
    
    trials = Trials()
    
    best_hyperparams = fmin(fn = objective_linear_reg,
                    space = space,
                    algo = tpe.suggest,
                    early_stop_fn=no_progress_loss(1000), 
                    trials = trials)
    
    
    model.fit(cv_filled_mean, cv_y)
    
    selected = val_X['running_minutes'] > 60
    
    val_pred = model.predict(val_filled_mean[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)
    
    plt.scatter(val_y[selected], np.abs((val_y[selected]-val_pred)))
    
elif method == 'svr':
    
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    df_one_hot = pd.get_dummies(train_X, columns=['string_team', 'string_opp_team', 'element_type', 'names', 'other_difficulty', 'own_difficulty'])
    
    #df_one_hot = df_one_hot.drop('names', axis=1)
    #get an validation set for fitting
    cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(df_one_hot, train_y, sample_weights, stratify, test_size=0.25, stratify=stratify, random_state=42)

    scaled_cv_X = scaler.fit_transform(cv_X)
    scaled_val_X = scaler.transform(val_X)  

    cv_filled_mean = cv_X.fillna(cv_X.mean(numeric_only=True))
    val_filled_mean = val_X.fillna(cv_X.mean(numeric_only=True))          
        
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
    
    from sklearn.svm import LinearSVR
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    df_one_hot = pd.get_dummies(train_X, columns=['string_team', 'string_opp_team', 'element_type', 'names'])
    
    #df_one_hot = df_one_hot.drop('names', axis=1)
    #get an validation set for fitting
    cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(df_one_hot, train_y, sample_weights, stratify, test_size=0.25, stratify=stratify, random_state=42)

    scaled_cv_X = scaler.fit_transform(cv_X)
    scaled_val_X = scaler.transform(val_X)  

    cv_filled_mean = cv_X.fillna(cv_X.mean(numeric_only=True))
    val_filled_mean = val_X.fillna(cv_X.mean(numeric_only=True))          
        
    space = {'C': hp.loguniform('C_linear', -3, 3), 
             'epsilon': hp.loguniform('epsilon_linear', -2, 2),
             'loss': hp.choice('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            }

    
    trials = Trials()
    
    best_hyperparams = fmin(fn = objective_linear_svr,
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

elif method == 'cnn':
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout, EarlyStopping
    from sklearn.preprocessing import StandardScaler
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ReduceLROnPlateau
    from keras.regularizers import l1, l2, l1_l2
    
    
    # Identify indices of rows to be removed
    mask = train_X.isna().any(axis=1)
    nan_indices = train_X[mask].index
    
    # Remove rows with NaN from the original DataFrame
    df_cleaned = train_X.dropna()
    
    # Remove corresponding rows from the output DataFrame
    outputs_cleaned = train_y.drop(nan_indices)
    stratify_cleaned = stratify[~mask]
    sample_weights_cleaned = sample_weights[~mask]
    
    df_one_hot = pd.get_dummies(df_cleaned, columns=['string_team', 'string_opp_team', 'element_type', 'names', 'other_difficulty', 'own_difficulty'])
    
    #df_one_hot = df_one_hot.drop('names', axis=1)
    #get an validation set for fitting
    cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(df_one_hot, outputs_cleaned, sample_weights_cleaned, stratify_cleaned, test_size=0.25, stratify=stratify_cleaned, random_state=42)

    fit_X, eval_X, fit_y, eval_y, fit_sample_weights, _ = train_test_split(scaled_cv_X, cv_y, cv_sample_weights, test_size=0.25, stratify=cv_stratify, random_state=42)   

    scaler = StandardScaler()    
    scaled_fit_X = scaler.fit_transform(fit_X)
    scaled_eval_X = scaler.transform(eval_X) 
    
    scaler = StandardScaler()
    scaled_cv_X = scaler.fit_transform(cv_X)
    scaled_val_X = scaler.transform(val_X)     
    
    alpha_l1 = 0.01
    alpha_l2 = 0.01

    # Define the model
    model = Sequential()

    model.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=alpha_l1, l2=alpha_l2)))
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
    selected = val_X['running_minutes'] > 60
    
    val_pred = model.predict(scaled_val_X[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)
    print(val_error)


if method == 'xgboost':
    
    grow_policy = ['depthwise', 'lossguide']

    space={'max_depth': hp.quniform("max_depth", 1, 55, 1), #try to decrease from 45 to 10?
            'min_split_loss': hp.uniform('min_split_loss', 0, 15),
            'reg_lambda' : hp.uniform('reg_lambda', 0, 20),
            'reg_alpha': hp.loguniform('reg_alpha', -3, np.log(1000)),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 45),
            'learning_rate': hp.uniform('learning_rate', 0, 0.5),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.5, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 5, 60, 1), #try to decrease to 35?
            'eval_fraction': hp.uniform('eval_fraction', 0.01, 0.4),
            'n_estimators': hp.qloguniform('n_estimators', np.log(2), np.log(1100), 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 20),
            'grow_policy': hp.choice('grow_policy', grow_policy),
            'max_leaves': hp.quniform('max_leaves', 0, 150, 1),
        }
    
    #get an validation set for fitting
    cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(train_X, train_y, sample_weights, stratify, test_size=0.25, stratify=stratify, random_state=42)


    if optimize:
    
        for i in range(len(trials.trials)+batch_size, max_evals + 1, batch_size):
            
            # Save the trials object every 'batch_size' iterations. Can save with any method you prefer
                
            #optmimize hyperparameters. use all training data       
            best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = i, 
                            trials = trials)
            
            filename = r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\hyperparams.pkl'
            pickle.dump(trials, open(filename, "wb"))
            
            best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
            best_hyperparams['min_child_weight'] = int(best_hyperparams['min_child_weight'])
            best_hyperparams['early_stopping_rounds'] = int(best_hyperparams['early_stopping_rounds'])
            best_hyperparams['n_estimators'] = int(best_hyperparams['n_estimators'])
            best_hyperparams['max_leaves'] = int(best_hyperparams['max_leaves'])
            best_hyperparams['grow_policy'] = grow_policy[best_hyperparams['grow_policy']]
        
                
            if i == batch_size:    
                print(best_hyperparams)
                
                selected = train_X['running_minutes'] > 60
                random_y = np.ones(sum(selected))*np.mean(train_y[selected])
                val_error = mean_squared_error(train_y[selected], random_y)
                
                print('Random error: ', val_error)
                
                eval_fraction = best_hyperparams['eval_fraction']
                del best_hyperparams['eval_fraction']
                
                #best_hyperparams['grow_policy'] = grow_policy[best_hyperparams['grow_policy']]
                
                best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
                best_hyperparams['min_child_weight'] = int(best_hyperparams['min_child_weight'])
                best_hyperparams['early_stopping_rounds'] = int(best_hyperparams['early_stopping_rounds'])
                best_hyperparams['n_estimators'] = int(best_hyperparams['n_estimators'])
                best_hyperparams['max_leaves'] = int(best_hyperparams['max_leaves'])
                
                #get som data for evaluation
                fit_X, eval_X, fit_y, eval_y, fit_sample_weights, _ =  train_test_split(train_X, train_y, sample_weights, test_size=eval_fraction, stratify=stratify)    
                
                #max bin sets the resolution of the output
                model = xgb.XGBRegressor(**best_hyperparams, tree_method="hist", enable_categorical=True)
                model.fit(fit_X, fit_y, verbose=False,
                    eval_set=[(fit_X, fit_y), (eval_X, eval_y)], sample_weight=fit_sample_weights) 
                
                summary = {'model': model, 'features': train_X}
                
                filename = r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\model.sav'
                pickle.dump(summary, open(filename, 'wb'))
        
        
    losses = []
    for i in range(len(trials.trials)):
        losses.append(trials.trials[i]['result']['loss'])
        
    sorted_losses = np.argsort(losses)
        
    if optimize:
        #train and test the best models
        
        
        best_loss = np.inf
        
        ind = 0
        k = 0
        
        cv_losses = []
        k_list = []
        
        from joblib import Parallel, delayed
        def parallell_train(t):
            #get an validation set for fitting
            cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(train_X, train_y, sample_weights, stratify, test_size=0.25, stratify=stratify, random_state=t)
            opt_out = objective(test_hyperparams)
            return opt_out['loss']
        
        #trian with cv until 5 models haven't made any better loss.
        while k < 100:
            
            hyperparams = trials.trials[sorted_losses[ind]]['misc']['vals']
            
            test_hyperparams = {}
            for field, val in hyperparams.items():
                test_hyperparams[field] = val[0]
        
            test_hyperparams['grow_policy'] = grow_policy[test_hyperparams['grow_policy']]
            test_hyperparams['max_depth'] = int(test_hyperparams['max_depth'])
            test_hyperparams['min_child_weight'] = int(test_hyperparams['min_child_weight'])
            test_hyperparams['early_stopping_rounds'] = int(test_hyperparams['early_stopping_rounds'])
            test_hyperparams['n_estimators'] = int(test_hyperparams['n_estimators'])
            test_hyperparams['max_leaves'] = int(test_hyperparams['max_leaves'])
            
            ind_losses = []
            
            #ind_losses = Parallel(n_jobs=-1)(delayed(parallell_train)(t) for t in range(6))
         
            
            for t in range(5):
                #get an validation set for fitting
                cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(train_X, train_y, sample_weights, stratify, test_size=0.25, stratify=stratify, random_state=t)
                opt_out = objective(test_hyperparams)
                ind_losses.append(opt_out['loss'])
        
                
            score = np.mean(ind_losses) + np.var(ind_losses, ddof=1)
                
            if score < best_loss:
                best_loss = score
                k=0
                best_trial = ind
                
            cv_losses.append(ind_losses)
            k_list.append(k)
                
            k += 1
            ind += 1
            print(score)
                
                
        mean_loss = np.mean(cv_losses, axis=1)
        var_loss =  np.std(cv_losses, ddof=1, axis=1)
        
        best_best_ind = np.argmin(mean_loss + var_loss)
        #best_best_ind = np.argmin(np.max(cv_losses, axis=1))
    else:
        best_best_ind = 84
    
    #train with all data
    best_cv_trial =  sorted_losses[best_best_ind]
     
    hyperparams = trials.trials[best_cv_trial]['misc']['vals']
    
    test_hyperparams = {}
    for field, val in hyperparams.items():
        test_hyperparams[field] = val[0]
    
    test_hyperparams['grow_policy'] = grow_policy[test_hyperparams['grow_policy']]
    test_hyperparams['max_depth'] = int(test_hyperparams['max_depth'])
    test_hyperparams['min_child_weight'] = int(test_hyperparams['min_child_weight'])
    test_hyperparams['early_stopping_rounds'] = int(test_hyperparams['early_stopping_rounds'])
    test_hyperparams['n_estimators'] = int(test_hyperparams['n_estimators'])
    test_hyperparams['max_leaves'] = int(test_hyperparams['max_leaves'])
    
    eval_fraction = test_hyperparams['eval_fraction']
    del test_hyperparams['eval_fraction']
            
    #get som data for evaluation
    fit_X, eval_X, fit_y, eval_y, fit_sample_weights, _ =  train_test_split(train_X, train_y, sample_weights, test_size=eval_fraction, stratify=stratify)    
    
    #max bin sets the resolution of the output
    model = xgb.XGBRegressor(**test_hyperparams, tree_method="hist", enable_categorical=True)
    model.fit(fit_X, fit_y, verbose=False,
        eval_set=[(fit_X, fit_y), (eval_X, eval_y)], sample_weight=fit_sample_weights)
        
    summary = {'model': model, 'features': train_X}
        
    filename = r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\model.sav'
    pickle.dump(summary, open(filename, 'wb'))
        
    xgb.plot_importance(model)
        
        
    
    # def unpack(x):
    #     if x:
    #         return x[0]
    #     return np.nan
    
    
    # # We'll first turn each trial into a series and then stack those series together as a dataframe.
    # trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])
    # trials_df["loss"] = [t["result"]["loss"] for t in trials]
    
    # trials_list = np.argsort(trials_df["loss"])
    
    # eval_hyperparams = trials.trials[trials_list[0]]["misc"]["vals"]
    
    # test_hyperparams = {}
    # for field, val in eval_hyperparams.items():
    #     if isinstance(val, list):
    #         test_hyperparams[field] = val[0]
    #     else:
    #         test_hyperparams[field] = val
            
    # test_hyperparams['grow_policy'] = grow_policy[test_hyperparams['grow_policy']]
    # test_hyperparams['max_depth'] = int(test_hyperparams['max_depth'])
    # test_hyperparams['min_child_weight'] = int(test_hyperparams['min_child_weight'])
    # test_hyperparams['early_stopping_rounds'] = int(test_hyperparams['early_stopping_rounds'])
    # test_hyperparams['n_estimators'] = int(test_hyperparams['n_estimators'])
    # test_hyperparams['max_leaves'] = int(test_hyperparams['max_leaves'])
    
    # #get som data for evaluation
    # fit_X, eval_X, fit_y, eval_y, fit_sample_weights, _ =  train_test_split(train_X, train_y, sample_weights, test_size=eval_fraction, stratify=stratify)    
    
    # #max bin sets the resolution of the output
    # model = xgb.XGBRegressor(**test_hyperparams, tree_method="hist", enable_categorical=True)
    # model.fit(fit_X, fit_y, verbose=False,
    #     eval_set=[(fit_X, fit_y), (eval_X, eval_y)], sample_weight=fit_sample_weights) 
        
    # filename = r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\model.sav'
    # pickle.dump(model, open(filename, 'wb'))
            
        