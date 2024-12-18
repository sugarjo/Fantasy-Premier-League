minutes_thisyear_treshold = -1
form_treshold = -1
points_per_game_treshold = -1

exclude_team = []


exclude_players = ['Henderson', 'Bobb', 'Watkins', 'Wilson', 'Mings', 'Elliott', 'De Bruyne', 'Willock', 'Ramsdale', 'Botman', 'Tomiyasu', 'Foden', 'Dúbravka', 'Turner', 'Chilwell', 'Odysseas', 'Bajcetic', 'Johnstone']

exclude_players_out = []

include_players = ['Haaland']

do_not_exclude_players = []

rounds_to_value = 5
           
wildcard = True

skip_gw = []
benchboost_gw = 38
jump_rounds = 0

force_90 = []

manual_pred = 1

# manual_blanks = {22: ['Mitoma', 'Onana', 'Son', 'Hee Chan', 'Salah', 'Kudus', 'Semenyo', 'Mbeumo', 'Sarr', 'Tomiyasu'],
#                  }
manual_blanks = {}

string = '{"picks":[{"element":310,"position":1,"selling_price":55,"multiplier":1,"purchase_price":55,"is_captain":false,"is_vice_captain":false},{"element":311,"position":2,"selling_price":70,"multiplier":1,"purchase_price":70,"is_captain":false,"is_vice_captain":false},{"element":24,"position":3,"selling_price":65,"multiplier":1,"purchase_price":65,"is_captain":false,"is_vice_captain":false},{"element":404,"position":4,"selling_price":45,"multiplier":1,"purchase_price":45,"is_captain":false,"is_vice_captain":false},{"element":199,"position":5,"selling_price":70,"multiplier":1,"purchase_price":70,"is_captain":false,"is_vice_captain":true},{"element":182,"position":6,"selling_price":105,"multiplier":1,"purchase_price":105,"is_captain":false,"is_vice_captain":false},{"element":17,"position":7,"selling_price":100,"multiplier":1,"purchase_price":100,"is_captain":false,"is_vice_captain":false},{"element":317,"position":8,"selling_price":75,"multiplier":1,"purchase_price":75,"is_captain":false,"is_vice_captain":false},{"element":401,"position":9,"selling_price":85,"multiplier":2,"purchase_price":85,"is_captain":true,"is_vice_captain":false},{"element":207,"position":10,"selling_price":75,"multiplier":1,"purchase_price":75,"is_captain":false,"is_vice_captain":false},{"element":447,"position":11,"selling_price":60,"multiplier":1,"purchase_price":60,"is_captain":false,"is_vice_captain":false},{"element":413,"position":12,"selling_price":50,"multiplier":0,"purchase_price":50,"is_captain":false,"is_vice_captain":false},{"element":584,"position":13,"selling_price":55,"multiplier":0,"purchase_price":55,"is_captain":false,"is_vice_captain":false},{"element":44,"position":14,"selling_price":45,"multiplier":0,"purchase_price":45,"is_captain":false,"is_vice_captain":false},{"element":241,"position":15,"selling_price":45,"multiplier":0,"purchase_price":45,"is_captain":false,"is_vice_captain":false}],"chips":[{"status_for_entry":"available","played_by_entry":[],"name":"bboost","number":1,"start_event":1,"stop_event":38,"chip_type":"team","is_pending":false},{"status_for_entry":"available","played_by_entry":[],"name":"3xc","number":1,"start_event":1,"stop_event":38,"chip_type":"team","is_pending":false}],"transfers":{"cost":4,"status":"unlimited","limit":null,"made":0,"bank":0,"value":1000}}'

season = '2024-25'
previous_season = '2023-24'

import requests
import pandas as pd
import numpy as np
import json
import pickle
import difflib
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import os
from hyperopt.pyll import scope
from hyperopt import STATUS_OK, STATUS_FAIL, space_eval, Trials, fmin, hp, tpe, rand, SparkTrials
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate
import random


#insert string for team
directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data' + '/' + season
prev_season_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data' + '/' + previous_season
team_path = directory + "/teams.csv"
    
df_teams = pd.read_csv(team_path)
string_names = df_teams['short_name'].values

#log in
session = requests.session()
url = 'https://users.premierleague.com/accounts/login/'
payload = {
 'password': 'jorgeN8#larseN(3',
 'login': 'jorgen.sugar@gmail.com',
 'redirect_uri': 'https://fantasy.premierleague.com',
 'app': 'plfpl-web'
}
session.post(url, data=payload)

#get my team and money in the bank
r = session.get('https://fantasy.premierleague.com/api/my-team/2088464/')
js = r.json()

js = json.loads(string)

my_players = pd.DataFrame(js['picks'])
a = json.dumps(js)
js = json.loads(a)
transfers = js["transfers"]
bank = transfers['bank']

if wildcard or transfers['status'] == 'unlimited':
    free_transfers = 15
    unlimited_transfers = True
    print('Free transfers: ', 15)
else:
    unlimited_transfers = False
    free_transfers = transfers["limit"] - transfers["made"]
    print('Free transfers: ', free_transfers)
    
if free_transfers < 0:
    free_transfers = 0
      
#subtract 1 since we add one for each gw later
free_transfers -= 1    

#get statistics of all players
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
js = r.json()

elements_df = pd.DataFrame(js['elements'])
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
events_df = pd.DataFrame(js['events'])


dfs_gw = []

#check if directory exist if not use the one from previous years
if os.path.exists(directory + '/gws'):
    gw_dir = directory + '/gws'
    season_data = True
    days_avg = days_avg
else:
    gw_dir = prev_season_directory + '/gws'
    print('Directory for current season not found')
    season_data = False   
    days_avg = '90D'
    
#open each gw and get data for players
for gw_csv in os.listdir(gw_dir):
    if gw_csv[0] == 'g':
        
        gw_path = gw_dir + '/' + gw_csv
                
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
df_gw['new_year_element'] = np.nan

#add column if they don't exist

from difflib import SequenceMatcher

def sequence_matcher_similarity(s1, s2):
    return SequenceMatcher(None, ' '.join(sorted(s1.split())), ' '.join(sorted(s2.split()))).ratio()
    
# Calculate rolling values not including the observaiton. 
for player in df_gw['element'].unique():
    
    #match by element ID
    if season_data:
        slim_ind = slim_elements_df['id'] == player
        selected_ind = (df_gw['element'] == player) & (df_gw['minutes'] > 0)
    #match by name if no gws are available
    else:
        selected = df_gw['element'] == player
        df_name = df_gw['name'][selected].iloc[0]
        slim_names = slim_elements_df['first_name'] + ' ' + slim_elements_df['second_name']
        
        similarity = slim_names.apply(lambda x: sequence_matcher_similarity(x, df_name))

        slim_ind = np.argmax(similarity)
        
        slim_name = slim_names[slim_ind]
        
        if max(similarity) < 0.8 and not (df_name in slim_name or slim_name in df_name):
            #print(df_name + ': does not exist in database')
            continue
        
        selected_ind = (df_gw['name'] == df_name)
        df_gw.loc[selected_ind, 'new_year_element'] = slim_elements_df.iloc[slim_ind]['id']
        
        selected_ind = (df_gw['name'] == df_name) & (df_gw['minutes'] > 0)
        
    
    
    # if name in force_90:
    #     selected_ind = (df_gw['element'] == player) & (df_gw['minutes'] > 0)
    # else:    
    #     selected_ind = df_gw['element'] == player
        
    
    if sum(selected_ind) == 0:
        continue
    
    name = slim_elements_df.loc[slim_ind, 'web_name']
        
    player_df = df_gw[selected_ind]
    player_df.set_index('kickoff_time', inplace=True)
    
    form = player_df['total_points'].rolling(days_avg).mean()
    xG = player_df['expected_goals'].rolling(days_avg).mean()
    xA = player_df['expected_assists'].rolling(days_avg).mean()
    xGI = player_df['expected_goal_involvements'].rolling(days_avg).mean()
    xGC = player_df['expected_goals_conceded'].rolling(days_avg).mean()
    xP = player_df['xP'].rolling(days_avg).mean()
    points_per_game =  player_df['total_points'].cumsum()/ (player_df['round'])
    ict = player_df['ict_index'].rolling(days_avg).mean()
    influence = player_df['influence'].rolling(days_avg).mean()
    threat = player_df['threat'].rolling(days_avg).mean()
    creativity = player_df['creativity'].rolling(days_avg).mean()
    bps = player_df['bps'].rolling(days_avg).mean()
    minutes = player_df['minutes'].rolling(days_avg).mean().values
    
    if name in force_90:
        #average across season (non zero matches)
        minutes[-1] = np.mean(player_df['minutes'])        
        form[-1] = np.mean(player_df['total_points'])
        xG[-1] = np.mean(player_df['expected_goals'])
        xA[-1] = np.mean(player_df['expected_assists'])
        xGI[-1] = np.mean(player_df['expected_goal_involvements'])
        xGC[-1] = np.mean(player_df['expected_goals_conceded'])
        xP[-1] = np.mean(player_df['xP'])
        #points_per_game[-1] =  np.mean(player_df['total_points'].cumsum()/ (player_df['round'])
        ict[-1] = np.mean(player_df['ict_index'])
        influence[-1] = np.mean(player_df['influence'])
        threat[-1] = np.mean(player_df['threat'])
        creativity[-1] = np.mean(player_df['creativity'])
        bps[-1] = np.mean(player_df['bps'])
        
    #if two last matches are not played then set to zero
    elif sum(player_df['minutes'][-2:]) < 90 and season_data:
        minutes[-1] = 0
        
    #points per played game
    result = np.zeros(len(player_df['total_points'])+1)  # initialize result array
    last_games = 0  # initialize last_vplayer_df['total_points']alue to 0
    last_point = 0
    
    for i in range(len(player_df['total_points'])):
        
        if player_df['minutes'].iloc[i] >= 60:
            last_point += player_df['total_points'].iloc[i]
            last_games += 1
        
        if last_games > 0:
            result[i+1] = last_point/last_games
            
    #redefine to isnert in all matches
    if season_data:
        selected_ind = df_gw['element'] == player
    else:
        selected_ind = (df_gw['name'] == df_name)
    
    df_gw.loc[selected_ind, 'running_ict'] = ict.values[-1]
    df_gw.loc[selected_ind, 'running_influence'] = influence.values[-1]
    df_gw.loc[selected_ind, 'running_threat'] = threat.values[-1]
    df_gw.loc[selected_ind, 'running_creativity'] = creativity.values[-1]
    df_gw.loc[selected_ind, 'running_bps'] = bps.values[-1]
    df_gw.loc[selected_ind, 'form'] = form.values[-1]
    df_gw.loc[selected_ind, 'running_xG'] = xG.values[-1]
    df_gw.loc[selected_ind, 'running_xA'] = xA.values[-1]
    df_gw.loc[selected_ind, 'running_xGI'] = xGI.values[-1]
    df_gw.loc[selected_ind, 'running_xGC'] = xGC.values[-1] 
    df_gw.loc[selected_ind, 'running_xP'] = xP.values[-1] 
    df_gw.loc[selected_ind, 'points_per_game'] = points_per_game.values[-1]
    df_gw.loc[selected_ind, 'points_per_played_game'] = result[:-1][-1]
    df_gw.loc[selected_ind, 'running_minutes'] = minutes[-1]
    
if not season_data:
    df_gw.element = df_gw.new_year_element

i=0

 
while pd.to_datetime(events_df.deadline_time[i], format='%Y-%m-%dT%H:%M:%SZ') < datetime.now() - timedelta(hours=2):
    i = i + 1

current_gameweek = i + 1

print('previous:')
#get statistics for the past gameweeks
df_past_games = pd.DataFrame(columns=['gameweek', 'team_h', 'team_a', 'difficulty_diff'])
for this_gw in range(1, current_gameweek):
    print(this_gw)
    url = 'https://fantasy.premierleague.com/api/fixtures' + '?event=' + str(this_gw)
    r = requests.get(url)
    gw = r.json()

    for game in gw:
        add_frame = pd.DataFrame({'gameweek': this_gw, 'team_h': int(game['team_h']), 'team_a': int(game['team_a']), 'difficulty_diff': int(game['team_h_difficulty']) - int(game['team_a_difficulty'])}, index = [0])        
        df_past_games = pd.concat([df_past_games, add_frame])
        
print('current gameweek: ' + str(current_gameweek))
        
print('predicting:')
#get statistics for the x next gameweeks
df_future_games = pd.DataFrame(columns=['gameweek', 'team_h', 'team_a', 'difficulty_diff'])
benchboost = []
for i in range(jump_rounds, rounds_to_value+jump_rounds):
    this_gw = i + current_gameweek
    
    if benchboost_gw == this_gw:
        benchboost.append(True)
    else:
        benchboost.append(False)
    
    if any(np.array(skip_gw) == this_gw):
        continue
    
    print(this_gw)
    
    url = 'https://fantasy.premierleague.com/api/fixtures' + '?event=' + str(this_gw)
    r = requests.get(url)
    gw = r.json()

    for game in gw:
        #positive home values == difficult games
        add_frame = pd.DataFrame({'gameweek': this_gw, 'gameweek_ind': i-jump_rounds, 'team_h': int(game['team_h']), 'team_a': int(game['team_a']), 'difficulty_diff': int(game['team_h_difficulty']) - int(game['team_a_difficulty']), 'difficulty_home': int(game['team_h_difficulty']), 'difficulty_away': int(game['team_a_difficulty'])}, index = [0])        
        df_future_games = pd.concat([df_future_games, add_frame])

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

for name in do_not_exclude_players:
    ind = name == names
    selected_players[ind] = False

# points_per_game[selected_players] = 0

with open(r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\model.sav', 'rb') as f:
    summary = pickle.load(f)

result = summary["model"]

train_X = summary["features"]

predictions = []


#remember to control order of features in df_predict!!!
keep_features = ['running_minutes', 'transfer_in', 'transfer_out', 'running_ict', 'running_influence', 'running_threat', 'running_creativity', 'running_bps', 'string_opp_team', 'string_team', 'names', 'element_type', 'was_home', 'running_xP', 'running_xG', 'running_xA', 'running_xGI', 'running_xGC', 'form',
                 'points_per_game', 'points_per_played_game', 'other_difficulty', 'own_difficulty']

    
#predict future points
for df_name in slim_elements_df.iterrows():
    
    # if df_name[1].second_name == 'C.Richards':
    #     print(df_name)
    # ind=576
    # df_name = (ind, slim_elements_df.iloc[ind])

    if not selected_players[df_name[0]]:
        team = int(df_name[1].team)
        position = int(df_name[1].element_type)     
        first_name = df_name[1].first_name
        second_name = df_name[1].second_name
        name = first_name + ' ' + second_name
        form = df_name[1].form
        player_id =  df_name[1].id
        
        url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id)
        r = requests.get(url)
        player = r.json()
        
        # player_games = pd.DataFrame(player['history'])
        
        # player_games['kickoff_time'] =  pd.to_datetime(player_games['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        # player_games = player_games.sort_values(by='kickoff_time')
        # player_games.set_index('kickoff_time', inplace=True)
        
        fixtures = pd.DataFrame(player['fixtures'])
        fixtures['kickoff_time'] =  pd.to_datetime(fixtures['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        fixtures = fixtures.sort_values(by='kickoff_time')        
        
        should_have_trainingdata = True
        should_have_database = False
        past_history  = player["history_past"]
        if past_history == []:
            should_have_trainingdata = False
        
        else:
            last_history = past_history[-1]['season_name']
            
            if last_history[:4] == previous_season[:4]:
                should_have_database = True
        
        #check if player does not exist in df_gw database. use data from slim
        if sum(df_gw.element == player_id) == 0:
            selected_ind = np.where(elements_df.id == player_id)[0][-1]
            
            if (not season_data and should_have_database) or season_data:
                print(elements_df.iloc[selected_ind].web_name, ': estimate values. Does not exist in game database.')
            
            #at beginnig of season data contains season sums
            if season_data:
                played_games = 1
                form = float(elements_df.iloc[selected_ind].form)
                points_per_game = float(elements_df.iloc[selected_ind].points_per_game)
            else:
                played_games = np.round((elements_df.iloc[selected_ind].total_points / (float(elements_df.iloc[selected_ind].points_per_game)+1e-6))) + 1e-6
                form = np.nan
                
                
            xP = np.nan
            xG = float(elements_df.iloc[selected_ind].expected_goals) / played_games
            xA = float(elements_df.iloc[selected_ind].expected_assists) / played_games
            xGI = float(elements_df.iloc[selected_ind].expected_goal_involvements) / played_games
            xGC = float(elements_df.iloc[selected_ind].expected_goals_conceded) / played_games
            ict = float(elements_df.iloc[selected_ind].ict_index) / played_games
            influence = float(elements_df.iloc[selected_ind].influence) / played_games
            threat = float(elements_df.iloc[selected_ind].threat) / played_games
            creativity = float(elements_df.iloc[selected_ind].creativity) / played_games
            bps = float(elements_df.iloc[selected_ind].bps) / played_games
            points_per_played_game = float(elements_df.iloc[selected_ind].points_per_game)            
            minutes = np.min([float(elements_df.iloc[selected_ind].minutes) / played_games, 90]) 
            points_per_game = float(elements_df.iloc[selected_ind].points_per_game)
            
        #if they exist use data from df_gw
        else:
            
            id_ind  = df_gw.element == player_id
            #min_ind = ~np.isnan(df_gw.running_minutes)
            selected_ind = np.where(id_ind)[0][-1]
            
            xG = df_gw.loc[selected_ind]['running_xG']
            xA = df_gw.loc[selected_ind]['running_xA']
            xGI = df_gw.loc[selected_ind]['running_xGI']
            xGC = df_gw.loc[selected_ind]['running_xGC']
            xP = df_gw.loc[selected_ind]['running_xP']
            ict = df_gw.loc[selected_ind]['running_ict']
            influence = df_gw.loc[selected_ind]['running_influence']
            threat = df_gw.loc[selected_ind]['running_threat']
            creativity = df_gw.loc[selected_ind]['running_creativity']
            bps = df_gw.loc[selected_ind]['running_bps']
            points_per_game = df_gw.loc[selected_ind]['points_per_game']
            points_per_played_game = df_gw.loc[selected_ind]['points_per_played_game']
            
            minutes = df_gw.loc[selected_ind]['running_minutes']
            form = df_gw.loc[selected_ind]['form']
            
        selected_ind = np.where(elements_df.id == player_id)[0][-1]
        
        sum_transfers = sum(elements_df.transfers_in_event)
        if sum_transfers == 0 or not season_data:
            transfer_in = np.nan
            transfer_out = np.nan
        else:
            transfer_in = elements_df.iloc[selected_ind].transfers_in_event/sum_transfers
            transfer_out = elements_df.iloc[selected_ind].transfers_out_event/sum_transfers
                
        if position == 4:
            xG_multiplier = 4
            xGC_multiplier = 0
        elif position == 3:
            xG_multiplier = 5
            xGC_multiplier = 1
        else:
            xG_multiplier = 6
            xGC_multiplier = 4           
        
        #matches with team
        selected_matches = np.logical_or(df_future_games.team_h == team, df_future_games.team_a == team)
        gws = df_future_games[selected_matches]
        
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

        for game in gws.iterrows():
            
            game_idx = game[0]
            gw_idx = int(game[1].gameweek_ind)
            gw = game[1].gameweek
            
            if home_matches[game_idx]:
                own_difficulty = home_difficulty[game_idx]
                other_difficulty = away_difficulty[game_idx]
                string_team = string_names[game[1].team_h-1]
                string_opp_team = string_names[game[1].team_a-1]
                home = 1
            else:
                own_difficulty = away_difficulty[game_idx]
                other_difficulty = home_difficulty[game_idx]
                string_opp_team = string_names[game[1].team_h-1]
                string_team = string_names[game[1].team_a-1]
                home = 0
            
                
            df_game = pd.DataFrame(columns=keep_features)
           
            if game[1].gameweek == current_gameweek:
                df_game.loc[0] = [minutes, transfer_in, transfer_out, ict, influence, threat, creativity, bps, string_opp_team, string_team, name,
                                                   position, home, xP, xG*xG_multiplier, xA*3, xGI, xGC*xGC_multiplier, form,
                                                   points_per_game, points_per_played_game, other_difficulty, own_difficulty]
            else:
                df_game.loc[0] = [minutes, np.nan, np.nan, ict, influence, threat, creativity, bps, string_opp_team, string_team, name,
                                                   position, home, xP, xG*xG_multiplier, xA*3, xGI, xGC*xGC_multiplier, form,
                                                   points_per_game, points_per_played_game, other_difficulty, own_difficulty]
            
            #merge with train X to maintain category labeling
            df_predict = pd.concat([df_game, train_X])
            
            df_predict['element_type'] = df_predict['element_type'].astype('category')
            df_predict['names'] = df_predict['names'].astype('category')
            df_predict['other_difficulty'] = df_predict['other_difficulty'].astype('category')
            df_predict['own_difficulty'] = df_predict['own_difficulty'].astype('category')
            df_predict['string_team'] = df_predict['string_team'].astype('category')
            df_predict['string_opp_team'] = df_predict['string_opp_team'].astype('category')
            


            #estimated = 10**result.predict(df_predict) 
            estimated = result.predict(df_predict.iloc[0:1])[0]     
            
            if minutes < 10 or np.isnan(minutes):
                estimated = 0.1
            
            #remove if unlikely to play: game_idx for game. gw_idx for gw
            if gw_idx==0 and gw_idx+jump_rounds == 0 and df_name[1]['chance_of_playing_next_round'] < 75:
                estimated = 0.1
                
            if string_team in exclude_team:
                estimated = 0.1
                
            if gw in manual_blanks.keys():
                if df_name[1]['web_name'] in manual_blanks[gw]:
                    estimated = 0.1
                    
            if sum(train_X.names == df_game.names[0]) == 0 and game[0]==0:
                if should_have_trainingdata:
                    print(df_game.names[0] + ': does not exist in training data.')
                estimated = 0.1 
                
            if df_name[1]['web_name'] in include_players:
                estimated = 100
                
            for name_inc in include_players:
                if df_name[1]['first_name'] in name_inc and df_name[1]['second_name'] in name_inc:
                    estimated = 100
                
            if df_name[1]['web_name'] in exclude_players:
                estimated = 0.1
            
            for exclude_name in exclude_players:
                if df_name[1]['first_name'] in exclude_name and df_name[1]['second_name'] in exclude_name:
                    estimated = 0.1
            
            pred_score[gw_idx] = pred_score[gw_idx] + estimated
            total_matches = total_matches + 1

            
        first_gw = pred_score[0]

        #predicted_points = pred_score/total_matches - (4 / rounds_to_reset)
        predicted_points = pred_score  #- (4 / rounds_to_reset)
        predicted_values[df_name[0]] = predicted_points
        predicted_values_1st_gw[df_name[0]] = first_gw
        predictions.append(pred_score)
        
    else:
        predictions.append(np.zeros(rounds_to_value).astype(float))


slim_elements_df['points_1st_gw'] = predicted_values_1st_gw

#set what to use for evaluation. can be points_per_game
prediction = np.copy(predicted_values)

#save for later
original_prediction = np.copy(predictions)

    
# remove (=set to zero) low form
selected = slim_elements_df['form'] < form_treshold
prediction[selected] = 0.1
for ind in np.where(selected)[0]:
    predictions[ind]=np.zeros(rounds_to_value).astype(float)


slim_elements_df['prediction'] = predictions

#start out with blank team (none are picked)
slim_elements_df['picked'] = False
slim_elements_df['original_player'] = False

#initiate variables counting number of players in each position/team
num_position = np.zeros([4, 1])
num_team = np.zeros([20, 1])



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

    print(list(slim_elements_df.web_name[selected])[0] + ' ' + str(sum((predicted_values[selected]))))
    
    
original_players = my_players_df

now_cost = slim_elements_df['now_cost'].astype(float)
value = slim_elements_df['prediction'].apply(sum) / now_cost
slim_elements_df['value'] = value

#find points for each match or a series of matches (depends on len of prediction)
def find_team_points(team_positions, gw_prediction, benchboost):
    
    if benchboost:
        captain_ind = np.argmax(gw_prediction)
        
        gw_prediction[captain_ind] = gw_prediction[captain_ind]*2
        
        return sum(gw_prediction)
    
    else:

        pred_points = []
                
        order = np.argsort(gw_prediction)
        ordered_points = np.sort(gw_prediction)
        ordered_positions = team_positions[order]
        
        #pick the 11 best players of the team
        for i in range(11):   
                    
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
        
        pred_points[captain_ind] = pred_points[captain_ind]*2
        
        return sum(pred_points)

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
#start out by putting some to nan and other to it's predicition

#loop players
for j in range(player_iteration): 
    
    #loop gws
    for i in range(gw_iteration):
        transfers = []
        probability_hit = []
        probability_main_ifhit = []
        probability_main_ifnohit = []
        
        ind_next = 0      

        #loop transfers
        for player_out in slim_elements_df.iterrows():
            #check if picked
            if player_out[1]['picked']:
                
                for player_in in slim_elements_df.iterrows():
                    
                    #check if not picked, not same the other player, any predictions >0 and same element
                    if not player_in[1]['picked'] and sum(player_in[1].prediction) > 0 and (any(player_in[1].prediction > player_out[1].prediction) or player_in[1].now_cost < player_out[1].now_cost) and  player_in[1].element_type == player_out[1].element_type:
                        
                        transfers.append([player_out[0], player_in[0]])
                        
                        if unlimited_transfers and j is not ind_next:
                            probability_hit.append(np.nan)
                            probability_main_ifhit.append(np.nan)
                            probability_main_ifnohit.append(np.nan)
                            continue  
                        
                        #if less expensive or more gain
                        if player_in[1].prediction[i] < player_out[1].prediction[i] and (player_in[1].now_cost > player_out[1].now_cost):
                            probability_hit.append(np.nan)
                            probability_main_ifhit.append(np.nan)
                            probability_main_ifnohit.append(np.nan)
                            continue
                        
                        #make sure that excluded players are transfered out.
                        if i==0 and len(exclude_players_out) > 0:
                            probability_hit.append(np.nan)
                            probability_main_ifhit.append(np.nan)
                            probability_main_ifnohit.append(np.nan)
                            continue
        
                            
                        preds = np.cumsum((predictions[player_in[0]] - predictions[player_out[0]])[::-1])[::-1]   
                        
                        #for main and hit we can not accept lower score and higher price
                        if ((i == rounds_to_value-1 and preds[i] <= 0) or (i < rounds_to_value-1 and (preds[i] <= preds[i+1]))) and (player_in[1].now_cost > player_out[1].now_cost):
                            probability_main_ifhit.append(np.nan)
                        else:
                            probability_main_ifhit.append(preds[i])
                            
                            
                        #for main an no hit we can not accept lower score
                        if ((i == rounds_to_value-1 and preds[i] <= 0) or (i < rounds_to_value-1 and (preds[i] <= preds[i+1]))):
                            probability_main_ifnohit.append(np.nan)
                        else:
                            probability_main_ifnohit.append(preds[i])
                            
                        #for hits we need at least 4 points increase, and we need increas in current round
                        if ((i == rounds_to_value-1 and preds[i] >= transfer_cost) or (i < rounds_to_value-1 and any(preds[i:] >= transfer_cost))) and player_in[1].prediction[i] > player_out[1].prediction[i]:                  
                            probability_hit.append(preds[i])
                        else:
                            probability_hit.append(np.nan)
                            
                ind_next += 1
        
        
        probability_main_ifhit.append(4)
        probability_main_ifnohit.append(4)
        
        transfers.append([np.nan, np.nan])
        
        #for each player-gw: add the probability into the initating variables. 3 transfers per round.
        
        
        if unlimited_transfers:
            point_diff.append(probability_main_ifhit)            
        #if all are nan for hits (no hots possible) and not wild card
        elif all(elem is np.nan for elem in probability_hit):
            point_diff.append(probability_main_ifnohit) 
            point_diff.append(probability_main_ifnohit) 
            point_diff.append(probability_main_ifnohit) 
        else:
            point_diff.append(probability_main_ifhit) 
            point_diff.append(probability_main_ifhit) 
            point_diff.append(probability_main_ifhit) 


#calculate points for a given set of transfers
def objective(check_transfers, free_transfers):      
    
    #print(check_transfers)
        
    team = slim_elements_df['picked'].values.copy()
    
    # print(params)
    
    if unlimited_transfers:
        gw_iteration = 1
    else:
        gw_iteration = rounds_to_value
    
    #loop through the transfers and check if they are possible
    for gw in range(gw_iteration):   
        if not unlimited_transfers:        
            transfer1 = check_transfers[gw*3]
            transfer2 = check_transfers[(gw*3)+1]
            transfer3 = check_transfers[(gw*3)+2]

            if not np.isnan(transfer1[0]):
                #check if players are already transfered
                if team[transfer1[0]] == False or team[transfer1[1]] == True:
                    return np.nan, np.nan
                
                team[transfer1[0]] = False
                team[transfer1[1]] = True
            if not np.isnan(transfer2[0]):
                #check if players are already transfered
                if team[transfer2[0]] == False or team[transfer2[1]] == True:
                    return np.nan, np.nan
                
                team[transfer2[0]] = False
                team[transfer2[1]] = True
                
            if not np.isnan(transfer3[0]):
                #check if players are already transfered
                if team[transfer3[0]] == False or team[transfer3[1]] == True:
                    return np.nan, np.nan
                
                team[transfer3[0]] = False
                team[transfer3[1]] = True
                    
        else:
            #check all transfers before moving on
            for transfer in check_transfers:
                if not np.isnan(transfer[0]):
                    team[transfer[0]] = False
                    team[transfer[1]] = True
                    
        #if too expensive or too many players from club
        total_price =  sum(slim_elements_df.loc[team, 'now_cost'])
    
        #count_clubs
        num_team = np.zeros((20))
        for team_ind in slim_elements_df.loc[team, 'team']:
            num_team[team_ind-1] += 1
            
        if total_money < total_price or np.max(num_team) > 3 or sum(team) != 15: 
            # if total_money < total_price:
            #     print('money')
            # if np.max(num_team) > 3:
            #     print('team')
            # if sum(team) != 15:               
            #     print('overlap')
    
            return np.nan, np.nan
        
    team = slim_elements_df['picked'].values.copy()
    
    team_points = []

    #loop through the transfers and count points
    for gw in range(gw_iteration):
        
        if not unlimited_transfers:        
            
            #if all pred is zero skip week (=free hit)
            if sum(predictions[:, gw]) == 0:
                estimated_points = 0
            else:                
                free_transfers +=1
            
                transfer1 = check_transfers[gw*3]
                transfer2 = check_transfers[(gw*3)+1]
                transfer3 = check_transfers[(gw*3)+2]
                if not np.isnan(transfer1[0]):
                    team[transfer1[0]] = False
                    team[transfer1[1]] = True
                    free_transfers -=1
                if not np.isnan(transfer2[0]):
                    team[transfer2[0]] = False
                    team[transfer2[1]] = True
                    free_transfers -=1
                if not np.isnan(transfer3[0]):
                    team[transfer3[0]] = False
                    team[transfer3[1]] = True
                    free_transfers -=1
                    
                if free_transfers < 0:
                    team_points.append(transfer_cost*free_transfers)
                    free_transfers = 0
                    
                if free_transfers > 5:
                    free_transfers = 5  
                        
                gw_prediction = predictions[team, gw]
                team_positions = slim_elements_df.loc[team, 'element_type'].values
            
                estimated_points = find_team_points(team_positions, gw_prediction, benchboost[gw])
        
            team_points.append(estimated_points)
        
        else:
            #loop all transfers before calculating the points.
            for transfer in check_transfers:
                if not np.isnan(transfer[0]):
                    team[transfer[0]] = False
                    team[transfer[1]] = True              
                    
            for gws in range(rounds_to_value):
                gw_prediction = predictions[team, gws]
                team_positions = slim_elements_df.loc[team, 'element_type'].values
                
                estimated_points = find_team_points(team_positions, gw_prediction, benchboost[gws])
                
                team_points.append(estimated_points)

        
        #print(sum(team_points))
    
        
    return sum(team_points), total_price



def check_random_transfers(i):    

    rng = np.random.default_rng(seed=i)
    
    evaluated_transfers = []
    points = []
    prices = []
    
    counts = np.zeros((len(point_diff), len(probabilities[0])), dtype='uint32')
    sum_points = np.zeros((len(point_diff), len(probabilities[0])))
    
    for j in range(batch_size):
        
        #loop to get a transfer combination
        transfer_ind = []
        putative_transfers = []
        for i in range(len(point_diff)):            
            trans_ind = rng.choice(np.arange(prob.shape[0]), 1, p=prob[:, i])[0]
            trans = transfers[trans_ind]
            
            #redo to nan if player is allready transfered in/out
            if i > 0:            
                for t in putative_transfers:
                    if t[0] == trans[0] or t[1] == trans[1]:
                        #skip every third transfer
                        trans_ind = prob.shape[0]-1
                        break
                
            transfer_ind.append(trans_ind)
            trans = transfers[trans_ind]
            putative_transfers.append(trans)
        
        # putative_transfers = []
        # for i in [973, 983, 983]:
        #     trans = transfers[i]
        #     putative_transfers.append(trans)
                
            
        
        # if (transfer_ind not in checked_transfers) and (transfer_ind not in evaluated_transfers):
        
        point, price = objective(putative_transfers, free_transfers)

        points.append(point)
        prices.append(price)
        evaluated_transfers.append(transfer_ind)
            
        for week, transfer in enumerate(transfer_ind):         
            if not np.isnan(point):                  
                sum_points[week, transfer] = sum_points[week, transfer] + (point-baseline)
                counts[week, transfer] += 1
            #punish also nan teams
            else:
                counts[week, transfer] += 1
        
    
    
    if not np.isnan(points).all():
        max_value = np.nanmax(points)
        
        indices_with_max_value = [i for i, value in enumerate(points) if value == max_value]
        min_value_other_list = min(prices[i] for i in indices_with_max_value)
        best_ind = next(i for i in indices_with_max_value if prices[i] == min_value_other_list)

        best_point = points[best_ind]
        best_price = prices[best_ind]
        best_transfer = evaluated_transfers[best_ind]
            
            
        check_guided = True
        while check_guided:        
            check_guided = False
            
            #guided part. exhange one transfer
            for k in range(prob.shape[1]):           
                
                guided_points, guided_prices, guided_evaluated_transfers, guided_sum_points, guided_counts = check_guided_transfers(k, best_transfer)
                
                points = points + guided_points
                prices = prices + guided_prices
                evaluated_transfers = evaluated_transfers + guided_evaluated_transfers
                sum_points += guided_sum_points
                counts += guided_counts
                
                max_value = np.nanmax(points)
                
                indices_with_max_value = [i for i, value in enumerate(points) if value == max_value]
                min_value_other_list = min(prices[i] for i in indices_with_max_value)
                best_ind = next(i for i in indices_with_max_value if prices[i] == min_value_other_list)
                
                guided_best_price = prices[best_ind]
                
                
                if max_value > best_point or (max_value == best_point and guided_best_price < best_price):
                    print(k)
                    check_guided = True
                    best_point = points[best_ind]
                    best_price = guided_best_price
                    best_transfer = evaluated_transfers[best_ind]                

    return [points, evaluated_transfers, sum_points, counts]

def check_guided_transfers(i, best_transfer):    
   
    evaluated_transfers = []
    points = []
    prices = []
    
    counts = np.zeros((len(point_diff), len(probabilities[0])), dtype='uint32')
    sum_points = np.zeros((len(point_diff), len(probabilities[0])))
    
    #loop to get the transfer combination
    transfer_ind = []
    putative_transfers = []
    for j in best_transfer:
        transfer_ind.append(j)
        putative_transfers.append(transfers[j])
            
    random_ordered_transfers = list(range(len(transfers)))
    random.shuffle(random_ordered_transfers)
    
    original_transfer = transfer_ind[i]
    
    #exhange one of the transfers
    for j in random_ordered_transfers:
        if prob[j, i] > 0:
            
            transfer_ind[i] = j
            putative_transfers[i] = transfers[transfer_ind[i]]
                
            # if (transfer_ind not in checked_transfers) and (transfer_ind not in evaluated_transfers):
            
            point, price = objective(putative_transfers, free_transfers)
            points.append(point)
            prices.append(price)
            evaluated_transfers.append(transfer_ind.copy())                
                   
            if not np.isnan(point):          
                sum_points[i, transfer_ind[i]] = sum_points[i, transfer_ind[i]] + (point-baseline)
                counts[i, transfer_ind[i]] += 1
                
                sum_points[i, original_transfer] = sum_points[i, original_transfer] - (point-baseline)
                counts[i, original_transfer] += 1
            #punish also nan teams
            else:
                counts[i, transfer_ind[i]] += 1

    return points, prices, evaluated_transfers, sum_points, counts



predictions = np.array(predictions)
probabilities = np.array(point_diff.copy())
#get baseline
no_transfers = []
for i in range(len(point_diff)):
    no_transfers.append([np.nan, np.nan])
    
baseline, _ = objective(no_transfers, free_transfers)

batch_size = 1000

best_points = 0

counts = np.ones((len(no_transfers), len(probabilities[0])), dtype='uint32')

counter = 0
best_counter = 0
identified_teams = 0
best_transfer = []

while True:
    print('Start')
    
    p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
    prob = (p.T) / np.nansum((p.T), axis=0)
    selected = np.isnan(prob)
    prob[selected] = 0
    
    #guessing part. try random combination followed up by a targeted selection
    print('Getting  teams')
    parallel_results = Parallel(n_jobs=-1)(delayed(check_random_transfers)(i) for i in range(counter, counter+6))
    print('Interpreting results')
    #store data for later
    #organize_output
    for par in parallel_results:
        if np.nanmax(par[0]) > best_points:
            best_points =  np.nanmax(par[0])
            best_transfer = par[1][np.nanargmax(par[0])]
            best_counter = counter
            
        probabilities += par[2]
        counts += par[3]
        
        counter += len(par[0])
        identified_teams += sum(~np.isnan(par[0]))
    
    
    if len(best_transfer) == 0:
        print('No acceptable teams')
        continue
    
    #print results
    price = []
        
    for gw_ind, transfer_ind in enumerate(best_transfer):
       
        transfer = transfers[transfer_ind]
        
        if not transfer == [np.nan, np.nan]:
            price.append(slim_elements_df.loc[transfer[1], 'now_cost'])
            
            if not unlimited_transfers:
                print('GW', int(1+gw_ind/3), slim_elements_df.loc[transfer[0], 'web_name'], 'for', slim_elements_df.loc[transfer[1], 'web_name'])
                print(predictions[transfer[0], :])
                print(predictions[transfer[1], :])
            else:
                print(int(gw_ind), slim_elements_df.loc[transfer[1], 'web_name'], np.round(predictions[transfer[1], :], 1),  np.round(prob[transfer_ind, gw_ind], 3))
               
                
        else:
            if unlimited_transfers:
                max_ind = np.nanargmax(p[gw_ind, :-1])
                transfer = transfers[max_ind]                    
                print(int(gw_ind), slim_elements_df.loc[transfer[0], 'web_name'], np.round(predictions[transfer[0], :], 1), np.round(prob[transfer_ind, gw_ind], 3))
                price.append(slim_elements_df.loc[transfer[0], 'now_cost'])
            
    
    print('points: ', best_points, '. diff: ', best_points-baseline, '. price: ', sum(price))
    print('\n')
    

#print original team
sort_list = np.argsort(my_players_df['points_1st_gw'])
print(my_players_df.iloc[sort_list][{'web_name', 'points_1st_gw', 'yellow_cards'}])


max_points = np.nanmax(checked_points)
max_ind = np.where(checked_points == max_points)[0]

for ind in max_ind:
    best_transfer =  checked_transfers[ind]
    for gw_ind, transfer_ind in enumerate(best_transfer):
        
        transfer = transfers[transfer_ind]
        
        
        if not transfer == [np.nan, np.nan]:
    
            print('GW', np.floor(gw_ind/3), slim_elements_df.loc[transfer[0], 'web_name'], 'for', slim_elements_df.loc[transfer[1], 'web_name'], 'yellow cards: ', slim_elements_df.loc[transfer[1], 'yellow_cards'], 'price: ', slim_elements_df.loc[transfer[1], 'yellow_cards'])
            print(predictions[transfer[0], :])
            print(predictions[transfer[1], :])
    
    print('points: ', checked_points[best_ind]-baseline)
    print('\n')



# print(picked_players[list({'web_name', 'prediction', 'now_cost', 'points_1st_gw', 'yellow_cards'})])

# print(selected_team_points)
# print(total_money - total_price)

# print_str = []
# diff_points = np.empty((0,1))
# for orig_index, orig_player in original_players.iterrows():
#     if not any(orig_player['id'] == picked_players['id']):
#         orig_position = orig_player['element_type']
#         selected = picked_players['element_type'] == orig_position
        
#         diff_1st = picked_players['points_1st_gw'][selected] - orig_player['points_1st_gw']
#         for ind, diff in enumerate(diff_1st):  
#             if not any(picked_players['web_name'][selected].iloc[ind] == original_players['web_name']):
#                 print_str.append(str(picked_players['web_name'][selected].iloc[ind] + ' for ' + orig_player['web_name'] + ': '))
#                 diff_points = np.append(diff_points, np.array(diff_1st)[ind]*100)

# sorted_list = np.argsort(diff_points)

# for ind in sorted_list:
#     print(print_str[ind] + str(diff_points[ind].astype(int)) + '/100 points')
