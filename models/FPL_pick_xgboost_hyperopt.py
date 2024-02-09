minutes_thisyear_treshold = 1
form_treshold = -1
similiarity_treshold = 0.85
points_per_game_treshold = -1



#brentford: 4, west ham: 19
exclude_team = []

#exclude player
exclude_players = ['D.D.Fofana', 'Zinchenko', 'Archer', 'Wan-Bissaka', 'Schade', 'Olise', 'Yarmoliuk', 'Doku', 'Mbeumo', 'Henry', 'Hickey', 'Zanka', 'Quansah', 'Lascelles', 'Digne', 'Ghoddos', 'D.D.Fofana']
include_players = []

include_minutes = []
do_not_exclude_players = []

rounds_to_reset = 99999
rounds_to_value = 4
#free_transfers = rounds_to_value + 1
free_transfers = 2

            
num_transfers = int((free_transfers + rounds_to_value))



skip_gw = []

force_90 = []

manual_pred = 1

# manual_blanks = {22: ['Mitoma', 'Onana', 'Son', 'Hee Chan', 'Salah', 'Kudus', 'Semenyo', 'Mbeumo', 'Sarr', 'Tomiyasu'],
#                  }
manual_blanks = {}

string = '{"picks":[{"element":597,"position":1,"selling_price":48,"multiplier":1,"purchase_price":50,"is_captain":false,"is_vice_captain":false},{"element":131,"position":2,"selling_price":50,"multiplier":1,"purchase_price":49,"is_captain":false,"is_vice_captain":false},{"element":506,"position":3,"selling_price":57,"multiplier":1,"purchase_price":55,"is_captain":false,"is_vice_captain":true},{"element":430,"position":4,"selling_price":66,"multiplier":1,"purchase_price":65,"is_captain":false,"is_vice_captain":false},{"element":396,"position":5,"selling_price":84,"multiplier":2,"purchase_price":84,"is_captain":true,"is_vice_captain":false},{"element":412,"position":6,"selling_price":59,"multiplier":1,"purchase_price":56,"is_captain":false,"is_vice_captain":false},{"element":353,"position":7,"selling_price":77,"multiplier":1,"purchase_price":75,"is_captain":false,"is_vice_captain":false},{"element":362,"position":8,"selling_price":56,"multiplier":1,"purchase_price":54,"is_captain":false,"is_vice_captain":false},{"element":8,"position":9,"selling_price":79,"multiplier":1,"purchase_price":79,"is_captain":false,"is_vice_captain":false},{"element":60,"position":10,"selling_price":85,"multiplier":1,"purchase_price":83,"is_captain":false,"is_vice_captain":false},{"element":343,"position":11,"selling_price":67,"multiplier":1,"purchase_price":66,"is_captain":false,"is_vice_captain":false},{"element":524,"position":12,"selling_price":41,"multiplier":0,"purchase_price":40,"is_captain":false,"is_vice_captain":false},{"element":29,"position":13,"selling_price":55,"multiplier":0,"purchase_price":55,"is_captain":false,"is_vice_captain":false},{"element":369,"position":14,"selling_price":54,"multiplier":0,"purchase_price":54,"is_captain":false,"is_vice_captain":false},{"element":557,"position":15,"selling_price":55,"multiplier":0,"purchase_price":55,"is_captain":false,"is_vice_captain":false}],"chips":[{"status_for_entry":"available","played_by_entry":[],"name":"wildcard","number":1,"start_event":21,"stop_event":38,"chip_type":"transfer"},{"status_for_entry":"available","played_by_entry":[],"name":"freehit","number":1,"start_event":2,"stop_event":38,"chip_type":"transfer"},{"status_for_entry":"available","played_by_entry":[],"name":"bboost","number":1,"start_event":1,"stop_event":38,"chip_type":"team"},{"status_for_entry":"available","played_by_entry":[],"name":"3xc","number":1,"start_event":1,"stop_event":38,"chip_type":"team"}],"transfers":{"cost":4,"status":"cost","limit":2,"made":0,"bank":78,"value":955}}'


#set to non-zero to override substitute price
substitute_override = 0

jump_rounds = 0

#benchboost = False
num_substitutes = 0

min_val = -8


season = '2023-24'

import requests
import pandas as pd
import numpy as np
import json
import pickle
import difflib
from datetime import datetime
from joblib import Parallel, delayed
import os
from hyperopt.pyll import scope
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, rand, SparkTrials
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate
import random


#insert string for team
directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data' + '/' + season
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
r = session.get('https://fantasy.premierleague.com/api/my-team/1454932/')
js = r.json()

js = json.loads(string)

my_players = pd.DataFrame(js['picks'])
a = json.dumps(js)
js = json.loads(a)
transfers = js["transfers"]
bank = transfers['bank']

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

#open each gw and get data for players
for gw_csv in os.listdir(directory + '/gws'):
    if gw_csv[0] == 'g':
        
        gw_path = directory + '/gws' + '/' + gw_csv
                
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

#add column if they don't exist
    
# Calculate rolling values not including the observaiton
for player in df_gw['element'].unique():
    selected_ind = df_gw['element'] == player                    
    player_df = df_gw[selected_ind]
    player_df.set_index('kickoff_time', inplace=True)
    form = player_df['total_points'].rolling('30D').mean()
    xG = player_df['expected_goals'].rolling('30D').mean()
    xA = player_df['expected_assists'].rolling('30D').mean()
    xGI = player_df['expected_goal_involvements'].rolling('30D').mean()
    xGC = player_df['expected_goals_conceded'].rolling('30D').mean()
    xP = player_df['xP'].rolling('30D').mean()
    points_per_game =  player_df['total_points'].cumsum()/ (player_df['round'])
    ict = player_df['ict_index'].rolling('30D').mean()
    influence = player_df['influence'].rolling('30D').mean()
    threat = player_df['threat'].rolling('30D').mean()
    creativity = player_df['creativity'].rolling('30D').mean()
    bps = player_df['bps'].rolling('30D').mean()
    minutes = player_df['minutes'].rolling('30D').mean().values
    
    name_ind = np.where(slim_elements_df['id'] == player)[0][-1]
    #average of two last matches
    minutes_2g = np.mean(player_df['minutes'][-1:])
    if minutes_2g > 60 or slim_elements_df.iloc[name_ind]['web_name'] in force_90:
        selected = player_df['minutes'] > 0
        use_minutes = np.max([np.mean(player_df['minutes'][selected]), player_df['minutes'].rolling('30D').mean().values[-1]])
        minutes[-1] = use_minutes

        if slim_elements_df.iloc[name_ind]['web_name'] in force_90:
            print(player, slim_elements_df.iloc[name_ind]['web_name'], use_minutes)
        
        
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
    df_gw.loc[selected_ind, 'running_xG'] = xG.values
    df_gw.loc[selected_ind, 'running_xA'] = xA.values
    df_gw.loc[selected_ind, 'running_xGI'] = xGI.values
    df_gw.loc[selected_ind, 'running_xGC'] = xGC.values    
    df_gw.loc[selected_ind, 'running_xP'] = xP.values   
    df_gw.loc[selected_ind, 'points_per_game'] = points_per_game.values
    df_gw.loc[selected_ind, 'points_per_played_game'] = result[:-1]
    df_gw.loc[selected_ind, 'running_minutes'] = minutes
    
if rounds_to_value > rounds_to_reset:
    rounds_to_value == rounds_to_reset

i=0

 
while pd.to_datetime(events_df.deadline_time[i], format='%Y-%m-%dT%H:%M:%SZ') < datetime.now():
    i = i + 1

current_gameweek = i + 1
print('Current gameweek: ' + str(current_gameweek))

print('previous')
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
        
print('future')
#get statistics for the x next gameweeks
df_future_games = pd.DataFrame(columns=['gameweek', 'team_h', 'team_a', 'difficulty_diff'])
for i in range(jump_rounds, rounds_to_value+1+jump_rounds):
    this_gw = i + current_gameweek
    
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
predicted_values = np.zeros((slim_elements_df.shape[0], rounds_to_value+1))
predicted_values_1st_gw = np.zeros((slim_elements_df.shape[0], 1))

#calculate value and add to database
total_points = slim_elements_df['total_points'].astype(float)
minutes_played = slim_elements_df['minutes'].astype(float)
form = slim_elements_df['form']

selected = points_per_game == 0
points_per_game[selected] = 0.1
games_played = np.round(total_points / points_per_game)

# names = slim_elements_df['web_name']
# for name in include_minutes:
#     ind = name == names
#     minutes_played[ind] = minutes_thisyear_treshold

selected_players = (form < form_treshold) | (minutes_played < minutes_thisyear_treshold) | (points_per_game < points_per_game_treshold)

for name in do_not_exclude_players:
    ind = name == names
    selected_players[ind] = False

# points_per_game[selected_players] = 0

with open(r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\model.sav', 'rb') as f:
    result = pickle.load(f)
    
predictions = []


#remember to control order of features in df_predict!!!
keep_features = ['running_minutes', 'transfer_in', 'transfer_out', 'running_ict', 'running_influence', 'running_threat', 'running_creativity', 'running_bps', 'string_opp_team', 'string_team', 'names', 'element_type', 'was_home', 'running_xP', 'running_xG', 'running_xA', 'running_xGI', 'running_xGC', 'form',
                 'points_per_game', 'points_per_played_game', 'other_difficulty', 'own_difficulty']

    
#predict future points
for df_name in slim_elements_df.iterrows():
    
    # if df_name[1].second_name == 'Gordon':
    #     print(df_name)
        
    # df_name = [564, slim_elements_df.iloc[564]]

    if not selected_players[df_name[0]]:
        team = int(df_name[1].team)
        position = int(df_name[1].element_type)     
        first_name = df_name[1].first_name
        second_name = df_name[1].second_name
        name = first_name + second_name
        form = df_name[1].form
        player_id =  df_name[1].id
        
        url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id)
        r = requests.get(url)
        player = r.json()
        
        player_games = pd.DataFrame(player['history'])
        
        player_games['kickoff_time'] =  pd.to_datetime(player_games['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        player_games = player_games.sort_values(by='kickoff_time')
        
        fixtures = pd.DataFrame(player['fixtures'])
        fixtures['kickoff_time'] =  pd.to_datetime(fixtures['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        fixtures = fixtures.sort_values(by='kickoff_time')        
        player_games.set_index('kickoff_time', inplace=True)
        
        
        if sum(df_gw.element == player_id) == 0:
            
            print('Estimate values for ', elements_df.iloc[selected_ind].web_name)

            xP = np.nan
            
            selected_ind = np.where(elements_df.id == player_id)[0][-1]
            xG = float(elements_df.iloc[selected_ind].expected_goals)
            xA = float(elements_df.iloc[selected_ind].expected_assists)
            xGI = float(elements_df.iloc[selected_ind].expected_goal_involvements)
            xGC = float(elements_df.iloc[selected_ind].expected_goals_conceded)
            ict = float(elements_df.iloc[selected_ind].ict_index)
            influence = float(elements_df.iloc[selected_ind].influence)
            threat = float(elements_df.iloc[selected_ind].threat)
            creativity = float(elements_df.iloc[selected_ind].creativity)
            bps = float(elements_df.iloc[selected_ind].bps)
            points_per_played_game = float(elements_df.iloc[selected_ind].points_per_game)            
            minutes = np.min([float(elements_df.iloc[selected_ind].minutes), 90])
            form = float(elements_df.iloc[selected_ind].form)
            points_per_game = float(elements_df.iloc[selected_ind].points_per_game)
            
        else:
                
            selected_ind = np.where(df_gw.element == player_id)[0][-1]
            
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
        transfer_in = elements_df.iloc[selected_ind].transfers_in_event
        transfer_out = elements_df.iloc[selected_ind].transfers_out_event
                
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
        pred_score = np.zeros(rounds_to_value+1)
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
            
                
            df_predict = pd.DataFrame(columns=keep_features)
            
            if gw_idx == 0:
                df_predict.loc[0] = [minutes, transfer_in, transfer_out, ict, influence, threat, creativity, bps, string_opp_team, string_team, name,
                                                   position, home, xP, xG*xG_multiplier, xA*3, xGI, xGC*xGC_multiplier, form,
                                                   points_per_game, points_per_played_game, other_difficulty, own_difficulty]
            else:
                df_predict.loc[0] = [minutes, np.nan, np.nan, ict, influence, threat, creativity, bps, string_opp_team, string_team, name,
                                                   position, home, xP, xG*xG_multiplier, xA*3, xGI, xGC*xGC_multiplier, form,
                                                   points_per_game, points_per_played_game, other_difficulty, own_difficulty]
      
                     
            df_predict['element_type'] = df_predict['element_type'].astype('category')
            df_predict['names'] = df_predict['names'].astype('category')
            df_predict['other_difficulty'] = df_predict['other_difficulty'].astype('category')
            df_predict['own_difficulty'] = df_predict['own_difficulty'].astype('category')
            df_predict['string_team'] = df_predict['string_team'].astype('category')
            df_predict['string_opp_team'] = df_predict['string_opp_team'].astype('category')

            #estimated = 10**result.predict(df_predict) + min_val  
            estimated = result.predict(df_predict)           
            #remove if unlikely to play
            if game_idx+jump_rounds == 0 and df_name[1]['chance_of_playing_next_round'] < 75:
                estimated = 0
                
            if string_team in exclude_team:
                estimated = 0
                
            if gw in manual_blanks.keys():
                if df_name[1]['web_name'] in manual_blanks[gw]:
                    estimated = 0
                               
            pred_score[gw_idx] = pred_score[gw_idx] + estimated
            total_matches = total_matches + 1

            
        first_gw = pred_score[0]

        #predicted_points = pred_score/total_matches - (4 / rounds_to_reset)
        predicted_points = pred_score  #- (4 / rounds_to_reset)
        predicted_values[df_name[0]] = predicted_points
        predicted_values_1st_gw[df_name[0]] = first_gw
        predictions.append(pred_score)
        
    else:
        predictions.append(np.zeros(rounds_to_value+1).astype(float))


slim_elements_df['points_1st_gw'] = predicted_values_1st_gw

#set what to use for evaluation. can be points_per_game
prediction = np.copy(predicted_values)

#save for later
original_prediction = np.copy(predictions)

    
# remove (=set to zero) low form
selected = slim_elements_df['form'] < form_treshold
prediction[selected] = 0
for ind in np.where(selected)[0]:
    predictions[ind]=np.zeros(rounds_to_value+1).astype(float)

# for player in include_players:
#     selected = slim_elements_df['web_name'] == player
#     prediction[selected] = manual_pred
#     for ind in np.where(selected)[0]:
#         predictions[ind]=np.zeros(rounds_to_value+1).astype(float) + manual_pred


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
def find_team_points(team_positions, gw_prediction):
    
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

#outcome is all possible transfers

predictions = np.stack(slim_elements_df.prediction)
preds = []

for player_out in slim_elements_df.iterrows():
    #check if picked
    if player_out[1]['picked']:
        
        for player_in in slim_elements_df.iterrows():
            
            #check if not picked, not same the other player, any predictions >0 and same element
            if not player_in[1]['picked'] and sum(player_in[1].prediction) > 0 and  player_in[1].element_type == player_out[1].element_type:
                
                preds.append(predictions[player_in[0]] - predictions[player_out[0]])
 


preds = np.array(preds)

#set all positive
preds = preds - np.min(preds, axis=0)

if free_transfers == 2:
    no_main_gain = 0
else:
    no_main_gain = 4   
    
sum_pred_main = np.sum(preds, axis=0) + no_main_gain


hit_ratio = 0.5**(1/5)
no_hit_gain = (np.sum(preds, axis=0)*hit_ratio)/(1-hit_ratio)
sum_pred_hit = np.sum(preds, axis=0) + no_hit_gain
    
    

gws_transfers = []
for i in range(5):
    gw_transfers_main = []
    gw_transfers_hit = []
    
    
    ind = 0
    for player_out in slim_elements_df.iterrows():
        #check if picked
        if player_out[1]['picked']:
            
            for player_in in slim_elements_df.iterrows():
                
                #check if not picked, not same the other player, any predictions >0 and same element
                if not player_in[1]['picked'] and sum(player_in[1].prediction) > 0 and  player_in[1].element_type == player_out[1].element_type:
                    
                    gw_transfers_main.append((preds[ind, i]/sum_pred_main[i], [player_out[0], player_in[0]]))
                    gw_transfers_hit.append((preds[ind, i]/sum_pred_hit[i], [player_out[0], player_in[0]]))
                    
                    ind += 1
                    
    gws_transfers.append([(no_main_gain/sum_pred_main[i], [np.nan, np.nan])] + gw_transfers_main) 
    gws_transfers.append([(no_hit_gain[i]/sum_pred_hit[i], [np.nan, np.nan])] + gw_transfers_hit)
    

def objective(inputs):
    
    space = inputs[0]
    team = inputs[1].copy()
    
    #loop through the transfers and check if they are possible
    
    #identify which gw each transfer belongs to
    gws = []
    for key, transfer in space.items():
        gws.append(int(key[0]))
        
    #loop the gameweeks        
    for gw in np.unique(gws):
        
        for key, transfer in space.items():
            
            if int(key[0]) == gw and not np.isnan(transfer[0]):
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
                
            return {'loss': np.inf, 'status': STATUS_OK }
    
        
    team = inputs[1].copy()
    
    team_points = []
    
    for gw in np.unique(gws):
        
        #loop and do the transfers
        for key, transfer in space.items():
            
            if int(key[0]) == gw and not np.isnan(transfer[0]):
                team[transfer[0]] = False
                team[transfer[1]] = True
                
                #if hit
                if int(key[1]) == 1:
                    team_points.append(-4)
            
        
        gw_prediction = predictions[team, gw]
        team_positions = slim_elements_df.loc[team, 'element_type'].values
        
        team_points.append(find_team_points(team_positions, gw_prediction))
        
    print(sum(team_points))
            
    return {'loss': -sum(team_points), 'status': STATUS_OK }


space = {'00': hp.pchoice("00", gws_transfers[0]),
         '01': hp.pchoice("01", gws_transfers[1]),
         '10': hp.pchoice("10", gws_transfers[2]),
         '11': hp.pchoice("11", gws_transfers[3]),
         '20': hp.pchoice("20", gws_transfers[4]),
         '21': hp.pchoice("21", gws_transfers[5]),
         '30': hp.pchoice("30", gws_transfers[6]),
         '31': hp.pchoice("31", gws_transfers[7]),
         '40': hp.pchoice("40", gws_transfers[8]),
         '41': hp.pchoice("41", gws_transfers[9]),
    }
#initiate by testing no transfers
trials = generate_trials_to_calculate([{'00': 0, '01': 0,  '10': 0,  '11': 0,  '20': 0, '21': 0,  '30': 0,  '31': 0, '40': 0, '41': 0}])


team = slim_elements_df['picked'].values.copy()


batch_size = 100
max_evals = 5000000

for i in range(len(trials.trials)+batch_size, max_evals + 1, batch_size):
    best_transfers = fmin(fn = objective,
                    space = [space, team],
                    algo = tpe.suggest,
                    max_evals = i,
                    trials = trials)
    
    filename = r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\transfers.pkl'
    pickle.dump(trials, open(filename, "wb"))
    
    for transfer_list_ind, (gw, transfer_ind) in enumerate(best_transfers.items()):
        
        transfer_list = gws_transfers[int(np.floor(transfer_list_ind/2))]
        transfer = transfer_list[transfer_ind][1]
        if not np.isnan(transfer[0]):
            print('GW', gw, slim_elements_df.loc[transfer[1], 'web_name'], 'for', slim_elements_df.loc[transfer[0], 'web_name'])
        
        
        

# #add variable to see when a player was added to the team
# slim_elements_df['iterations'] = 0

# #get those that we have picked not including substitutes.


# check_again = True

# total_price = 1

# def delta_points_per_price(slim_elements_df, my_player, include_players, exclude_players, num_team, total_money, total_price, num_transfers):
    
#     #if picked_continue
#     web_name_my_player = slim_elements_df.loc[my_player, 'web_name']
    
#     #trade out
#     slim_elements_df.loc[my_player, 'picked'] = False
    
#     cost_my_player = slim_elements_df.loc[my_player, 'now_cost']
    
#     prediction_my_player = slim_elements_df.loc[my_player, 'prediction'] 
    
#     position_my_player = slim_elements_df.loc[my_player, 'element_type'] 
    
#     team_my_player = slim_elements_df.loc[my_player, 'team'] 
    
#     cost_my_player = slim_elements_df.loc[my_player, 'now_cost']
    
#     # initiate matrix that will be used to get most valuable transfer. 11 players and the price per point increase for all possible transfers of that player. set all to zeros to begin with.
#     delta_price = np.zeros(slim_elements_df.shape[0]) + np.inf
    
#     for putative_player in range(slim_elements_df.shape[0]):
        
#         #if so do ot allow original player to be transfered out for a new player
#         if sum(slim_elements_df['picked'] != slim_elements_df['original_player'])/2 >= num_transfers:
#             if slim_elements_df.loc[my_player, 'original_player']:
#                 if not slim_elements_df.loc[putative_player, 'original_player']:
#                     continue
                
#             if not slim_elements_df.loc[my_player, 'original_player']:
#                 if slim_elements_df.loc[putative_player, 'original_player']:
#                     continue
                    
        
#         #if not already picked and same category
#         if not slim_elements_df.loc[putative_player, 'picked'] and position_my_player == slim_elements_df.loc[putative_player, 'element_type']:
            
#             prediction_putative_player = slim_elements_df.loc[putative_player, 'prediction'] 
            
#             team_putative_player = slim_elements_df.loc[putative_player, 'team'] 
            
#             cost_putative_player = slim_elements_df.loc[putative_player, 'now_cost'] 
            
#             web_name_putative_player = slim_elements_df.loc[putative_player, 'web_name']
                               
#             #if any increasing points or team is full and we have money
#             if any(prediction_putative_player > prediction_my_player) and (num_team[team_putative_player-1]<3 or team_putative_player==team_my_player) and (total_money-total_price+cost_my_player-cost_putative_player) >= 0 and web_name_putative_player not in exclude_players:
                
#                 #trade in
#                 slim_elements_df.loc[putative_player, 'picked'] = True
                
#                 potential_team_points = find_team_points(slim_elements_df[slim_elements_df['picked'] == True])
                
#                 #trade out 
#                 slim_elements_df.loc[putative_player, 'picked'] = False
                
#                 cost_potential_player = slim_elements_df.loc[putative_player, 'now_cost']
#                 #calculate price per point increase for each possible transfer of this player
#                 if web_name_putative_player in include_players or web_name_my_player in exclude_players:
#                     delta_price[putative_player] = -1000
#                 elif potential_team_points == selected_team_points and  cost_potential_player < cost_my_player:
#                     delta_price[putative_player] = (cost_potential_player - cost_my_player)
#                 elif potential_team_points > selected_team_points:
#                     delta_price[putative_player] = (cost_potential_player - cost_my_player)/(potential_team_points - selected_team_points)*100000
    
#     return delta_price

# #loop until there are no more improvement (delta_points = 0) or that we cannot afford any changes
# while check_again and total_price < total_money:
    
#     picked_players = slim_elements_df[slim_elements_df['picked'] == True]
#     selected_team_points =  find_team_points(picked_players)
           
#     total_price = sum(picked_players['now_cost'])
    
#     print('\n', int(sum(slim_elements_df['iterations'])/15), total_money-total_price, selected_team_points)
    
#     #testable players are my players and is not in include players (probably to save time)
#     testable_players = np.where(slim_elements_df['picked'] & ~slim_elements_df['web_name'].isin(include_players))[0]

#     #fill out delta points. loop through all of my players
#     results = Parallel(n_jobs=-1)(delayed(delta_points_per_price)(slim_elements_df, my_player, include_players, exclude_players, num_team, total_money, total_price, num_transfers) for my_player in testable_players)
    
#     # initiate matrix that will be used to get most valuable transfer. 11 players and the price per point increase for all possible transfers of that player. set all to zeros to begin with.
#     delta_price = np.zeros([slim_elements_df.shape[0], slim_elements_df.shape[0]]) + np.inf
    
#     #organize output
    
#     #columns are my players and rows are possible transfers in
#     for ind, my_player in enumerate(testable_players):
#         # initiate matrix that will be used to get most valuable transfer. 11 players and the price per point increase for all possible transfers of that player. set all to zeros to begin with.
#         delta_price[:, my_player] = results[ind]
    
            
#     #continue if there are any transfers that will improve points return
#     if np.min(delta_price) < np.inf:
#         #find index of max valuable transfer
        
#         #if no more transfers. do not allow transfer out of original players. divide by two since the algo picks both diff players out and in
#         if sum(slim_elements_df['picked'] != slim_elements_df['original_player'])/2 >= num_transfers:
#             print('no more transfers:', sum(slim_elements_df['picked'] != slim_elements_df['original_player'])/2)
#             # mesh_grid = np.ix_(~slim_elements_df['original_player'], slim_elements_df['original_player'])
#             # delta_price[mesh_grid] = np.inf
            
#             # max_single = np.where(delta_price  == np.min(delta_price))
            
#             # own_out_mesh = np.ix_(~slim_elements_df[~'original_player'], slim_elements_df['original_player'])
#             # own_in_mesh = np.ix_(~slim_elements_df['original_player'], ~slim_elements_df['original_player'])
            
#             # #these two must be combined
#             # min_double1 = np.min(delta_price[own_out_mesh]) + np.max(delta_price[own_out_mesh])
            
#         else:
#             #single transfer
#             print('more transfers:', sum(slim_elements_df['picked'] != slim_elements_df['original_player'])/2)
        
        
#         max_index = np.where(delta_price  == np.min(delta_price))

#         #exchange players (row is player in, coloumn is player out)
#         index_player_out = max_index[1][0]
#         index_player_in = max_index[0][0]

#         team_out = slim_elements_df['team'].iloc[index_player_out]
#         position_out = slim_elements_df['element_type'].iloc[index_player_out]

#         team_in = slim_elements_df['team'].iloc[index_player_in]
#         position_in = slim_elements_df['element_type'].iloc[index_player_in]

#         #set player as (not) selected
#         slim_elements_df.loc[index_player_in, 'picked'] = True
#         slim_elements_df.loc[index_player_out, 'picked'] = False

#         #increase/decrease selected position
#         num_position[position_in - 1] = num_position[position_in - 1] + 1
#         num_position[position_out - 1] = num_position[position_out - 1] - 1

#         #increase/decrease selected team
#         num_team[team_in - 1] = num_team[team_in - 1] + 1
#         num_team[team_out - 1] = num_team[team_out - 1] - 1

#         # add to see how many iterations a player is in team
#         slim_elements_df.loc[index_player_in, 'iterations'] = 0
#         slim_elements_df.loc[slim_elements_df['picked'], 'iterations'] = slim_elements_df.loc[slim_elements_df['picked'], 'iterations'] + 1
            
#         print(slim_elements_df.loc[index_player_in, 'web_name'] + ' for ' + slim_elements_df.loc[index_player_out, 'web_name'])
#     else:
#         check_again = False


# slim_elements_df['prediction'] = np.sum(original_prediction, axis=1)
# my_players_df['prediction'] = my_players_df['prediction'].apply(sum)
# picked_players = slim_elements_df[slim_elements_df['picked'] == True]

# #print original team
# sort_list = np.argsort(my_players_df['points_1st_gw'])
# print(my_players_df.iloc[sort_list][{'web_name', 'points_1st_gw', 'prediction', 'yellow_cards'}])

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
