my_players = [
    {'web_name': 'A.Becker', 'selling_price': 55, 'element_type': 1},
    {'web_name': 'Dúbravka', 'selling_price': 40, 'element_type': 1},
    
    {'web_name': 'Doherty', 'selling_price': 45, 'element_type': 2},
    {'web_name': 'Tosin', 'selling_price': 45, 'element_type': 2},
    {'web_name': 'Chalobah', 'selling_price': 50, 'element_type': 2},
    {'web_name': 'Virgil', 'selling_price': 60, 'element_type': 2},
    {'web_name': 'Trippier', 'selling_price': 50, 'element_type': 2},
    
    {'web_name': 'Kudus', 'selling_price': 66, 'element_type': 3},
    {'web_name': 'Tielemans', 'selling_price': 59, 'element_type': 3},
    {'web_name': 'M.Salah', 'selling_price': 145, 'element_type': 3},
    {'web_name': 'Barnes', 'selling_price': 65, 'element_type': 3},
    {'web_name': 'Sarr', 'selling_price': 65, 'element_type': 3},
    
    {'web_name': 'Watkins', 'selling_price': 89, 'element_type': 4},
    {'web_name': 'Thiago', 'selling_price': 60, 'element_type': 4},
    {'web_name': 'Mateta', 'selling_price': 75, 'element_type': 4},
]


bank = 29
free_transfers = 1
save_transfers_for_later = 2 #transfers left at end of last round


minutes_thisyear_treshold = 60
form_treshold = -1
points_per_game_treshold = -1
running_minutes_threshold = -1

#
exclude_team = []

exclude_players = ['Foden', 'Cunha', 'Harrison', 'Doherty', 'Saka', 'J.Palhinha', 'Trafford', 'Mac Allister', 'L.Miley', 'Mykolenko', 'Wilson', 'Nmecha', 'Botman', 'Osula', 'M.Bizot',  'Raúl',  'Maatsen', 'Richarlison',  'Doku', 'Welbeck', 'Devenny', 'Acheampong', 'Aït-Nouri', 'Martinelli', 'Aké', 'Marmoush', 'Saliba']
include_players = []
#tarkowski
do_not_exclude_players = []



do_not_transfer_out = []
rounds_to_value = 5
#transfer to evaluate per week
trans_per_week = 3

jump_rounds = 0
#if you also want to evaluate players on the bench. in case of uncertain starters.
number_players_eval = 11

wildcard = False
skip_gw = [66]

benchboost_gw = 77
tripple_captain_gw = 100


forward_price_limit = -1 #in millions
midfield_price_limit = -1


#assistant manager in 2024-25 season
assistant_manager_gw = 100
assistant_manager_team = 'CRY'
assistant_manager_price = 0.8 #in millions


force_90 = []

manual_pred = 1

#players
manual_blanks = { 5: ['Gordon', 'Sarr', 'Perri'], 6: ['Darlow']} #nothing:  Spence for Burn, Marmoush for Wood. Isak: Isak for Wood. Robertson: Robertson for Kayode. Robertson and Isak: Robertson for Burn, Isak for Wood. All three: wood, schar and burn


#GW               
manual_blank = {}
manual_double = {}


season = '2025-26'
previous_season = '2024-25'

skip_free_hit_calc = False


import requests
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import random
import xgboost as xgb
import time

from pandas.api.types import CategoricalDtype


#insert string for team
directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data' + '/' + season
prev_season_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data' + '/' + previous_season
team_path = directory + "/teams.csv"
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"

try:
    df_teams = pd.read_csv(team_path)

except:
    #insert string for team
    directory = r'C:\Users\jorgels\Documents\GitHub\Fantasy-Premier-League\data' + '/' + season
    prev_season_directory = r'C:\Users\jorgels\Documents\GitHub\Fantasy-Premier-League\data' + '/' + previous_season
    team_path = directory + "/teams.csv"
    

    df_teams = pd.read_csv(team_path)


string_names = df_teams['short_name'].values


am_num_team = np.where(string_names == assistant_manager_team)[0][0]

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
benchboost = []
tripple_captain = []
assistant_manager = []
free_hit = []
for i in range(jump_rounds, rounds_to_value+jump_rounds):
    this_gw = i + current_gameweek
    
    if this_gw in skip_gw:
        free_hit.append(True)
        skip_free_hit_calc = True
    else:
        print(this_gw)
        free_hit.append(False)

    if benchboost_gw == this_gw:
        benchboost.append(True)
    else:
        benchboost.append(False)
        
    if tripple_captain_gw == this_gw:
        tripple_captain.append(True)
    else:
        tripple_captain.append(False)
        
    if assistant_manager_gw >= this_gw-2 and  assistant_manager_gw <= this_gw:
        assistant_manager.append(True)
    else:
        assistant_manager.append(False)

    # if any(np.array(skip_gw) == this_gw):
    #     continue

    

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
    ind = slim_elements_df['team'] == club
    selected_players.loc[ind] = True  

for name in do_not_exclude_players:
    ind = slim_elements_df['web_name'] == name
    selected_players.loc[ind] = False

# points_per_game[selected_players] = 0

with open(model_path, 'rb') as f:
    summary = pickle.load(f)
    
predictions = []

result = summary["model"]
hyperparamaters = summary["hyperparameters"]
temporal_window = int(hyperparamaters["temporal_window"])

train_X = summary["train_features"]
#all_rows = summary["all_rows"]

with open(r'\\platon.uio.no\med-imb-u1\jorgels\model_data.pkl', 'rb') as file:
    all_rows = pickle.load(file) 

#min_y = np.min(train_X['0total_points'])

predictions = []


#add nan categories
dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
       'other_difficulty'] #'difficulty',

int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon', 'SoT']

float_variables = ['transfers_in', 'transfers_out', 'threat']

#features that I don't have access to in advance.
#opp_team_points included because it already calculate in model
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
       'total_points', 'expected_goals', 'expected_assists',
       'expected_goal_assists', 'expected_goals_conceded', 'own_team_points', 'own_element_points']
       #'points_per_game', 'points_per_played_game']

temporal_single_features = ['points_per_game', 'points_per_played_game']


#total_points, minutes, kickoff time not for prediction
fixed_features = ['element_type', 'string_team', 'season', 'name']

dynamic_features = ['string_opp_team', 'transfers_in', 'transfers_out',
       'was_home', 'own_difficulty', 'other_difficulty']#, 'difficulty']




#free hit
keep_ind = []
if rounds_to_value == 1 and wildcard:
    for el in [1, 2, 3, 4]:
        selected = slim_elements_df.element_type == el
        min_keeper_price = np.min(slim_elements_df.loc[selected, 'now_cost'])
        keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0][0])
        
        if len(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0]) > 1 and el > 1:  
            keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0][1])
    
else:
    selected = slim_elements_df.element_type == 1
    min_keeper_price = np.min(slim_elements_df.loc[selected, 'now_cost'])
    
    keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == 1))[0][0])
              
    
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
        element_type = df_name[1].element_type  

        url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id)
        downloaded = False
        while not downloaded:
            try:
                r = requests.get(url)
                player = r.json()
                downloaded = True
            except:
                time.sleep(30)

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

        for game in gws.iterrows():

            #add empty row
            new_row = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in predicting_df.dtypes.items()})

            #add fixed
            new_row.loc[0, fixed_features] = predicting_df[fixed_features].iloc[-1]

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

            new_row['difficulty'] = diff_difficulty[game_idx]
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
        
        if temporal_window > 0:
            for k in range(int(temporal_window)):
    
    
                temporal_names = [str(k) + s for s in temporal_features]
                dynamic_names = [str(k) + s for s in dynamic_features]
    
                # Create an empty DataFrame with the specified columns
                if k==0:
                    temporal_single_names = [str(k) + s for s in temporal_single_features]
                    col_names = temporal_names + dynamic_names + temporal_single_names
    
                else:
                    col_names = temporal_names + dynamic_names
    
                temp_train = pd.DataFrame(index=predicting_df.index, columns=col_names)
    
                temporal_data = predicting_df[temporal_features].shift(k+1)
                dynamic_data = predicting_df[dynamic_features].shift(k)
    
                temp_train[temporal_names] = temporal_data.values
                temp_train[dynamic_names] = dynamic_data.values
    
                if k==0:
                    temporal_single_data = predicting_df[temporal_single_features].shift(k+1)
                    temp_train[temporal_single_names] = temporal_single_data.values
    
    
                #set dtype
                for col in temp_train.columns:
    
                    col_stem = ''.join([char for char in col if not char.isdigit()])
    
                    if col_stem in dynamic_categorical_variables:
                        temp_train[col] = temp_train[col].astype('category')
                    elif col_stem in int_variables:
                        temp_train[col] = temp_train[col].astype('Int64')
                    elif col_stem in temporal_features or col_stem in float_variables or col_stem in temporal_single_features:
                        temp_train[col] = temp_train[col].astype('float')
                    else:
                        print('CHECK', col)
    
                predicting_df = pd.concat([predicting_df, temp_train], axis=1)
                
                
                
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
                    full_ooop[-len(opponents_of_opponents_points):] = opponents_of_opponents_points
    
                temp_train.loc[index, opponent_point_names] = full_ooop[::-1]
                
                
                opp_elem_selected =  opp_selected & (all_rows['element_type'] == element_type)
                      
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
                    
                    
                    
                        
            predicting_df = pd.concat([predicting_df, temp_train], axis=1)   
            
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
                    
                    print(val_values[~mask] + ': does not exist in training data. Set to nan')




        #prediciting one by one:
        for game in gws.iterrows():

            game_idx = game[0]
            gw_idx = int(game[1].gameweek_ind)
            
            gw = game[1].gameweek

            Dgame = xgb.DMatrix(data=predicting_df.iloc[[game_idx]], enable_categorical=True)

            estimated = result.predict(Dgame)[0]
            
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
            
            
            #keep a budget keeper
            if df_name[0] in keep_ind:
                if estimated < 0.1:
                    estimated = 0.1
                    
                if game_idx == 0:
                    print('Including because of low price:', df_name[1].web_name)
                    
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
    
                if sum(all_rows.name == name) == 0 and (game_idx == 0):
                    if should_have_trainingdata:
                        print(name + ': does not exist in training data. Shoul dbe predicted without name')
                    #estimated = 0
                elif game_idx == 0:
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
            #a=1
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

        if not unlimited_transfers:

            #if all pred is zero skip week (=free hit)
            if sum(predictions[:, gw]) == 0:
                estimated_points = 0

                all_points.append(0)
            else:
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
                        free_transfers -=1
                    
                        #pay if negative
                        if free_transfers < 0:
                            team_points.append(-transfer_cost)
                            free_transfers += 1
                    
                    #ceil the possible number of transfers. 4 since we add one before next round
                    if free_transfers > 4:
                        free_transfers = 4

                gw_prediction = predictions[team, gw]
                team_positions = slim_elements_df.loc[team, 'element_type'].values

                estimated_points = find_team_points(team_positions, gw_prediction, benchboost[gw], tripple_captain[gw])
                
                captain_bonus = np.max(gw_prediction)
                
                all_points.append(np.sum(gw_prediction)+captain_bonus)

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
        deduct_points = deduct_transfers*-transfer_cost
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
    parallel_results = Parallel(n_jobs=-1, backend='threading')(delayed(check_random_transfers)(i, True, free_transfers) for i in range(counter, counter+6))
    
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
    
    for ind, k in enumerate(best_transfer):
        trans = transfers[k]
        if np.isnan(trans[1]):
            print(slim_elements_df.loc[selected, 'web_name'].iloc[ind], slim_elements_df.loc[selected, 'element_type'].iloc[ind])
        else:
            print(slim_elements_df.iloc[trans[1]]['web_name'], slim_elements_df.iloc[trans[1]]['element_type'])
            
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

#check current team
baseline_point, baseline_price, baseline_all_point = objective(no_transfers, unlimited_transfers, free_transfers)


best_points = sum(baseline_point)
best_all_points = sum(baseline_all_point)
best_price = baseline_price
counts = np.ones((len(no_transfers), len(probabilities[0])), dtype='uint32')
best_transfer = [len(transfers)-1 for _ in range(15)]

best_pitch = baseline_point.copy()
best_bench = [a - b for a, b in zip(baseline_all_point, baseline_point)]

#all_evaluated_transfers = [no_transfers]

p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
prob = (p.T) / np.nansum((p.T), axis=0)
selected = np.isnan(prob)
prob[selected] = 0

check_guided = True
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
            


counter = 0
best_counter = 0
old_num_teams = 0


if rounds_to_value == 1:
    batch_size = 1
else:
    batch_size = 100000

import time

while True:

    all_evaluated_transfers = []

    if counter > 0:
        print('Start')
    
        p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
        prob = (p.T) / np.nansum((p.T), axis=0)
        selected = np.isnan(prob)
        prob[selected] = 0
    
        #guessing part. try random combination followed up by a targeted selection
        print('Getting  teams')
        t1_start = time.time()
        parallel_results = Parallel(n_jobs=-1, backend='threading')(delayed(check_random_transfers)(i, unlimited_transfers, free_transfers) for i in range(counter, counter+6))
        t1_stop = time.time()
        print("Elapsed time:", t1_stop - t1_start)
        print('Interpreting results')
    
    
        #store data for later
        #organize_output
        
        len_par = 0
                    
                    
        for par in parallel_results:
            
            for k in par[0]:
                if len(k) > len_par:
                    len_par = len(k)
                    print(len_par)
                    
                    
            #to get the last most positive
            sum_points = [np.sum(inner_list) for inner_list in par[0]]
            
            max_points = np.nanmax(sum_points)
            
            max_indices = np.where(sum_points == max_points)[0]
            
            #loop_those in reverse order (because of delayed transfer)
            for ind_max in max_indices[::-1]:
                if max_points > best_points or (max_points == best_points and par[1][ind_max] < best_price) or (max_points == best_points and par[1][ind_max] == best_price and sum(par[2][ind_max]) >  best_all_points):
                    best_points =  max_points
                    best_transfer = par[3][ind_max]
                    best_counter = counter
                    best_price = par[1][ind_max]
                    best_all_points = sum(par[2][ind_max])
                    
                    best_pitch = par[0][ind_max].copy()
                    best_bench = [a - b for a, b in zip(par[2][ind_max], best_pitch)]
                
    
            all_evaluated_transfers = all_evaluated_transfers + par[3]
            
            #the first prob of each week is different than the others
            k = 0
            for w in range(rounds_to_value):
                for t_res in range(trans_per_week):
                    index = w*3 + t_res
    
                    probabilities[index, :] += par[4][index, :]
                    counts[index, :] += par[5][index, :]


        counter += len(par[0])

    # print('Checked', len(all_evaluated_transfers)-old_num_teams, 'teams')
    # old_num_teams = len(all_evaluated_transfers)

    # # Convert each list to a tuple
    # unique_tuples = set(tuple(x) for x in all_evaluated_transfers)
    # # Convert the tuples back to lists
    # all_evaluated_transfers = [list(x) for x in unique_tuples]

    # print(len(all_evaluated_transfers), 'unique teams')

    counter = 1
    
    if len(best_transfer) == 0:
        print('No acceptable teams')
        continue

    #print results
    price = []
    last_gw = 0

    for gw_ind, transfer_ind in enumerate(best_transfer):

        transfer = transfers[transfer_ind]
        
        gw = int(1+gw_ind/trans_per_week)
        
        if not gw == last_gw:
            #print('\n')
            print('GW', gw, np.round(free_hit_points[gw-1] - best_pitch[gw-1], decimals=1), np.round(best_bench[gw-1], decimals=1))
            last_gw = gw
            if np.round(best_bench[gw-1], decimals=1) < 0:
                a = hfhfhfff

        if not transfer == [np.nan, np.nan]:
            price.append(slim_elements_df.loc[transfer[1], 'now_cost'])

            if not unlimited_transfers:
                print( slim_elements_df.loc[transfer[0], 'web_name'], 'for', slim_elements_df.loc[transfer[1], 'web_name'], np.round(prob[transfer_ind, gw_ind], 4))
                print( np.round(predictions[transfer[0], :], decimals=1))
                print( np.round(predictions[transfer[1], :], decimals=1))
                #print(prob[transfer_ind, gw_ind])
            else:
                print(int(gw_ind), slim_elements_df.loc[transfer[1], 'web_name'], np.round(predictions[transfer[1], :], 1),  np.round(prob[transfer_ind, gw_ind], 4))


        else:
            if unlimited_transfers:
                max_ind = np.nanargmax(p[gw_ind, :-1])
                transfer = transfers[max_ind]
                print(int(gw_ind), slim_elements_df.loc[transfer[0], 'web_name'], np.round(predictions[transfer[0], :], 1), np.round(prob[transfer_ind, gw_ind], 4))
                price.append(slim_elements_df.loc[transfer[0], 'now_cost'])
        
    print('points: ', np.round(sum(best_pitch), decimals=1), '. diff: ',  np.round(best_points-sum(baseline_point), decimals=1), '. price: ', sum(price))
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

    print('points: ', checked_points[best_ind]-baseline_point)
    print('\n')
