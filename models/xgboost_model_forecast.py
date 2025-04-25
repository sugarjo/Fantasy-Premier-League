import os
import re
import pickle
import requests

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

if optimize:
    check_last_data = False
else:
    check_last_data = True

#add 2. one because threshold is bounded upwards. and one because last week is only partly encoded (dynamic features)
#+1. e.g 28 here means 29 later.
temporal_window = 28

season_start = False

method = 'xgboost'

season_dfs = []

season_count = 0

print('HARD CODED short names for HULL, MIDlesborough, SUNDERLAND (2016-17). DOUBLE CHECK IF PROMOTED')

#get each season
for folder in folders:

    directory = os.path.join(directories, folder)
    fixture_data = os.path.join(directory, 'fixtures.csv')
    gws_data = os.path.join(directory, 'gws')
    team_path = os.path.join(directory, "teams.csv")

    #if os.path.isfile(fixture_data) and  os.path.isdir(gws_data):
    if os.path.isdir(gws_data):

        #check that it is not a file
        if folder[-4] != '.':

            print('\n', folder)

            #get id so it can be matched with position
            player_path = directory + '/players_raw.csv'
            df_player = pd.read_csv(player_path)

            #rename befor merge
            df_player = df_player.rename(columns={"id": "element"})

            #insert string for team
            df_teams = pd.read_csv(team_path)
            try:
                string_names = df_teams["short_name"].values
            except:
                string_names = df_teams[" short_name"].values   
                
            df_player["string_team"] = string_names[df_player["team"]-1]

            dfs_gw = []

            #open each gw and get data for players
            for gw_csv in os.listdir(directory + '/gws'):
                if gw_csv[0] == 'g':

                    gw_path = directory + '/gws' + '/' + gw_csv


                    if folder == '2018-19' or folder == '2016-17' or folder == '2017-18':
                        gw = pd.read_csv(gw_path, encoding='latin1')
                    else:
                        gw = pd.read_csv(gw_path)
                        
                        
                    #remove assistant manager
                    if 'position' in gw.keys():
                        gw = gw.loc[gw['position'] != 'AM']
                    
                    sum_transfers = sum(gw.transfers_in)

                    if sum_transfers == 0 or season_start:
                        gw[['transfers_in', 'transfers_out']] = np.nan
                    else:
                        gw[['transfers_in', 'transfers_out']] = gw[['transfers_in', 'transfers_out']]/sum_transfers

                    if gw.shape[0] == 0:
                        print(gw_csv, 'is empty')
                    else:
                        print(gw_csv)
                        dfs_gw.append(gw)

            df_gw = pd.concat(dfs_gw)

            df_gw['kickoff_time'] =  pd.to_datetime(df_gw['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
            df_gw = df_gw.sort_values(by='kickoff_time')

            df_gw.reset_index(inplace=True)

            if  folder == '2016-17' or folder == '2017-18' or folder == '2018-19' or folder == '2019-20':
                df_gw['xP'] = np.nan
                
            if  folder == '2016-17' or folder == '2017-18' or folder == '2018-19' or folder == '2019-20' or folder == '2020-21' or folder == '2021-22':
                df_gw[['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']] = np.nan


            #variables I calculate myself
            df_gw['points_per_game'] = np.nan
            df_gw['points_per_played_game'] = np.nan
            df_gw['string_opp_team'] = None

            # Calculate values on my own
            for player in df_gw['element'].unique():

                selected_ind = df_gw['element'] == player
                player_df = df_gw[selected_ind]
                player_df.set_index('kickoff_time', inplace=True)
                
                
                opp_team = []
                for team in player_df['opponent_team'].astype(int).values-1:
                    opp_team.append(string_names[team])

                df_gw.loc[selected_ind, 'string_opp_team'] = opp_team.copy()
                
                
                points_per_game =  player_df['total_points'].cumsum() / (player_df['round'])
                
                #points per played game
                result = np.zeros(len(player_df['total_points'])+1)  # initialize result array
                last_games = 0  # initialize last_vplayer_df['total_points']alue to 0
                last_point = 0
                
                own_team_points = []
                own_element_points = []
                wins = []


                for i in range(len(player_df['total_points'])):

                    if player_df['minutes'].iloc[i] >= 60:
                        last_point += player_df['total_points'].iloc[i]
                        last_games += 1

                    if last_games > 0:
                        result[i+1] = last_point/last_games

                df_gw.loc[selected_ind, 'points_per_game'] = points_per_game.values.copy()
                df_gw.loc[selected_ind, 'points_per_played_game'] = result[:-1].copy()
                # df_gw.loc[selected_ind, 'own_team_points'] = own_team_points.copy()
                # df_gw.loc[selected_ind, 'own_wins'] = wins.copy()
                # df_gw.loc[selected_ind, 'own_element_points'] = wins.copy()
    
                
                
            season_df = df_gw[['minutes', 'string_opp_team', 'transfers_in', 'transfers_out', 'ict_index', 'influence', 'threat', 'creativity', 'bps', 'element', 'fixture', 'total_points', 'round', 'was_home', 'kickoff_time', 'xP', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded', 'points_per_game', 'points_per_played_game']]#, 'own_team_points', 'own_wins', 'own_element_points']]
            
            if  folder == '2016-17' or folder == '2017-18':
                season_df[["team_a_difficulty", "team_h_difficulty", "fixture"]] = np.nan
            else:
                #get fixture difficulty difference for each datapoint
                #open fixtures data
                fixture_df = pd.read_csv(fixture_data)
    
                #rename befor merge
                fixture_df = fixture_df.rename(columns={"id": "fixture"})
                season_df = pd.merge(season_df, fixture_df[["team_a_difficulty", "team_h_difficulty", "fixture"]], on='fixture')
    
            season_df = pd.merge(season_df, df_player[["element_type", "first_name", "second_name", "web_name", "string_team", "element"]], on="element")

            season_df["season"] = folder

            season_dfs.append(season_df)

season_df = pd.concat(season_dfs)
season_df['transfers_in'] = season_df['transfers_in'].astype(float)
season_df['transfers_out'] = season_df['transfers_out'].astype(float)
season_df['points_per_game'] = season_df['points_per_game'].astype(float)

season_df['names'] = season_df['first_name'] + ' ' + season_df['second_name']

#MATCH NAMES
name_position_list = season_df[['names', 'element_type']].drop_duplicates(keep='first')

#get current names
#get statistics of all players
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
js = r.json()


if check_last_data:
    current_season_names = season_df.season == season_df.season.iloc[-1]
    current_season = season_df.loc[current_season_names]
    
    downloaded_fixtures = False
    
    for element in js["elements"]:
        
        # if element["web_name"] == 'Pickford':
        #     k = element
        

        if np.double(element["form"]) > 0:
            matched = 0
            
            for name_xls in np.unique(current_season["web_name"]):
                
               
                #get last match from excel sheet
                selected = current_season["web_name"] == element["web_name"]
                
                
                if sum(selected) > 0:
                    xls_matches = current_season.loc[selected]
                # else: 
                #     last_xls_match =  datetime.now() - timedelta(days=365)
                
                if element["web_name"] == name_xls:
                    
                    matched += 1
                    
                    player_id =  element["id"]
                    
                    downloaded = False
                    while not downloaded:
                        try:
                            url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id)
                            r = requests.get(url)
                            player = r.json()
                            downloaded = True
                        except:
                            print('Error in download')
                            time.sleep(30)     
                    
                    for history in player["history"]:
                    
                        if history["minutes"] > 0:
                            kick_off = history["kickoff_time"]
                            kickoff_timestamp = datetime.fromisoformat(kick_off.replace('Z', '+00:00')).astimezone(pytz.UTC) 
                            kickoff_timestamp = kickoff_timestamp.replace(tzinfo=None)    
                            
                            # Convert this datetime to a pandas.Timestamp
                            kickoff_timestamp = pd.Timestamp(kickoff_timestamp)
                            
                            time_diffs = kickoff_timestamp - xls_matches.kickoff_time
                            
                            #check last match
                            if np.min(np.abs(time_diffs)) > pd.Timedelta(0):

                                
                                fixture = history["fixture"]
                                
                                if not downloaded_fixtures:
                                    url = 'https://fantasy.premierleague.com/api/fixtures' + '?event=' + str(history["round"])
                                    r = requests.get(url)
                                    gw = r.json()
                                    downloaded_fixtures = True
                                    
                                for g in gw:
                                    if g["id"] == fixture:
                                        #print(g)
                                        team_h_difficulty = g['team_h_difficulty']
                                        team_a_difficulty = g['team_a_difficulty']
                                        
                                    
                                string_opp_team = js["teams"][history["opponent_team"]-1]["short_name"]
                                string_team = js["teams"][element["team"]-1]["short_name"]
                                
                                #insert data
                                print('Insert live data for',  kickoff_timestamp, string_team, element["web_name"])
                                
                                element_type = element["element_type"]
                                
                                new_row = {
                                        'index': [season_df.index[-1]+1],
                                        'minutes': history["minutes"],          
                                        'string_opp_team': string_opp_team,
                                        'transfers_in': history["transfers_in"],    # Replace with actual value
                                        'transfers_out': history["transfers_out"],    # Replace with actual value
                                        'ict_index': history["ict_index"],       # Replace with actual value
                                        'influence': history["influence"],       # Replace with actual value
                                        'threat': history["threat"],          # Replace with actual value
                                        'creativity': history["creativity"],      # Replace with actual value
                                        'bps': history["bps"],   
                                        'element': history["element"],   
                                        'fixture': fixture,       
                                        'total_points': history["total_points"],      
                                        'round': history["round"], 
                                        'was_home': history["was_home"], 
                                        'kickoff_time': pd.Timestamp(kickoff_timestamp),
                                        'xP': np.nan, 
                                        'expected_goals': history["expected_goals"], 
                                        'expected_assists': history["expected_assists"], 
                                        'expected_goal_involvements': history["expected_goal_involvements"], 
                                        'expected_goals_conceded': history["expected_goals_conceded"], 
                                        'points_per_game': np.float64(element["points_per_game"]), 
                                        'points_per_played_game': np.nan, 
                                        'team_a_difficulty': team_a_difficulty, 
                                        'team_h_difficulty': team_h_difficulty, 
                                        'element_type': element["element_type"],  
                                        'first_name': element["first_name"],  
                                        'second_name': element["second_name"],
                                        'web_name': element["web_name"],
                                        'string_team': string_team,
                                        'season': season_df.season.iloc[-1],
                                        'names': element["first_name"] + ' ' + element["second_name"],
                                    }
                                
                                season_df = pd.concat([season_df, pd.DataFrame(new_row)], ignore_index=True)
                                
                                
                            
            if matched > 1:
                print('Matched too many', element["web_name"])
            if matched == 0:
                print('Matched too few', element["web_name"])
                    

season_df = season_df.reset_index()

temp_season = pd.DataFrame(index=season_df.index, columns=['own_team_points', 'own_element_points'])

#add info about wins, team points, and element points
for team in np.unique(season_df.string_team):
    
    team_df = season_df.loc[season_df.string_team == team]
    
    for match in np.unique(team_df.kickoff_time):
        
        match_df = team_df.loc[team_df.kickoff_time == match]
        
        temp_season.loc[match_df.index, "own_team_points"] = np.sum(match_df.total_points)       
        
        for element in range(1,5):
            element_point_df = match_df.loc[(match_df.element_type == element) & (match_df.minutes > 60)]
            element_df = match_df.loc[(match_df.element_type == element)]
            temp_season.loc[element_df.index, 'own_element_points'] = np.mean(element_point_df.total_points)
            
season_df = pd.concat([season_df, temp_season], axis=1)

own_keys = ['ict_index', 'influence', 'threat', 'creativity', 'bps', 'total_points', 'xP',
       'expected_goals', 'expected_assists', 'expected_goal_involvements',
       'expected_goals_conceded']

selected = season_df.minutes == 0
season_df.loc[selected, own_keys] = np.nan    

elements_df = pd.DataFrame(js['elements'])
current_names = (elements_df['first_name'] + ' ' + elements_df['second_name']).unique()
current_positions = elements_df['element_type']

names = np.concatenate((current_names, name_position_list['names'][::-1]))
positions =  np.concatenate((current_positions, name_position_list['element_type'][::-1]))

_, indices = np.unique(names, return_index=True)
sorted_indices = np.sort(indices)
all_names = names[sorted_indices]
all_positions = positions[sorted_indices]

from difflib import SequenceMatcher

def sequence_matcher_similarity(s1, s2):
    similarity = SequenceMatcher(None, ' '.join(sorted(s1.split())), ' '.join(sorted(s2.split()))).ratio()
    first_name_similarity = SequenceMatcher(None, s1.split()[0], s2.split()[0]).ratio()
    second_name_similarity = SequenceMatcher(None, s1.split()[1], s2.split()[1]).ratio()

    return similarity, first_name_similarity, second_name_similarity

#make list that keep tracks of the changed names
new_names = all_names.copy()

#not that dangerous to merge previous players, but avoid to merge into current player
#loop through the most recent players first
for name_ind, name in enumerate(all_names[:-1]):

    #where in list to check to avoid merges in the same season
    check_ind = np.max([len(current_names), name_ind+1])

    previous_names = all_names[check_ind:]

    results = [sequence_matcher_similarity(prev_name, name) for prev_name in previous_names]

    # Now unpack the results list into separate variables
    similarity_scores = []
    first_name_similarities = []
    second_name_similarities = []

    for result in results:
        similarity_score, first_name_similarity, second_name_similarity = result
        similarity_scores.append(similarity_score)
        first_name_similarities.append(first_name_similarity)
        second_name_similarities.append(second_name_similarity)

    max_match = np.argmax(similarity_scores)
    matched_name = previous_names[max_match]


    match_ind = -1

    first_name_criteria = (np.array(similarity_scores) > 0.71) & (np.array(first_name_similarities) > 0.47)
    second_name_criteria = (np.array(similarity_scores) > 0.70) & (np.array(second_name_similarities) > 0.6)
    all_criteria = (np.array(similarity_scores) > 0.56) & (np.array(first_name_similarities) > 0.55) & (np.array(second_name_similarities) > 0.67)
    test_criteria = max(similarity_scores) > 1

    print_test = False

    if (matched_name in name or name in matched_name) or max(similarity_scores) > 0.7:
        match_ind = np.argmax(similarity_scores)
    elif any(first_name_criteria):
        match_ind = np.where(first_name_criteria)[0][0]
    elif any(second_name_criteria):
        match_ind = np.where(second_name_criteria)[0][0]
    elif any(all_criteria):
        match_ind = np.where(all_criteria)[0][0]
    elif test_criteria:
        match_ind = np.argmax(similarity_scores)
        print_test = True

    if match_ind > -1:

        matched_name = previous_names[match_ind]

        change_names = season_df['names'] == matched_name

        matched_position = season_df.loc[change_names, 'element_type'].unique()

        root_position = all_positions[name_ind]

        if any(matched_position == root_position):
            new_name = new_names[name_ind]

            do_not_match_names = [['David Martin', 'David Raya Martin'],
                                  ['Caleb Taylor', 'Charlie Taylor'],
                                  ['Solomon March', 'Manor Solomon'],
                                  ['Michael Olise', 'Michael Olakigbe'],
                                  ['Ryan Bennett', 'Rhys Bennett'],
                                  ['Joe Powell', 'Joe Rothwell'],
                                  ['Ashley Williams', 'Ashley Phillips'],
                                  ['Aaron Ramsey', 'Jacob Ramsey'],
                                  ['Lewis Richards', 'Chris Richards'],
                                  ['Ashley Williams', 'Rhys Williams'],
                                  ['Killian Phillips', 'Kalvin Phillips'],
                                  ['Josh Murphy', 'Jacob Murphy'],
                                  ['Matthew Longstaff', 'Sean Longstaff'],
                                  ['Charlie Cresswell', 'Aaron Cresswell'],
                                  ['Dale Taylor', 'Joe Taylor'],
                                  ['Jackson Smith', 'Jordan Smith'],
                                  ['Kayne Ramsay', 'Calvin Ramsay'],
                                  ['Haydon Roberts', 'Connor Roberts'],
                                  ['Mason Greenwood', 'Sam Greenwood'],
                                  ['Joe Bryan', 'Kean Bryan'],
                                  ['Lewis Gibson', 'Liam Gibson'],
                                  ['Daniel Sturridge', 'Sam Surridge'],
                                  ['Alexis Sánchez', 'Carlos Sánchez'],
                                  ['Danny Simpson', 'Jack Simpson'],
                                  ['Bakary Sako', 'Bukayo Saka'],
                                  ['James Tomkins', 'Jake Vokins'],
                                  ['Lewis Brunt', 'Lewis Dunk'],
                                  ['James Tomkins', 'James Tarkowski'],
                                  ['Ben Jackson', 'Ben Johnson'],
                                  ['Tyler Roberts', 'Tyler Morton'],
                                  ['James McArthur', 'James McAtee'],
                                  ['Josh Brownhill', 'Josh Bowler'],
                                  ['Andy King', 'Andy Irving'],
                                  ['Joshua Sims', 'Joshua King'],
                                  ['James Storer', 'James Shea'],
                                  ['Owen Beck', 'Owen Bevan'],
                                  ['Joseph Hungbo' , 'Joe Hodge'],
                                  ['Jonathan Leko', 'Jonathan Rowe'],
                                  [' Christian Fuchs', 'Christian Marques'],
                                  ['Anthony Martial', 'Anthony Mancini'],
                                  ['Jack Simpson', 'Jack Robinson'],
                                  ['Jack Cork', 'Jack Colback'],
                                  ['Simon Mignolet', 'Simon Moore'],
                                  ['Aaron Ramsey', 'Aaron Rowe'],
                                  ['Antonio Valencia', 'Antonio Barreca'],
                                  ['Callum Paterson', 'Callum Slattery'],
                                  ['Ben Wilmot', 'Benjamin White'],
                                  ['Scott Dann', 'Scott Malone'],
                                  ['Sergio Romero', 'Sergio Rico'],
                                  ['James Daly', 'Jamie Donley'],
                                  ['Jason Puncheon', 'Jadon Sancho'],
                                  ['Killian Phillips', 'Philip Billing'],
                                  ['Christian Saydee', 'Christian Nørgaard'],
                                  ['James Sweet', 'Reece James'],
                                  ['Ollie Harrison', 'Harrison Reed'],
                                  ['Richard Nartey', 'Omar Richards'],
                                  ['Charles Sagoe', 'Shea Charles'],
                                  ['Benjamin Mendy', 'Benjamin Fredrick'],
                                  ['Scott Dann', 'Dan Potts'],
                                  ['Ashley Williams', 'Neco Williams'],
                                  ['Matty James', 'James McCarthy'],
                                  ['Ashley Williams', 'William Fish'],
                                  ['Christian Fuchs', 'Christian Marques'],
                                  ['Charlie Savage', 'Charles Sagoe'],
                                  ['Andrew Surman', 'Andrew Moran'],
                                  ['Niels Nkounkou', 'Nicolas Nkoulou']
                                  ]

            continue_marker = False
            for avoid_match in do_not_match_names:
                if matched_name in avoid_match and new_name in avoid_match:
                    continue_marker = True

            if continue_marker:
                continue

            season_df.loc[change_names, 'names'] = new_name

            matched_index = all_names == matched_name
            new_names[matched_index] = new_name

            if new_name in current_names:
                print(name_ind, matched_name + ' changed to ' + new_name)

            if print_test:
                print(name_ind, matched_name + ' changed to ' + new_name, similarity_scores[match_ind], first_name_similarities[match_ind], second_name_similarities[match_ind])

print('Done matching')



#calculate difficulties
home_diff = season_df["team_h_difficulty"].copy()
away_diff = season_df["team_a_difficulty"].copy()

difficulty_diff = (home_diff - away_diff)

season_df['difficulty'] = difficulty_diff

home = season_df['was_home'] == 1
season_df.loc[home, 'difficulty'] = -season_df.loc[home, 'difficulty']

#for all away matches
season_df['own_difficulty'] = season_df["team_a_difficulty"].copy()
season_df['other_difficulty'] = season_df["team_h_difficulty"].copy()
#correct home matches
season_df.loc[home, 'own_difficulty'] = season_df.loc[home, "team_h_difficulty"]
season_df.loc[home, 'other_difficulty'] = season_df.loc[home, "team_a_difficulty"]




#categories for dtype
categorical_variables = ['element_type', 'string_team', 'season', 'names']
season_df[categorical_variables] = season_df[categorical_variables].astype('category')
#add nan categories
dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
        'other_difficulty'] #'difficulty',

int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points']
season_df[int_variables] = season_df[int_variables].astype('Int64')

float_variables = ['transfers_in', 'transfers_out', 'threat', 'own_element_points', 'xP', 'expected_goals', 'expected_assists',
'expected_goal_involvements', 'expected_goals_conceded', 'creativity', 'ict_index', 'influence']
season_df[float_variables] = season_df[float_variables].astype('float')



#ALL VARIABLES:
#how variables are includedvariables
#always included. also for current weak
dynamic_features = ['string_opp_team', 'transfers_in', 'transfers_out',
        'was_home', 'own_difficulty', 'other_difficulty']#, 'difficulty']

#features that I don't have access to in advance.
#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'xP', 'expected_goals', 'expected_assists',
        'expected_goal_involvements', 'expected_goals_conceded', 'own_team_points', 'own_element_points']
#included once
temporal_single_features = ['points_per_game', 'points_per_played_game']
#total_points, minutes, kickoff time not for prediction
#included once
fixed_features = ['total_points', 'minutes', 'kickoff_time', 'element_type', 'string_team', 'season', 'names']


# #how variables are included
# #always included. also for current weak
# dynamic_features = ['string_opp_team', 'was_home', 'own_difficulty']#, 'difficulty']

# #features that I don't have access to in advance.
# #included for all windows, but not current
# temporal_features = []
# #included once
# temporal_single_features = ['points_per_played_game']
# #total_points, minutes, kickoff time not for prediction
# #included once
# fixed_features = ['total_points', 'minutes', 'kickoff_time', 'string_team', 'names']


#initiate train dataframe
# Reorder the DataFrame based on the 'date' column
season_df = season_df.sort_values(by='kickoff_time')

# If you want to sort the DataFrame in descending order
# df_sorted = df.sort_values(by='date', ascending=False)

train = pd.DataFrame(season_df[fixed_features])
    

#for each week iteration
for k in range(temporal_window):
    print('Window', k)

    temporal_names = [str(k) + s for s in temporal_features]

    # Create an empty DataFrame with the specified columns
    if k==0:
        
        dynamic_names = [s for s in dynamic_features]        
        temporal_single_names = [s for s in temporal_single_features]
        
        col_names = temporal_single_names + dynamic_names + temporal_names 

    else:
        dynamic_names = [str(k-1) + s for s in dynamic_features] 
        
        col_names = dynamic_names + temporal_names

    temp_train = pd.DataFrame(index=train.index, columns=col_names)
    

    for name in season_df.names.unique():
        
        selected_ind = season_df.names == name
        
        if k==0:
            temporal_single_data = season_df.loc[selected_ind, temporal_single_features].shift(k+1)

            temp_train.loc[selected_ind, temporal_single_names] = temporal_single_data.values
            

        temporal_data = season_df.loc[selected_ind, temporal_features].shift(k+1)
        dynamic_data = season_df.loc[selected_ind, dynamic_features].shift(k)
        
        temp_train.loc[selected_ind, dynamic_names] = dynamic_data.values
        temp_train.loc[selected_ind, temporal_names] = temporal_data.values       
        
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

    #concatenate
    train = pd.concat([train, temp_train], axis=1)


    # dynamic_cat_names = [str(k) + s for s in dynamic_categorical_variables]
    # #set categories of opponents:
    # train[dynamic_cat_names] = train[dynamic_cat_names].astype('category')

    #get the possible opponents (#in case of new team in the dataset)    
    if k == 0:
        opponent_feature = 'string_opp_team'
        possible_opponents = train[opponent_feature].cat.categories
        opp_cats = CategoricalDtype(categories=possible_opponents, ordered=False)
    else:
        opponent_feature = str(k-1) + 'string_opp_team'
        train[opponent_feature] = train[opponent_feature].astype(opp_cats)
  
#add in data about the opponent   
opponent_point_names = [str(k) + 'opp_team_points' for k in range(temporal_window)]  
opponent_element_names = [str(k) + 'opp_element_points' for k in range(temporal_window)]  

temp_train = pd.DataFrame(index=train.index, columns=opponent_point_names + opponent_element_names)
     

for opponent_club in season_df.string_opp_team.unique():
    
    opp_selected = season_df.string_opp_team == opponent_club
    
    #loop through all matches 
    for kickoff in np.unique(season_df.loc[opp_selected, 'kickoff_time']):
        
        #find all matches of the opponent before the current match
        opp_match_selected =  opp_selected & (season_df['kickoff_time'] < kickoff)
            
        #find the unique kickoff times
        first_indices = season_df.loc[opp_match_selected].drop_duplicates(subset='kickoff_time', keep='first').index
        
        full_ooop = np.full(len(opponent_point_names), np.nan)
        
        opponents_of_opponents_points = season_df.loc[first_indices[-temporal_window:], "own_team_points"]
        
        if len(opponents_of_opponents_points):
            full_ooop[-len(opponents_of_opponents_points):] = opponents_of_opponents_points
        
        relevant_players =  opp_selected & (season_df['kickoff_time'] == kickoff)
        temp_train.loc[relevant_players, opponent_point_names] = full_ooop[::-1]
        
        
        for element_type in range(1,5):       
            #find all matches of the opponent before the current match
            opp_elem_selected =  opp_selected & (season_df['kickoff_time'] < kickoff) & (season_df['element_type'] == element_type)
                
            #find the unique kickoff times
            first_indices = season_df.loc[opp_elem_selected].drop_duplicates(subset='kickoff_time', keep='first').index
            
            full_oooep = [np.nan] * len(opponent_element_names)
            
            opponents_of_opponents_elements = season_df.loc[first_indices[-temporal_window:], "own_element_points"]
            
            if len(opponents_of_opponents_points):
                full_oooep[-len(opponents_of_opponents_elements):] = opponents_of_opponents_elements
            
            relevant_elements =  opp_selected & (season_df['kickoff_time'] == kickoff) & (season_df['element_type'] == element_type)
            temp_train.loc[relevant_elements, opponent_element_names] = full_oooep[::-1]


#set dtype
for col in opponent_point_names:
    temp_train[col] = temp_train[col].astype('Int64')
    
for col in opponent_element_names:
    temp_train[col] = temp_train[col].astype('float')
    
            
train = pd.concat([train, temp_train], axis=1)            

#exchange old names with nan
for name in train.names.unique():
    if not name in current_names:
        selected = train.names == name
        train.loc[selected, 'names'] = np.nan

original_df = season_df.copy()

#remove players that didn't play 60min
selected = season_df["minutes"] >= 60
season_df = train.loc[selected]
season_df = season_df.drop(['minutes'], axis=1)

#remove players with few matches
unique_names = season_df.names.unique()

n_tresh = 3

for unique_ind, name in enumerate(unique_names):
    selected = (season_df.names == name)

    if sum(selected) < n_tresh:
        season_df.loc[selected, 'names'] = np.nan

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
    # Extract all numbers from the column name
    numbers = re.findall(r'\d+', column_name)
    for number in numbers:
        # If any number is lower than the threshold, return False
        if int(number) < threshold:
            return True
        else:
            return False
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
    
    # interaction_constraints = get_interaction_constraints(objective_X.columns)
    # pars['interaction_constraints'] = str(interaction_constraints)

    #fit_X, eval_X, fit_y, eval_y, fit_sample_weights, eval_sample_weights = train_test_split(objective_X, cv_y, cv_sample_weights, test_size=space['eval_fraction'], stratify=cv_stratify, random_state=42)

    
    val_ind = int(objective_X.shape[0]*space['eval_fraction'])
    
    fit_X = objective_X.iloc[:-val_ind].copy()
    eval_X =  objective_X.iloc[-val_ind:].copy()
    fit_y =  cv_y.iloc[:-val_ind].copy()
    eval_y = cv_y.iloc[-val_ind:].copy()
    fit_sample_weights =  cv_sample_weights[:-val_ind].copy()
    eval_sample_weights = cv_sample_weights[-val_ind:].copy()
    
    #make sure all categories in val_x is present in cv_x
    for column in eval_X.columns:
        if pd.api.types.is_categorical_dtype(eval_X[column]):
            # Get the values in the current column of val_X
            val_values = eval_X[column]
            
            # Check which values are present in the corresponding column of cv_X
            mask = val_values.isin(fit_X[column])
            
            # Set values that are not present in cv_X[column] to NaN
            eval_X.loc[~mask, column] = np.nan
    
    dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True, weight=fit_sample_weights)
    deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True, weight=eval_sample_weights)

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
    dval_objective = xgb.DMatrix(data= objective_val_X, label=val_y, enable_categorical=True, weight=val_sample_weights)

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

    print(space)

    model = LinearSVR(**space, fit_intercept=False, dual="auto")

    model.fit(scaled_cv_X, cv_y)

    #selected = val_sample_weights >= 1
    selected = val_X['running_minutes'] > 60

    val_pred = model.predict(scaled_val_X[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)

    print(val_error)

    return {'loss': val_error, 'status': STATUS_OK }


#weight samples for time
last_year = season_df['kickoff_time'].iloc[-1] - season_df['kickoff_time']
selected = last_year < timedelta(365)
sample_weights = np.ones(selected.shape)
sample_weights[selected] = 4

season_df = season_df.drop(['kickoff_time'], axis=1)

#season_df.replace(to_replace=[None], value=np.nan, inplace=True)

#train model. no changes of catgeories in train_X after this point!
train_X = season_df.drop(['total_points'], axis=1)
train_y = season_df['total_points'].astype(int)

# Identify categorical columns
categorical_columns = train_X.select_dtypes(['category']).columns

# Reset categories for each categorical column
for column in categorical_columns:
    train_X[column] = train_X[column].cat.remove_unused_categories()

# Define the number of quantiles/bins
num_bins = 100

# Calculate the quantile boundaries of the outcome variable
centiles = pd.qcut(train_y, q=100, duplicates="drop", retbins=True)[1]
centiles[0] = -np.inf
# Discretize the outcome variable using the quantile boundaries
stratify = pd.cut(train_y, bins=centiles, labels=False)

# min_y = np.min(train_y)
# train_y = np.log10(train_y-min_y+1)

cw = compute_class_weight('balanced', classes=np.unique(stratify), y=stratify)

for k in np.unique(stratify):
    selected = stratify == k
    sample_weights[selected] = sample_weights[selected]*cw[k]


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

    df_one_hot = pd.get_dummies(train_X, columns=['string_team', 'string_opp_team', 'element_type', 'names'])

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

    #remove columns with a lot of nans
    keep_col = ['names', 'running_minutes', 'transfer_in', 'transfer_out', 'running_ict', 'running_influence', 'running_threat', 'running_creativity', 'running_bps', 'string_opp_team', 'string_team', 'element_type', 'was_home', 'form', 'points_per_game', 'points_per_played_game',  'other_difficulty', 'own_difficulty']
    pruned_df = train_X[keep_col]


    # Identify indices of rows to be removed
    mask = pruned_df.isna().any(axis=1)
    nan_indices = pruned_df[mask].index

    # Remove rows with NaN from the original DataFrame
    df_cleaned = pruned_df.dropna()

    # Remove corresponding rows from the output DataFrame
    outputs_cleaned = train_y.drop(nan_indices)
    stratify_cleaned = stratify[~mask]
    sample_weights_cleaned = sample_weights[~mask]

    df_one_hot = pd.get_dummies(df_cleaned, columns=['string_team', 'string_opp_team', 'element_type', 'names', 'other_difficulty', 'own_difficulty'])

    #df_one_hot = df_one_hot.drop('names', axis=1)
    #get an validation set for fitting
    cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(df_one_hot, outputs_cleaned, sample_weights_cleaned, stratify_cleaned, test_size=0.25, stratify=stratify_cleaned, random_state=42)


    scaler = StandardScaler()
    scaled_cv_X = scaler.fit_transform(cv_X)
    scaled_val_X = scaler.transform(val_X)

    # # Compute the column-wise mean of the array
    # col_mean = np.nanmedian(scaled_cv_X, axis=0)

    # # Find indices where NaNs are present
    # inds = np.where(np.isnan(scaled_cv_X))
    # # Replace NaNs with the mean of the column
    # scaled_cv_X[inds] = np.take(col_mean, inds[1])
    # # Find indices where NaNs are present
    # inds = np.where(np.isnan(scaled_val_X))
    # # Replace NaNs with the mean of the column
    # scaled_val_X[inds] = np.take(col_mean, inds[1])


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
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from sklearn.preprocessing import StandardScaler
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ReduceLROnPlateau
    from keras.regularizers import l1, l2, l1_l2

    #remove columns with a lot of nans
    keep_col = ['names', 'running_minutes', 'transfer_in', 'transfer_out', 'running_ict', 'running_influence', 'running_threat', 'running_creativity', 'running_bps', 'string_opp_team', 'string_team', 'element_type', 'was_home', 'form', 'points_per_game', 'points_per_played_game',  'other_difficulty', 'own_difficulty']
    pruned_df = train_X[keep_col]


    # Identify indices of rows to be removed
    mask = pruned_df.isna().any(axis=1)
    nan_indices = pruned_df[mask].index

    # Remove rows with NaN from the original DataFrame
    df_cleaned = pruned_df.dropna()

    # Remove corresponding rows from the output DataFrame
    outputs_cleaned = train_y.drop(nan_indices)
    stratify_cleaned = stratify[~mask]
    sample_weights_cleaned = sample_weights[~mask]

    df_one_hot = pd.get_dummies(df_cleaned, columns=['string_team', 'string_opp_team', 'element_type', 'names', 'other_difficulty', 'own_difficulty'])

    #df_one_hot = df_one_hot.drop('names', axis=1)
    #get an validation set for fitting
    cv_X, val_X, cv_y, val_y, cv_sample_weights, _, cv_stratify, _ = train_test_split(df_one_hot, outputs_cleaned, sample_weights_cleaned, stratify_cleaned, test_size=0.25, stratify=stratify_cleaned, random_state=42)

    fit_X, eval_X, fit_y, eval_y, fit_sample_weights, _ = train_test_split(cv_X, cv_y, cv_sample_weights, test_size=0.25, stratify=cv_stratify, random_state=42)

    scaler = StandardScaler()
    scaled_fit_X = scaler.fit_transform(fit_X)
    scaled_eval_X = scaler.transform(eval_X)

    scaler = StandardScaler()
    scaled_cv_X = scaler.fit_transform(cv_X)
    scaled_val_X = scaler.transform(val_X)

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
    selected = val_X['running_minutes'] > 60

    val_pred = model.predict(scaled_val_X[selected])
    val_error = mean_squared_error(val_y[selected], val_pred)
    print(val_error)

elif method == 'mixedLM':

    train_y.columns = ['total_points']

    min_val = np.min(train_y) - 1

    #train_df = pd.concat([train_X, np.log10(train_y-min_val)], axis=1)
    train_df = pd.concat([train_X, train_y], axis=1)

    #remove with columns with a lot of nans
    keep_col = ['names', 'total_points', 'running_minutes', 'transfer_in', 'transfer_out', 'running_ict', 'running_influence', 'running_threat', 'running_creativity', 'running_bps', 'string_opp_team', 'string_team', 'element_type', 'was_home', 'form', 'points_per_game', 'points_per_played_game',  'other_difficulty', 'own_difficulty']
    pruned_df = train_df[keep_col]

    # Identify indices of rows to be removed
    mask = pruned_df.isna().any(axis=1)
    nan_indices = pruned_df[mask].index

    # Remove rows with NaN from the original DataFrame
    df_cleaned = pruned_df.dropna()
    stratify_cleaned = stratify[~mask]

    #df_one_hot = df_one_hot.drop('names', axis=1)
    #get an validation set for fitting
    cv, val = train_test_split(df_cleaned, test_size=0.25, stratify=stratify_cleaned, random_state=42)

    #doesn't work with categorical variables
    #fit_string = "total_points ~ running_minutes + transfer_in + transfer_out + running_ict + running_influence + running_threat + running_creativity + running_bps + C(string_opp_team) + C(string_team) + C(element_type) + was_home + running_xP + running_xG + running_xA + running_xGI + running_xGC + form + points_per_game + points_per_played_game + C(other_difficulty) + C(own_difficulty)"
    fit_string = "total_points ~ running_minutes + transfer_in + transfer_out + running_ict + running_influence + running_threat + C(string_opp_team) + C(element_type) + was_home + points_per_game + points_per_played_game + C(other_difficulty) + C(own_difficulty)"

    model = sm.MixedLM.from_formula(fit_string, groups='names', data=cv)
    result = model.fit()

    prediction = result.predict(val)

    re = result.random_effects

    name_list = list(re.keys())

    for row in val.iterrows():
        if row[1]["names"] in name_list:
            random_effect = re[row[1]["names"]].iloc[0]
        else:
            random_effect = 0

        # if row[1]["element_type"] in name_list:
        #     random_effect = re[row[1]["element_type"]].iloc[0]
        # else:
        #     random_effect = 0

        prediction[row[0]] = prediction[row[0]] + random_effect

    selected = val['running_minutes'] > 60

    val_error = mean_squared_error(val['total_points'][selected], prediction[selected])
    #val_error = mean_squared_error((10**val['total_points'][selected] + min_val), 10**prediction[selected] + min_val)
    print(val_error)


elif method == 'xgboost':

    
    #get an validation set for fitting
    #cv_X, val_X, cv_y, val_y, cv_sample_weights, val_sample_weights, cv_stratify, _= train_test_split(train_X, train_y, sample_weights, stratify, test_size=0.20, stratify=stratify, random_state=42)
    #use half of the current seasons's data as test set   
    
    #20% for testing
    val_ind = int(train_X.shape[0]*0.2)
    
    cv_X = train_X.iloc[:-val_ind].copy()
    val_X =  train_X.iloc[-val_ind:].copy()
    cv_y =  train_y.iloc[:-val_ind].copy()
    val_y = train_y.iloc[-val_ind:].copy()
    cv_sample_weights =  sample_weights[:-val_ind].copy()
    val_sample_weights = sample_weights[-val_ind:].copy()
    cv_stratify = stratify[:-val_ind].copy()
    
    
    #make sure all categories in val_x is present in cv_x
    for column in val_X.columns:
        if pd.api.types.is_categorical_dtype(val_X[column]):
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
    
    
    min_eval_fraction = 1/cv_X.shape[0] #len(np.unique(cv_stratify))/cv_X.shape[0]
    

    space={'max_depth': hp.quniform("max_depth", 1, 1000, 1),
            'min_split_loss': hp.uniform('min_split_loss', 0, 45), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 150),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 150),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 400),
            'learning_rate': hp.uniform('learning_rate', 0, 0.05),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 75, 2000, 1),
            'eval_fraction': hp.loguniform('eval_fraction', np.log(min_eval_fraction), np.log(0.2)),
            'n_estimators': hp.quniform('n_estimators', 2, 17000, 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 25),
            'grow_policy': hp.choice('grow_policy', grow_policy), #111
            'max_leaves': hp.quniform('max_leaves', 0, 1100, 1),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(125), 1),
            'temporal_window': hp.quniform('temporal_window', 0, temporal_window+1, 1),
        }


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
    
    loss = objective_xgboost(old_hyperparams)
    old_loss = loss['loss']
    
    print('Old loss: ', old_loss)
        
    #optimize and iteratively get hyperparamaters
    batch_size = 100
    if optimize:
        max_evals = 500000
    
    if continue_optimize:
        hyperparam_path = main_directory + '\models\hyperparams_test.pkl'
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
            hyperparam_path = main_directory + '\models\hyperparams_test.pkl'
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
        print(losses[best_cv_trial])
    
        hyperparams = trials.trials[best_cv_trial]['misc']['vals']
        print(hyperparams)
    
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
    
        #fit_X, eval_X, fit_y, eval_y, fit_sample_weights, eval_sample_weights = train_test_split(objective_X, train_y, sample_weights, test_size=space['eval_fraction'], stratify=stratify, random_state=42)
        
        current_season = objective_X.iloc[-1].season
        selected_test = objective_X.season == current_season
        
        val_ind = int(objective_X.shape[0]*space['eval_fraction'])
        
        fit_X = objective_X.iloc[:-val_ind].copy()
        eval_X =  objective_X.iloc[-val_ind:].copy()
        fit_y =  train_y.iloc[:-val_ind].copy()
        eval_y = train_y.iloc[-val_ind:].copy()
        fit_sample_weights =  sample_weights[:-val_ind].copy()
        eval_sample_weights = sample_weights[-val_ind:].copy()
        
        #make sure all categories in val_x is present in cv_x
        for column in eval_X.columns:
            if pd.api.types.is_categorical_dtype(eval_X[column]):
                # Get the values in the current column of val_X
                val_values = eval_X[column]
                
                # Check which values are present in the corresponding column of cv_X
                mask = val_values.isin(fit_X[column])
                
                # Set values that are not present in cv_X[column] to NaN
                eval_X.loc[~mask, column] = np.nan
        
        
        dfit = xgb.DMatrix(data=fit_X, label=fit_y, enable_categorical=True, weight=fit_sample_weights)
        deval = xgb.DMatrix(data=eval_X, label=eval_y, enable_categorical=True, weight=eval_sample_weights)
    
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
    
        summary = {'model': model, 'train_features': train_X, 'hyperparameters': space, 'all_rows': original_df}
    
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

        print(sorted_mean_values)  # Output will be sorted by mean values

