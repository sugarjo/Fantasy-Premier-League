import os
import re
import requests
import json

import pandas as pd
import numpy as np
from datetime import datetime
import pytz

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


season_start = False


check_last_data = False


temporal_window = 30
    


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
    cleaned_string = cleaned_string.replace("'", "")
    # Remove all numbers
    cleaned_string = re.sub(r'\d+', '', cleaned_string)
    return cleaned_string.strip()  # Optional: strip leading/trailing spaces

if check_last_data:
    
    print('Scrap static game info')
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    static = r.json()
    static_df = pd.DataFrame(static['elements'])
        
    if check_last_data:
        #scrap fantasy data:
        #get current names
        #get statistics of all players
    
        
        # Save to a JSON file
        with open(r'C:\Users\jorgels\Git\Fantasy-Premier-League\data\2025-26\static_2025-26.json', 'w') as json_file:
            json.dump(static, json_file)
        
        #loop players
        #store to only scrap once
        players = []
        print('Scrap player info')
        for player_id in static_df.iterrows():
            
            # if element["web_name"] == 'Pickford':
            #     k = element
        
            downloaded = False
            while not downloaded:
                try:
                    url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id[1].id)
                    r = requests.get(url)
                    player = r.json()
                    downloaded = True
                    
                    player["id"] = player_id[1].id
                    
                    # Save to a JSON file
                    filename = fr'C:\Users\jorgels\Git\Fantasy-Premier-League\data\2025-26\players\{player_id[1].id}_2025-26.json'
                    with open(filename , 'w') as json_file:
                        json.dump(player, json_file)
                except:
                    print('Error in download')
                    time.sleep(30)
                
                players.append(player) 
                    
            
        #download gw
        print('Scrap fixtures and gameweeks')
        for i in range(1, 39):
            url = 'https://fantasy.premierleague.com/api/fixtures' + '?event=' + str(i)
            r = requests.get(url)
            #has the length of matches (10)
            gw = r.json()
            
            # Save to a JSON file
            filename = fr'C:\Users\jorgels\Git\Fantasy-Premier-League\data\2025-26\fixtures\{i}_2025-26.json'
            with open(filename , 'w') as json_file:
                json.dump(gw, json_file)
                
            gw_data = []
            #create gw data:
            #loop the scrapped players
            for player in players:
               
                for history in player["history"]:
                    
                    if history["round"] == i:
        
                        if history["minutes"] > 0:
                            
                            player_id = player["id"]
                            
                            fixture = history["fixture"]
                            
                            kick_off = history["kickoff_time"]
                            kickoff_timestamp = datetime.fromisoformat(kick_off.replace('Z', '+00:00')).astimezone(pytz.UTC) 
                            #kickoff_timestamp = kickoff_timestamp.replace(tzinfo=None)    
                            
                            # Convert this datetime to a pandas.Timestamp
                            #kickoff_timestamp = pd.Timestamp(kickoff_timestamp)
                                
                            for g in gw:
                                if g["id"] == fixture:
                                    #print(g)
                                    team_h_difficulty = g['team_h_difficulty']
                                    team_a_difficulty = g['team_a_difficulty']
                            string_opp_team = static["teams"][history["opponent_team"]-1]["short_name"]
                            
                            selected_static = static_df.id == player_id
                            team_num = static_df.loc[selected_static ].team.iloc[0]
                            string_team = static["teams"][team_num-1]["short_name"]
                            
                            #insert data
                            #print('Insert live data for',  kickoff_timestamp, string_team, element["web_name"])
                            
                            element_type = static_df.loc[selected_static ]["element_type"].iloc[0]
                            
                            new_row = {
                                    #'index': [season_df.index[-1]+1],
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
                                    'kickoff_time': kickoff_timestamp,
                                    #'xP': np.nan, 
                                    'expected_goals': history["expected_goals"], 
                                    'expected_assists': history["expected_assists"], 
                                    #'expected_goal_involvements': history["expected_goal_involvements"], 
                                    'expected_goals_conceded': history["expected_goals_conceded"], 
                                    #'points_per_game': static_df.loc[selected_static]["points_per_game"].iloc[0], 
                                    #'points_per_played_game': np.nan, 
                                    'team_a_difficulty': team_a_difficulty, 
                                    'team_h_difficulty': team_h_difficulty, 
                                    'element_type': element_type,  
                                    'first_name': static_df.loc[selected_static]["first_name"].iloc[0],  
                                    'second_name': static_df.loc[selected_static]["second_name"].iloc[0],
                                    'web_name': static_df.loc[selected_static]["web_name"].iloc[0],
                                    'string_team': string_team,
                                    'season': '2025-26',
                                    'name': static_df.loc[selected_static ]["first_name"].iloc[0] + ' ' + static_df.loc[selected_static ]["second_name"].iloc[0],
                                }
                            
                            gw_data.append(new_row)
            if gw_data:  
                gw_df = pd.DataFrame(gw_data)
                filename = fr'C:\Users\jorgels\Git\Fantasy-Premier-League\data\2025-26\gws\gw{i}.json'
                gw_df.to_json(filename)
                        
            
    
    season_dfs = [] 
    #get each previous season
    for folder in folders:
        
        #get data from vastaav
        directory = os.path.join(directories, folder)
        fixture_csv = os.path.join(directory, 'fixtures.csv')
        gws_data = os.path.join(directory, 'gws')
        team_path = os.path.join(directory, "teams.csv")
    
        #if os.path.isfile(fixture_data) and  os.path.isdir(gws_data):
        if os.path.isdir(gws_data):
    
            #check that it is not a file
            if folder[-4] != '.':
    
                print('\n', folder)   
    
                season_data = pd.read_csv(directory + '\\fbref/' + folder[:-2] + '20' + folder[-2:] + '_player_data.csv')
                fixture_data = pd.read_csv(directory + '\\fbref/'  + folder[:-2] + '20' + folder[-2:] + '_fixture_data.csv')
         
                #correct assign  game_id. Assume they are ordered
                for id, ind in enumerate(np.unique(fixture_data.game_id)):
                    selected_id = fixture_data.game_id == ind
                    fixture_data.loc[selected_id, 'game_id'] = id
                
                
                fixture_data['kickoff_time'] = pd.to_datetime(fixture_data['Date'] + ' ' + fixture_data['Time'])
                   
                season_data = pd.merge(season_data, fixture_data.loc[:, ['Wk', 'game_id', 'kickoff_time']], on='game_id', how='left')
                season_data.rename(columns={'Wk': 'gameweek'}, inplace=True)
                
                print('Season_data range from', min(season_data.gameweek), max(season_data.gameweek))
    
                #get id so it can be matched with position
                try: 
                    player_path = directory + f'/static_{folder}.json'
                    with open(player_path, 'r') as json_file:
                        static_data = json.load(json_file)  # Load the data from the file
                    
                    # Convert the loaded data to a DataFrame
                    df_player = pd.DataFrame(static_data["elements"])
                    
                    string_names = np.array([t["short_name"] for t in static_data["teams"]])
                    
                    json_file = True
                
    
                except:
                    player_path = directory + '/players_raw.csv'
                    df_player = pd.read_csv(player_path)
                    
                    #insert string for team
                    df_teams = pd.read_csv(team_path)
                    
                    string_names = df_teams["short_name"].values
                    
                    json_file = False
    
                #rename befor merge
                df_player = df_player.rename(columns={"id": "element"})   
                df_player["string_team"] = string_names[df_player["team"]-1]
    
                dfs_gw = []
                
    
                #open each gw and get data for players
                for gw_csv in os.listdir(directory + '/gws'):
                    if gw_csv[0] == 'g':
    
                        gw_path = directory + '/gws' + '/' + gw_csv
    
                        if not json_file:
                            if folder == '2018-19' or folder == '2016-17' or folder == '2017-18':
                                gw = pd.read_csv(gw_path, encoding='latin1')
                            else:
                                gw = pd.read_csv(gw_path)
                                
                            
                        else:
                            
                            gw = pd.read_json(gw_path)
                            
                            
                        gw_num = int(re.findall(r'\d+', gw_csv)[0])
                        
                        #allocate rows
                        if folder in ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']:
                            gw['expected_goals'] = np.nan
                            gw['expected_assists'] = np.nan
                            gw['expected_goals_conceded'] = np.nan
                        
                        gw['expected_goal_assists'] = np.nan
                        gw['defcon'] = np.nan
                        gw['SoT'] = np.nan
                        
                        
    
                        
                        #covid weeks
                        if gw_num > 38:
                            gw_num = gw_num - 9
                    
                        
                        for el in gw.iterrows():
                            
                            if el[1].minutes == 0:
                                continue
                            
                            name_string = clean_string(el[1]['name']) 
                            
                            results = [sequence_matcher_similarity(prev_name, name_string) for prev_name in np.unique(season_data['Player'])]
                            max_val = -1
                            for ind, k in enumerate(results):
                                if k[0] > max_val:
                                    max_val =  k[0]
                                    max_ind = ind
                            
                            if max_val > 0.77:
                                closest_match = [np.unique(season_data['Player'])[max_ind]]
                            else:
                                closest_match = difflib.get_close_matches(name_string, season_data['Player'], n=1)
    
                            #manually change some names:
                            fantasy_manual_names = ['Bernardo Mota Veiga de Carvalho e Silva', 'Andrey Nascimento dos Santos', 'Alisson Becker', 'João Maria Lobo Alves Palhares Costa Palhinha Gonçalves', 'Igor Jesus Maciel da Cruz', 'João Pedro Ferreira da Silva', 'Murillo Costa dos Santos', 'Matheus Santos Carneiro da Cunha', 'Rúben dos Santos Gato Alves Dias', 'Estêvão Almeida de Oliveira Gonçalves', 'Rodrigo Rodri Hernandez', 'Vitor de Oliveira Nunes dos Reis', 'Welington Damascena Santos', 'Felipe Rodrigues da Silva', 'André Trindade da Costa Neto', 'Francisco Evanilson de Lima Barbosa', 'João Pedro Ferreira Silva', 'Igor Thiago Nascimento Rodrigues', 'Sávio Savinho Moreira de Oliveira', 'Norberto Bercique Gomes Betuncal', 'Anssumane Fati Vieira', 'Victor da Silva', 'Manuel Benson Hedilazio', 'Igor Julio dos Santos de Paulo', 'Murillo Santiago Costa dos Santos', 'Felipe Augusto de Almeida Monteiro', 'Mateus Cardoso Lemos Martins', 'João Victor Gomes da Silva', 'Danilo dos Santos de Oliveira', 'Alexandre Moreno Lopera', 'Carlos Ribeiro Dias', 'Matheus Santos Carneiro Da Cunha', 'Renan Augusto Lodi dos Santos', 'Lyanco Silveira Neves Vojnovic', 'Willian Borges da Silva', 'Carlos Henrique Casimiro', 'Norberto Murara Neto', 'Antony Matheus dos Santos', 'Lucas Tolentino Coelho de Lima', 'Diogo Teixeira da Silva','Fábio Freitas Gouveia Carvalho', 'Emerson Leite de Souza Junior', 'Bernardo Veiga de Carvalho e Silva', 'Francisco Jorge Tomás Oliveira', 'Samir Caetano de Souza Santos', 'Lyanco Evangelista Silveira Neves Vojnovic', 'Emerson Aparecido Leite de Souza Junior', 'Juan Camilo Hernández Suárez', 'José Malheiro de Sá', 'Francisco Machado Mota de Castro Trincão', 'Francisco Casilla Cortés', 'Rúben Santos Gato Alves Dias', 'Raphael Dias Belloli', 'Vitor Ferreira', 'Oluwasemilogo Adesewo Ibidapo Ajayi', 'Ivan Ricardo Neves Abreu Cavaleiro', 'Hélder Wander Sousa de Azevedo e Costa', 'Allan Marques Loureiro', 'Bruno André Cavaco Jordao', 'João Pedro Junqueira de Jesus', 'Borja González Tomás', 'José Reina', 'Roberto Jimenez Gago', 'José Ángel Esmorís Tasende', 'Rodrigo Hernandez', 'Mahmoud Ahmed Ibrahim Hassan', 'José Ignacio Peleteiro Romallo', 'Joelinton Cássio Apolinário de Lira', 'Alexandre Nascimento Costa Silva', 'Fabio Henrique Tavares', 'Bernard Anício Caldeira Duarte', 'André Filipe Tavares Gomes', 'André Filipe Tavares', 'Rúben Gonçalo Silva Nascimento Vinagre', 'Rúben Diogo da Silva Neves', 'Rui Pedro dos Santos Patrício', 'João Filipe Iria Santos Moutinho', 'Jorge Luiz Frello Filho', 'Frederico Rodrigues de Paula Santos', 'Fabricio Agosto Ramírez', 'Bonatini Lohner Maia Bonatini', 'Bernardo Fernandes da Silva Junior', 'Alisson Ramses Becker', 'Olayinka Fredrick Oladotun Ladapo', 'Lucas Rodrigues Moura da Silva', 'João Mário Naval Costa Eduardo', 'Adrien Sebastian Perruchet Silva', 'Danilo Luiz da Silva', 'Jesé Rodríguez Ruiz', 'Jose Luis Mato Sanmartín', 'Ederson Santana de Moraes', 'Bruno Saltor Grau', 'Bernardo Mota Veiga de Carvalho e Silva', 'Robert Kenedy Nunes do Nascimento', 'Gabriel Armando de Abreu', 'Fabio Pereira da Silva', 'Fernando Francisco Reges', 'David Luiz Moreira Marinho', 'Willian Borges Da Silva', 'Pedro Rodríguez Ledesma', 'Oscar dos Santos Emboaba Junior', 'Manuel Agudo Durán', 'Fernando Luiz Rosa', 'Adrián San Miguel del Castillo']
                            fbref_manual_names = ['Bernardo Silva', 'Andrey Santos', 'Alisson', 'João Palhinha', 'Igor Jesus', 'Jota Silva', 'Murillo', 'Matheus Cunha', 'Rúben Dias', 'Estêvão Willian', 'Rodri', 'Vitor Reis', 'Welington', 'Morato', 'André', 'Evanilson', 'Jota Silva', 'Thiago', 'Sávio', 'Beto', 'Ansu Fati', 'Vitinho', 'Benson Manuel', 'Igor', 'Murillo', 'Felipe', 'Tetê', 'João Gomes', 'Danilo', 'Álex Moreno', 'Cafú', 'Matheus Cunha', 'Renan Lodi', 'Lyanco', 'Willian', 'Casemiro', 'Neto', 'Antony', 'Lucas Paquetá', 'Diogo Jota', 'Fabio Carvalho', 'Emerson', 'Bernardo Silva', 'Chiquinho', 'Samir Santos', 'Lyanco', 'Emerson', 'Cucho', 'José Sá', 'Francisco Trincão', 'Kiko Casilla', 'Rúben Dias', 'Raphinha', 'Vitinha', 'Semi Ajayi', 'Ivan Cavaleiro', 'Hélder Costa', 'Allan', 'Bruno Jordão', 'João Pedro', 'Borja Bastón', 'Pepe Reina', 'Roberto', 'Angeliño', 'Rodri', 'Trézéguet', 'Jota', 'Joelinton', 'Xande Silva', 'Fabinho', 'Bernard', 'André Gomes', 'André Gomes', 'Rúben Vinagre', 'Rúben Neves', 'Rui Patrício', 'João Moutinho', 'Jorginho', 'Fred', 'Fabricio', 'Léo Bonatini', 'Bernardo', 'Alisson', 'Freddie Ladapo', 'Lucas Moura', 'João Mário', 'Adrien Silva', 'Danilo', 'Jesé', 'Joselu', 'Ederson', 'Bruno', 'Bernardo Silva', 'Kenedy', 'Gabriel Paulista', 'Fábio', 'Fernando', 'David Luiz', 'Willian', 'Pedro', 'Oscar', 'Nolito', 'Fernandinho', 'Adrián']
                            
                            if name_string in fantasy_manual_names:
                                name_selected = [k == name_string for k in fantasy_manual_names]
                                name_string =  [value for value, flag in zip(fbref_manual_names, name_selected) if flag][0]
                                closest_match = [name_string]
    
    
                            if not closest_match:
                                print('No player matched in fbref', el[1]['name'], name_string)  
                                continue
                            
                            
                            #use names from fbref. same across seasons
                            #use names from fbref. same name across seasons
                            gw.loc[el[0], 'name'] = closest_match[0]
                            
                            #gameweek doesn't work for double rounds. use date instead
                            utc_time = pd.to_datetime(el[1].kickoff_time)   
                            try:
                                london_time = utc_time.tz_convert('Europe/London')
                            except: 
                                london_time = utc_time + pd.Timedelta(hours=1)
                            player_selected = season_data.Player.values == closest_match
                            
                            #find_minimum time
                            player_data = season_data[player_selected]
    
                            # Calculate the absolute time differences for that player
                            time_differences = (player_data['kickoff_time'] - london_time.tz_localize(None)).abs()
                            
                            # Find the index of the minimum difference
                            closest_index = time_differences.idxmin()
                            
                            # Create a boolean index for the whole original DataFrame
                            fbref_selected  = (player_selected) & (season_data['kickoff_time'] == season_data.loc[closest_index, 'kickoff_time'])
                            if min(time_differences) >  pd.Timedelta(minutes=30):
                                print('Time difference is', min(time_differences), el[0], gw.loc[el[0], 'name'], name_string)
                            #fbref_selected = (season_data.kickoff_time == london_time.tz_localize(None)) & (season_data.Player.values == closest_match)
                            #fbref_selected = (season_data.gameweek == gw_num) & (season_data.Player.values == closest_match)
                            
                            
                            
                            if sum(fbref_selected) == 0:
                                print('No fbref games for player', name_string)
                                continue
                            # #also use kickofftime to acocmodate multiple matches
                            # if sum(fbref_selected) > 1:
                            #     utc_time = pd.to_datetime(el[1].kickoff_time)   
                            #     london_time = utc_time.tz_convert('Europe/London')
                            #     fbref_selected = (season_data.kickoff_time == london_time.tz_localize(None)) & (season_data.gameweek == gw_num) & (season_data.Player.values == closest_match)
                                
                            #check if there are two. then merge
                            if sum(fbref_selected) == 2:
                               
                                indices = season_data[fbref_selected].index
                                
                                print('Duplicate recordings for', name_string, indices, 'Merge!')
                                
                                
                                season_data.iloc[indices[0]] = season_data.iloc[indices[0]].fillna(season_data.iloc[indices[1]])
                                season_data.iloc[indices[1]] = season_data.iloc[indices[1]].fillna(season_data.iloc[indices[0]])
                                
                                if season_data.iloc[indices[0]].Min > season_data.iloc[indices[1]].Min:
                                    fbref_selected = season_data.index == indices[0]
                                else:
                                    fbref_selected = season_data.index == indices[1]
                            
                            if not sum(fbref_selected) == 1:
                                a = gjjdjkd
                        
                            #add cbirt points and cbirt data to previous seasons
                            #find position
                            element = el[1].element
                            player_selected =  df_player.element == element
                            position = df_player.loc[player_selected].element_type.iloc[0]
                            
                            #get cbirt scores
                            if folder == '2016-17':
                                if position == 2:
                                    cbirt = sum(gw.loc[el[0], ['clearances_blocks_interceptions', 'tackles']])
                                elif not position == 1:
                                    cbirt = sum(gw.loc[el[0], ['clearances_blocks_interceptions', 'tackles', 'recoveries']])
                                else:
                                    cbirt = gw.loc[el[0], 'saves']
                            else:
                                if position == 2:
                                    cbirt = season_data.loc[fbref_selected, ['Clr', 'Blocks',
                                    'Int', 'Tkl']].sum(axis=1).iloc[0]
                                elif not position == 1:
                                    cbirt = season_data.loc[fbref_selected, ['Clr', 'Blocks',
                                    'Int', 'Tkl', 'Recov']].sum(axis=1).iloc[0]
                                else:
                                    cbirt = season_data.loc[fbref_selected, ['Saves']].sum(axis=1).iloc[0]
                            
                            #add defcon to the stats
                            gw.loc[el[0], 'defcon'] = cbirt
                            
                            #add points for previous seasons
                            if folder in ['2016-17', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']:
                                
                                if position == 2 and cbirt >= 10:
                                    gw.loc[el[0], 'total_points'] += 2
                              
                                elif not position == 1 and cbirt >= 12:
                                    gw.loc[el[0], 'total_points'] += 2
                        
                            #insert statistics from fbref
                            if not folder == '2016-17':                         
                                gw.loc[el[0], 'expected_goals'] = season_data.loc[fbref_selected, 'xG'].iloc[0]
                                gw.loc[el[0], 'expected_goal_assists'] = season_data.loc[fbref_selected, 'xAG'].iloc[0]
                                gw.loc[el[0], 'expected_assists'] = season_data.loc[fbref_selected, 'xA'].iloc[0]
                            
                            #insert xGI
                            if folder in ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22']:
                                #the season data is only correct for players who played 90 min
                                if gw.loc[el[0]].minutes == 90:
                                    gw.loc[el[0], 'expected_goals_conceded']  = season_data.loc[fbref_selected, 'xGC'].iloc[0]
                            
                            gw.loc[el[0], 'SoT'] = season_data.loc[fbref_selected, 'SoT'].iloc[0]
                        
    
                        #remove assistant manager
                        if 'position' in gw.keys():
                            gw = gw.loc[gw['position'] != 'AM']
                        
                        sum_transfers = sum(gw.transfers_in) +  sum(np.abs(gw.transfers_out))
                        
                        #turn to percentage of all transfers
                        if sum_transfers == 0:
                            gw[['transfers_in', 'transfers_out']] = np.nan
                        else:
                            gw[['transfers_in', 'transfers_out']] = gw[['transfers_in', 'transfers_out']]/sum_transfers
                            
                        # selected = gw.name == 'Patrick_van Aanholt'
                        # if (sum(selected)) > 0:
                        #     if gw[selected].opponent_team.values[0] == 4:
                        #         print(gw_csv)
    
                        if gw.shape[0] == 0:
                            print(gw_csv, 'is empty')
                        else:
                            print(gw_csv)
                            dfs_gw.append(gw)
                            
                    if sum_transfers == 0:
                        print(gw_csv, 'no transfers')
                        
                #FINISHED LOOPING GAMEWWEKS
                df_gw = pd.concat(dfs_gw)
    
                df_gw['kickoff_time'] =  pd.to_datetime(df_gw['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
                df_gw = df_gw.sort_values(by='kickoff_time')
    
                df_gw.reset_index(inplace=True)
    
    
                #variables I calculate myself
                if not json_file:
                    df_gw['string_opp_team'] = None
                 
                df_gw['points_per_game'] = np.nan       
                df_gw['points_per_played_game'] = np.nan
    
                # Calculate values on my own
                for player in df_gw['element'].unique():
    
                    selected_ind = df_gw['element'] == player
                    player_df = df_gw[selected_ind]
                    player_df.set_index('kickoff_time', inplace=True)
                    
                    
                    opp_team = []
                    
                    if not json_file:
                        for team in player_df['opponent_team'].astype(int).values-1:
                            opp_team.append(string_names[team])
                    
                            
                        # own_team = []
                        # for fix in player_df['fixture'].astype(int).values:
                        #     sel_fix = fixture_df.fixture == fix
                        #     if was_home:                      
                        #         own_team.append(fixture_df[sel_fix].team_h.values[0]-1)
                        #     else:
                        #         own_team.append(fixture_df[sel_fix].team_a.values[0]-1)
        
                        df_gw.loc[selected_ind, 'string_opp_team'] = opp_team.copy() 
                        #df_gw.loc[selected_ind, 'string_team'] = own_team.copy() 
    
                    
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
        
                    
                    
                season_df = df_gw[['name', 'minutes', 'string_opp_team', 'transfers_in', 'transfers_out', 'ict_index', 'influence', 'threat', 'creativity', 'bps', 'element', 'fixture', 'total_points', 'round', 'was_home', 'kickoff_time', 'expected_goals', 'expected_assists', 'expected_goal_assists', 'expected_goals_conceded', 'SoT', 'defcon', 'points_per_game', 'points_per_played_game']]#, 'own_team_points', 'own_wins', 'own_element_points']]
                
                if  folder == '2016-17' or folder == '2017-18':
                    season_df[["team_a_difficulty", "team_h_difficulty"]] = np.nan
                else:
                    if not json_file:
                        #get fixture difficulty difference for each datapoint
                        fixture_df = pd.read_csv(fixture_csv)
                    else:
                        filename = fr'C:\Users\jorgels\Git\Fantasy-Premier-League\data\2025-26\fixtures\{gw_num}_2025-26.json'
                        with open(filename, 'r') as json_file:
                            fixture_json = json.load(json_file)  # Load the data from the file
                            fixture_df = pd.DataFrame(fixture_json)
                        
                    #rename befor merge
                    fixture_df = fixture_df.rename(columns={"id": "fixture"})
                    season_df = pd.merge(season_df, fixture_df[["team_a_difficulty", "team_h_difficulty", "fixture"]], on='fixture')
        
                season_df = pd.merge(season_df, df_player[["element_type", "web_name", "string_team", "element"]], on="element")
    
                season_df["season"] = folder
                
                # for (ind, f) in np.unique(season_df['fixture']):
                #     selected_f = season_df['fixture'] == f
                    
                
                # #apply correct club for those who has transferred
                season_df = season_df.groupby(['fixture', 'string_opp_team'], group_keys=False).apply(correct_string_team)
                
                # def get_majority(series):
                #     return series.mode()[0]  # mode() returns the most common value
                #df['string_team'] = season_df.groupby(['fixture', 'string_opp_team'])['string_team'].transform(get_majority)
                print(len(season_df))
    
    
                season_dfs.append(season_df)
                
            
    
    season_df = pd.concat(season_dfs)
    season_df['transfers_in'] = season_df['transfers_in'].astype(float)
    season_df['transfers_out'] = season_df['transfers_out'].astype(float)
    season_df['points_per_game'] = season_df['points_per_game'].astype(float)
    
    #season_df['names'] = season_df['first_name'] + ' ' + season_df['second_name']
    
    
    #insert fantasy names in names.
    fantasy_names = static_df.first_name + ' ' + static_df.second_name
    selected_season = season_df.season == '2025-26'
    matched_inds = []
    
    for name_string in np.unique(season_df.loc[selected_season, 'name']):
                    
    
    
        if name_string in fbref_manual_names:
            name_selected = [k == name_string for k in fbref_manual_names]
            manual_ind =  np.where(name_selected)[0][0]
            closest_match = [fantasy_manual_names[manual_ind]]
        else:                           
            results = [sequence_matcher_similarity(fantasy_name, name_string) for fantasy_name in fantasy_names.values]
            max_val = -1
            for ind, k in enumerate(results):
                if k[0] > max_val:
                    max_val =  k[0]
                    max_ind = ind
            
            if max_val > 0.77:
                closest_match = [fantasy_names.values[max_ind]]
            else:
                closest_match = difflib.get_close_matches(name_string, fantasy_names.values, n=1)
            
        
            
    
        selected = season_df.name == name_string
        if not name_string == closest_match[0]:
            
            print(name_string, ' - ', closest_match[0], sum(selected))
            
            season_df.loc[selected, 'name'] = closest_match[0]
        
        if sum(selected) == 0:
            print('No matches for', name_string)
            continue
            
        matched_ind = np.where(closest_match[0] == fantasy_names.values)[0][0]
        if matched_ind in matched_inds:
            print('Double matched for', name_string)
        matched_inds.append(matched_ind)
    
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
    
    # own_keys = ['ict_index', 'influence', 'threat', 'creativity', 'bps', 'total_points', 'xP',
    #        'expected_goals', 'expected_assists', 'expected_goal_involvements',
    #        'expected_goals_conceded']
    
    # selected = season_df.minutes == 0
    # season_df.loc[selected, own_keys] = np.nan    
    
    
    # #match names to online names!
    # elements_df = pd.DataFrame(js['elements'])
    # current_names = (elements_df['first_name'] + ' ' + elements_df['second_name']).unique()
    # current_positions = elements_df['element_type']
    
    # #current from online. name position from historical data
    # names = np.concatenate((current_names, name_position_list['names'][::-1]))
    # positions =  np.concatenate((current_positions, name_position_list['element_type'][::-1]))
    
    # _, indices = np.unique(names, return_index=True)
    # sorted_indices = np.sort(indices)
    # all_names = names[sorted_indices]
    # all_positions = positions[sorted_indices]
    
    
    
    # #make list that keep tracks of the changed names
    # new_names = all_names.copy()
    
    # #not that dangerous to merge previous players, but avoid to merge into current player
    # #loop through the most recent players first
    # for name_ind, name in enumerate(all_names[:-1]):
    #     # if 'Matheus' in name:
    #     #     print(name, name_ind)
    
    #     #where in list to check to avoid merges in the same season
    #     check_ind = np.max([len(current_names), name_ind+1])
    
    #     previous_names = all_names[check_ind:]
    
    #     results = [sequence_matcher_similarity(prev_name, name) for prev_name in previous_names]
    
    #     # Now unpack the results list into separate variables
    #     similarity_scores = []
    #     first_name_similarities = []
    #     second_name_similarities = []
    
    #     for result in results:
    #         similarity_score, first_name_similarity, second_name_similarity = result
    #         similarity_scores.append(similarity_score)
    #         first_name_similarities.append(first_name_similarity)
    #         second_name_similarities.append(second_name_similarity)
    
    #     max_match = np.argmax(similarity_scores)
    #     matched_name = previous_names[max_match]
    
    
    #     match_ind = -1
    
    #     first_name_criteria = (np.array(similarity_scores) > 0.71) & (np.array(first_name_similarities) > 0.47)
    #     second_name_criteria = (np.array(similarity_scores) > 0.70) & (np.array(second_name_similarities) > 0.6)
    #     all_criteria = (np.array(similarity_scores) > 0.56) & (np.array(first_name_similarities) > 0.55) & (np.array(second_name_similarities) > 0.67)
    #     test_criteria = max(similarity_scores) > 1
    
    #     print_test = False
    
    #     if (matched_name in name or name in matched_name) or max(similarity_scores) > 0.7:
    #         match_ind = np.argmax(similarity_scores)
    #     elif any(first_name_criteria):
    #         match_ind = np.where(first_name_criteria)[0][0]
    #     elif any(second_name_criteria):
    #         match_ind = np.where(second_name_criteria)[0][0]
    #     elif any(all_criteria):
    #         match_ind = np.where(all_criteria)[0][0]
    #     elif test_criteria:
    #         match_ind = np.argmax(similarity_scores)
    #         print_test = True
    
    #     if match_ind > -1:
    
    #         matched_name = previous_names[match_ind]
    
    #         change_names = season_df['names'] == matched_name
    
    #         matched_position = season_df.loc[change_names, 'element_type'].unique()
    
    #         root_position = all_positions[name_ind]
    
    #         if any(matched_position == root_position):
    #             new_name = new_names[name_ind]
    
    #             do_not_match_names = [['David Martin', 'David Raya Martin'],
    #                                   ['Caleb Taylor', 'Charlie Taylor'],
    #                                   ['Solomon March', 'Manor Solomon'],
    #                                   ['Michael Olise', 'Michael Olakigbe'],
    #                                   ['Ryan Bennett', 'Rhys Bennett'],
    #                                   ['Joe Powell', 'Joe Rothwell'],
    #                                   ['Ashley Williams', 'Ashley Phillips'],
    #                                   ['Aaron Ramsey', 'Jacob Ramsey'],
    #                                   ['Lewis Richards', 'Chris Richards'],
    #                                   ['Ashley Williams', 'Rhys Williams'],
    #                                   ['Killian Phillips', 'Kalvin Phillips'],
    #                                   ['Josh Murphy', 'Jacob Murphy'],
    #                                   ['Matthew Longstaff', 'Sean Longstaff'],
    #                                   ['Charlie Cresswell', 'Aaron Cresswell'],
    #                                   ['Dale Taylor', 'Joe Taylor'],
    #                                   ['Jackson Smith', 'Jordan Smith'],
    #                                   ['Kayne Ramsay', 'Calvin Ramsay'],
    #                                   ['Haydon Roberts', 'Connor Roberts'],
    #                                   ['Mason Greenwood', 'Sam Greenwood'],
    #                                   ['Joe Bryan', 'Kean Bryan'],
    #                                   ['Lewis Gibson', 'Liam Gibson'],
    #                                   ['Daniel Sturridge', 'Sam Surridge'],
    #                                   ['Alexis Sánchez', 'Carlos Sánchez'],
    #                                   ['Danny Simpson', 'Jack Simpson'],
    #                                   ['Bakary Sako', 'Bukayo Saka'],
    #                                   ['James Tomkins', 'Jake Vokins'],
    #                                   ['Lewis Brunt', 'Lewis Dunk'],
    #                                   ['James Tomkins', 'James Tarkowski'],
    #                                   ['Ben Jackson', 'Ben Johnson'],
    #                                   ['Tyler Roberts', 'Tyler Morton'],
    #                                   ['James McArthur', 'James McAtee'],
    #                                   ['Josh Brownhill', 'Josh Bowler'],
    #                                   ['Andy King', 'Andy Irving'],
    #                                   ['Joshua Sims', 'Joshua King'],
    #                                   ['James Storer', 'James Shea'],
    #                                   ['Owen Beck', 'Owen Bevan'],
    #                                   ['Joseph Hungbo' , 'Joe Hodge'],
    #                                   ['Jonathan Leko', 'Jonathan Rowe'],
    #                                   [' Christian Fuchs', 'Christian Marques'],
    #                                   ['Anthony Martial', 'Anthony Mancini'],
    #                                   ['Jack Simpson', 'Jack Robinson'],
    #                                   ['Jack Cork', 'Jack Colback'],
    #                                   ['Simon Mignolet', 'Simon Moore'],
    #                                   ['Aaron Ramsey', 'Aaron Rowe'],
    #                                   ['Antonio Valencia', 'Antonio Barreca'],
    #                                   ['Callum Paterson', 'Callum Slattery'],
    #                                   ['Ben Wilmot', 'Benjamin White'],
    #                                   ['Scott Dann', 'Scott Malone'],
    #                                   ['Sergio Romero', 'Sergio Rico'],
    #                                   ['James Daly', 'Jamie Donley'],
    #                                   ['Jason Puncheon', 'Jadon Sancho'],
    #                                   ['Killian Phillips', 'Philip Billing'],
    #                                   ['Christian Saydee', 'Christian Nørgaard'],
    #                                   ['James Sweet', 'Reece James'],
    #                                   ['Ollie Harrison', 'Harrison Reed'],
    #                                   ['Richard Nartey', 'Omar Richards'],
    #                                   ['Charles Sagoe', 'Shea Charles'],
    #                                   ['Benjamin Mendy', 'Benjamin Fredrick'],
    #                                   ['Scott Dann', 'Dan Potts'],
    #                                   ['Ashley Williams', 'Neco Williams'],
    #                                   ['Matty James', 'James McCarthy'],
    #                                   ['Ashley Williams', 'William Fish'],
    #                                   ['Christian Fuchs', 'Christian Marques'],
    #                                   ['Charlie Savage', 'Charles Sagoe'],
    #                                   ['Andrew Surman', 'Andrew Moran'],
    #                                   ['Niels Nkounkou', 'Nicolas Nkoulou'],
    #                                   ['Christian Marques changed', 'Cristhian Mosquera'],
    #                                   ['Joe Allen', 'Josh Cullen'],
    #                                   ['Alex Palmer', 'Alex Paulsen'],
    #                                   ['Kyle Scott', 'Alex Scott'],
    #                                   ['Daniel Agyei', 'Daniel Adu-Adjei'],
    #                                   ['Michael Dawson', 'Michael Kayode'],
    #                                   ['Louie Watson', 'Tom Watson'],
    #                                   ['Mamadou Sakho', 'Mamadou Sarr'],
    #                                   ['Adam Clayton', 'Adam Wharton'],
    #                                   ['Michael Hefele', 'Michael Keane'],
    #                                   ['Stuart Armstrong', 'Harrison Armstrong'],
    #                                   ['Josh Robson', 'Joe Rodon'],
    #                                   ['Conor Coady', 'Conor Bradley'],
    #                                   ['Jamie McDonnell', 'James McConnell'],
    #                                   ['Jamal Lewis', 'Lewis Hall'], 
    #                                   ['Kieran Tierney', 'Kieran Trippier'],
    #                                   ['Ibrahim Osman', 'Ibrahim Sangaré'],
    #                                   ['Daniel Ayala', 'Daniel Ballard'],
    #                                   ['Zak Swanson', 'Zak Johnson'],
    #                                   ['Ollie Harrison', 'Harrison Jones'],
    #                                   ['Matthew Daly', 'Jay Matete'],
    #                                   ['Cristian Gamboa', 'Cristian Romero'],
    #                                   ['Mike van der Hoorn', 'Micky van de Ven'],
    #                                   ['Ben Johnson', 'Brennan Johnson'],
    #                                   ['James Morrison', 'James Maddison'],
    #                                   ['Rodrigo Hernandez', 'Rodrigo Bentancur'],
    #                                   ['Leiva Lucas', 'Lucas Bergvall'],
    #                                   ['Maximillian Aarons', 'Maximilian Kilman'],
    #                                   ['Callum Robinson', 'Callum Wilson'],
    #                                   ['Alfie Jones', 'Alfie Pond'],
    #                                   ['Ben Watson', 'Tom Watson'],
    #                                   ['David Martin', 'David Raya Martín'],
    #                                   ['Christian Fuchs', 'Cristhian Mosquera'],
    #                                   ['Matthew James', 'Jay Matete'],
    #                                   ['Glen Johnson', 'Brennan Johnson'],
    #                                   ['Jefferson Montero', 'Jefferson Lerma Solís'],
    #                                   ['Michael Ledger', 'Michael Keane'],
    #                                   ['Ander Herrera', 'Andreas Hoelgebaum Pereira'],
    #                                   ['Christian Marques', 'Cristhian Mosquera']
    #                                   ]
    
    #             continue_marker = False
    #             for avoid_match in do_not_match_names:
    #                 if matched_name in avoid_match and new_name in avoid_match:
    #                     continue_marker = True
    
    #             if continue_marker:
    #                 continue
    
    #             season_df.loc[change_names, 'names'] = new_name
    
    #             matched_index = all_names == matched_name
    #             new_names[matched_index] = new_name
    
    #             if new_name in current_names:
    #                 print(name_ind, matched_name + ' changed to ' + new_name)
    
    #             if print_test:
    #                 print(name_ind, matched_name + ' changed to ' + new_name, similarity_scores[match_ind], first_name_similarities[match_ind], second_name_similarities[match_ind])
    
    # print('Done matching')
    
    
    
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
    season_df.to_csv(r'C:\Users\jorgels\Git\Fantasy-Premier-League\data\all_historical_data.csv', index=False)  # Set index=False to not include row indices


    
season_df = pd.read_csv(r'C:\Users\jorgels\Git\Fantasy-Premier-League\data\all_historical_data.csv')

#ALL VARIABLES:
#how variables are includedvariables
#always included. also for current weak
dynamic_features = ['string_opp_team', 'transfers_in', 'transfers_out',
        'was_home', 'own_difficulty', 'other_difficulty']#, 'difficulty']

#features that I don't have access to in advance.
#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goal_assists', 'expected_goals_conceded', 'own_team_points', 'own_element_points', 'SoT', 'defcon']
#included once
temporal_single_features = ['points_per_game', 'points_per_played_game']
#total_points, minutes, kickoff time not for prediction
#included once
fixed_features = ['total_points', 'minutes', 'kickoff_time', 'element_type', 'string_team', 'season', 'name']

#categories for dtype
categorical_variables = ['element_type', 'string_team', 'season', 'name']
season_df[categorical_variables] = season_df[categorical_variables].astype('category')
#add nan categories
dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
        'other_difficulty'] #'difficulty',

int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon', 'SoT']
season_df[int_variables] = season_df[int_variables].astype('Int64')

float_variables = ['transfers_in', 'transfers_out', 'threat', 'own_element_points',  'expected_goals', 'expected_assists',
'expected_goal_assists', 'expected_goals_conceded', 'creativity', 'ict_index', 'influence']
season_df[float_variables] = season_df[float_variables].astype('float')



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
    

    for name in season_df.name.unique():
        
        selected_ind = season_df.name == name
        
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
    
            
train_data = pd.concat([train, temp_train], axis=1)            


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
 
train_data.to_pickle(r'C:\Users\jorgels\Git\Fantasy-Premier-League\models\model_data.pkl')  # Set index=False to not include row indices
