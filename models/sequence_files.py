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


#old office PC
directories = r'C:\Users\jorgels\GitHub\Fantasy-Premier-League\data'
try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\GitHub\Fantasy-Premier-League'
    
    
#laptop
except:
    directories = r'C:\Users\jorgels\Git\Fantasy-Premier-League\data'
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


build_from_scratch = False
check_last_data = True

temporal_window = 11
    

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
    #cleaned_string = cleaned_string.replace("'", "")
    # Remove all numbers
    cleaned_string = re.sub(r'\d+', '', cleaned_string)
    return cleaned_string.strip()  # Optional: strip leading/trailing spaces

static_path = os.path.join(directories, '2025-26\static_2025-26.json')

if check_last_data:
    
    print('Scrap static game info')
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    static = r.json()
    static_df = pd.DataFrame(static['elements'])
        

    #scrap fantasy data:
    #get current names
    #get statistics of all players

    
    # Save to a JSON file
    with open(static_path, 'w') as json_file:
        json.dump(static, json_file)
    
    #loop players
    #store to only scrap once
    players = []
    print('Scrap player info')
    for player_id in static_df.iterrows():
        
        if player_id[1].minutes == 0:
            continue
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
                
                filename = os.path.join(directories, f'2025-26\players\{player_id[1].id}_2025-26.json')
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
        filename = os.path.join(directories, f'2025-26\\fixtures\\{i}_2025-26.json') 

        with open(filename , 'w') as json_file:
            json.dump(gw, json_file)
            
        gw_data = []
        #create gw data:
        #loop the scrapped players
        for player in players:
            
            player_id = player["id"]
            selected = static_df.id == player_id
            
            if static_df.loc[selected, 'minutes'].values == 0:
                continue

            for history in player["history"]:
                
                if history["round"] == i:
                        
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
                            'first_name': static_df.loc[selected_static]["first_name"].iloc[0],  
                            'second_name': static_df.loc[selected_static]["second_name"].iloc[0],
                            'web_name': static_df.loc[selected_static]["web_name"].iloc[0],
                            'string_team': string_team,
                            'season': '2025-26',
                            'name': static_df.loc[selected_static ]["first_name"].iloc[0] + ' ' + static_df.loc[selected_static ]["second_name"].iloc[0],
                            'clearances_blocks_interceptions': history['clearances_blocks_interceptions'],
                            'recoveries': history['recoveries'],
                            'tackles': history['tackles'],
                            'saves': history["saves"],
                        }
                    
                    gw_data.append(new_row)
        if gw_data:  
            gw_df = pd.DataFrame(gw_data)
            filename = os.path.join(directories, f'2025-26\\gws\\gw{i}.json')
            gw_df.to_json(filename)
                    
            
            
            
season_filename = os.path.join(directories, 'all_historical_data.csv')
if build_from_scratch:
    
    season_dfs = [] 
    
else:
    season_dfs = [pd.read_csv(season_filename)]
        
        
#get each previous season
for folder in folders:
    
    #get data from vastaav
    directory = os.path.join(directories, folder)
    fixture_csv = os.path.join(directory, 'fixtures.csv')
    gws_data = os.path.join(directory, 'gws')
    team_path = os.path.join(directory, "teams.csv")

    #if os.path.isfile(fixture_data_fbref) and  os.path.isdir(gws_data):
    if os.path.isdir(gws_data):

        #check that it is not a file
        if folder[-4] != '.':

            print('\n', folder)   
            
            if folder in ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']:
                #I have scrapped data from these folders
                season_data = pd.read_csv(directory + '\\fbref/' + folder[:-2] + '20' + folder[-2:] + '_player_data.csv')
                fixture_data_fbref = pd.read_csv(directory + '\\fbref/'  + folder[:-2] + '20' + folder[-2:] + '_fixture_data.csv')
                
                #correct assign  game_id. Assume they are ordered
                for id, ind in enumerate(np.unique(fixture_data_fbref.game_id)):
                    selected_id = fixture_data_fbref.game_id == ind
                    fixture_data_fbref.loc[selected_id, 'game_id'] = id
                
                #fixture_data_fbref.loc[selected_id, 'fantasy_id'] = None
                
                fixture_data_fbref['kickoff_time'] = pd.to_datetime(fixture_data_fbref['Date'] + ' ' + fixture_data_fbref['Time'])
                   
                season_data = pd.merge(season_data, fixture_data_fbref.loc[:, ['Wk', 'game_id', 'kickoff_time']], on='game_id', how='left')
                season_data.rename(columns={'Wk': 'gameweek'}, inplace=True)
                
                fbref_names = season_data.Player.to_list()
            
                print('Season_data range from', min(season_data.gameweek), max(season_data.gameweek))
            else:
                
                manual_player_data = pd.read_csv(directory + '\\fbref\\' + folder[:-2] + '20' + folder[-2:] +  '_manually_scrapped_players.csv', sep=';', header=[0])
                fbref_names = manual_player_data.Player.to_list()

            #get id so it can be matched with position
            try: 
                player_path = directory + f'/static_{folder}.json'
                with open(player_path, 'r') as json_file:
                    static_data = json.load(json_file)  # Load the data from the file
                
                # Convert the loaded data to a DataFrame
                df_player = pd.DataFrame(static_data["elements"])
                
                string_names = np.array([t["short_name"] for t in static_data["teams"]])
                
                is_json_file = True
            
            except:
                player_path = directory + '/players_raw.csv'
                df_player = pd.read_csv(player_path)
                
                #insert string for team
                df_teams = pd.read_csv(team_path)
                
                string_names = df_teams["short_name"].values
                
                is_json_file = False
            
            
            if not (folder == '2016-17' or folder == '2017-18'):
                #we need to get dfficulties in and string team!
                if not is_json_file:
                    #get fixture difficulty difference for each datapoint
                    fixture_data_fantasy = pd.read_csv(fixture_csv)
                        
                    fixture_data_fantasy = fixture_data_fantasy.rename(columns={"id": "fixture"}) 
                else:
                    #load below every gameweek
                    fixture_data_fantasy  = pd.DataFrame()

            #rename befor merge
            df_player = df_player.rename(columns={"id": "element"}) 
            #remove players that haven't played
            df_player = df_player.rename(columns={"minutes": "season_minutes"})  
            df_player["string_team"] = string_names[df_player["team"]-1] 
            
            

            dfs_gw = []
            
            #gw 7 of 2022 is empty
            
            updated_player_names = {}
            
            

            #open each gw and get data for players
            for gw_csv in os.listdir(directory + '/gws'):
                if gw_csv[0] == 'g':

                    gw_path = directory + '/gws' + '/' + gw_csv

                    if not is_json_file:
                        if folder == '2018-19' or folder == '2016-17' or folder == '2017-18':
                            gw = pd.read_csv(gw_path, encoding='latin1')
                        else:
                            gw = pd.read_csv(gw_path)   
                        
                    else:
                        
                        gw = pd.read_json(gw_path)
                        
                    
                        
                    #skip if any of the fixtures are in the seasons df
                    if not build_from_scratch:
                        #empty corone disrupts stuff
                        if len(gw) > 0:
                            if sum((season_dfs[0].season == folder) & (season_dfs[0].fixture == gw.loc[0, 'fixture'])) > 0:
                                continue
                            
                            
                            
                    gw_num = int(re.findall(r'\d+', gw_csv)[0])
                    
                    if not (folder == '2016-17' or folder == '2017-18'):
                        #we need to get dfficulties in and string team!
                        if is_json_file:
                            filename = directories + fr'\\2025-26\fixtures\{gw_num}_2025-26.json'
                            with open(filename, 'r') as json_file:
                                fixture_json = json.load(json_file)  # Load the data from the file
                            fixture_data_gw = pd.DataFrame(fixture_json)
                            
                            fixture_data_gw = fixture_data_gw.rename(columns={"id": "fixture"}) 
                                
                            fixture_data_fantasy = pd.concat((fixture_data_fantasy, fixture_data_gw))
                    
                    #allocate rows
                    if folder in ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']:
                        gw['expected_goals'] = np.nan
                        gw['expected_assists'] = np.nan
                        gw['expected_goals_conceded'] = np.nan
                    
                    #gw['expected_goal_assists'] = np.nan
                    
                    #gw['SoT'] = np.nan
                    
                      
                    gw = pd.merge(gw, df_player.loc[:, ['season_minutes', 'element', 'element_type']], on='element')
                    gw = gw.loc[gw.season_minutes > 0].reset_index(drop=True)
                    
                    #add cbirts to those where we do not have fantasy statistics
                    if folder in ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']:  
                        #enter later from fbfref
                        gw['defcon'] = np.nan

                        
                        
                    else:
                            
                        gw['defcon'] = 0
                        
                        defenders = gw.element_type == 2
                        gw.loc[defenders, 'defcon'] = gw.loc[defenders, ['clearances_blocks_interceptions', 'tackles']].sum(axis=1)

                        attackers = (gw.element_type == 3) | (gw.element_type == 4)                    
                        gw.loc[attackers, 'defcon'] = gw.loc[attackers, ['clearances_blocks_interceptions', 'tackles', 'recoveries']].sum(axis=1)
                        
                        gk = (gw.element_type == 1)          
                        gw.loc[gk, 'defcon'] = gw.loc[gk, 'saves']
                         
                        if folder in ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']:
                        
                          add_points =  (gw.loc[defenders, 'defcon'] > 10) * 2
                          gw.loc[defenders, 'total_points'] += add_points
                          
                          add_points =  (gw.loc[attackers, 'defcon'] > 12) * 2
                          gw.loc[attackers, 'total_points'] += add_points
                        
                    
                    #manually change some names:
                    fantasy_manual_names = ['Pablo Felipe Pereira de Jesus', 'John Victor Maciel Furtado', 'Lucas Pires Silva', 'Jair Paula da Cunha Filho', 'Yéremy Pino Santos', 'Manuel Ugarte Ribeiro', 'Sávio Moreira de Oliveira', 'Kevin Santos Lopes de Macedo', 'Mateus Gonçalo Espanha Fernandes', 'Nico González Iglesias', 'Marc Guiu Paz', 'Marcus Oliveira Alencar', "Rodrigo 'Rodri' Hernandez Cascante", 'Rodrigo Moreno', 'Josh Sims', 'Bernardo Mota Veiga de Carvalho e Silva', 'Andrey Nascimento dos Santos', 'Alisson Becker', 'João Maria Lobo Alves Palhares Costa Palhinha Gonçalves', 'Igor Jesus Maciel da Cruz', 'João Pedro Ferreira da Silva', 'Murillo Costa dos Santos', 'Matheus Santos Carneiro da Cunha', 'Rúben dos Santos Gato Alves Dias', 'Estêvão Almeida de Oliveira Gonçalves', "Rodrigo 'Rodri' Hernandez", 'Vitor de Oliveira Nunes dos Reis', 'Welington Damascena Santos', 'Felipe Rodrigues da Silva', 'André Trindade da Costa Neto', 'Francisco Evanilson de Lima Barbosa', 'João Pedro Ferreira Silva', 'Igor Thiago Nascimento Rodrigues', "Sávio 'Savinho' Moreira de Oliveira", 'Norberto Bercique Gomes Betuncal', 'Anssumane Fati Vieira', 'Victor da Silva', 'Manuel Benson Hedilazio', 'Igor Julio dos Santos de Paulo', 'Murillo Santiago Costa dos Santos', 'Felipe Augusto de Almeida Monteiro', 'Mateus Cardoso Lemos Martins', 'João Victor Gomes da Silva', 'Danilo dos Santos de Oliveira', 'Alexandre Moreno Lopera', 'Carlos Ribeiro Dias', 'Matheus Santos Carneiro Da Cunha', 'Renan Augusto Lodi dos Santos', 'Lyanco Silveira Neves Vojnovic', 'Willian Borges da Silva', 'Carlos Henrique Casimiro', 'Norberto Murara Neto', 'Antony Matheus dos Santos', 'Lucas Tolentino Coelho de Lima', 'Diogo Teixeira da Silva','Fábio Freitas Gouveia Carvalho', 'Emerson Leite de Souza Junior', 'Bernardo Veiga de Carvalho e Silva', 'Francisco Jorge Tomás Oliveira', 'Samir Caetano de Souza Santos', 'Lyanco Evangelista Silveira Neves Vojnovic', 'Emerson Aparecido Leite de Souza Junior', 'Juan Camilo Hernández Suárez', 'José Malheiro de Sá', 'Francisco Machado Mota de Castro Trincão', 'Francisco Casilla Cortés', 'Rúben Santos Gato Alves Dias', 'Raphael Dias Belloli', 'Vitor Ferreira', 'Oluwasemilogo Adesewo Ibidapo Ajayi', 'Ivan Ricardo Neves Abreu Cavaleiro', 'Hélder Wander Sousa de Azevedo e Costa', 'Allan Marques Loureiro', 'Bruno André Cavaco Jordao', 'João Pedro Junqueira de Jesus', 'Borja González Tomás', 'José Reina', 'Roberto Jimenez Gago', 'José Ángel Esmorís Tasende', 'Rodrigo Hernandez', 'Mahmoud Ahmed Ibrahim Hassan', 'José Ignacio Peleteiro Romallo', 'Joelinton Cássio Apolinário de Lira', 'Alexandre Nascimento Costa Silva', 'Fabio Henrique Tavares', 'Bernard Anício Caldeira Duarte', 'André Filipe Tavares Gomes', 'André Filipe Tavares', 'Rúben Gonçalo Silva Nascimento Vinagre', 'Rúben Diogo da Silva Neves', 'Rui Pedro dos Santos Patrício', 'João Filipe Iria Santos Moutinho', 'Jorge Luiz Frello Filho', 'Frederico Rodrigues de Paula Santos', 'Fabricio Agosto Ramírez', 'Bonatini Lohner Maia Bonatini', 'Bernardo Fernandes da Silva Junior', 'Alisson Ramses Becker', 'Olayinka Fredrick Oladotun Ladapo', 'Lucas Rodrigues Moura da Silva', 'João Mário Naval Costa Eduardo', 'Adrien Sebastian Perruchet Silva', 'Danilo Luiz da Silva', 'Jesé Rodríguez Ruiz', 'Jose Luis Mato Sanmartín', 'Ederson Santana de Moraes', 'Bruno Saltor Grau', 'Bernardo Mota Veiga de Carvalho e Silva', 'Robert Kenedy Nunes do Nascimento', 'Gabriel Armando de Abreu', 'Fabio Pereira da Silva', 'Fernando Francisco Reges', 'David Luiz Moreira Marinho', 'Willian Borges Da Silva', 'Pedro Rodríguez Ledesma', 'Oscar dos Santos Emboaba Junior', 'Manuel Agudo Durán', 'Fernando Luiz Rosa', 'Adrián San Miguel del Castillo']
                    fbref_manual_names = ['Pablo', 'John', 'Lucas', 'Jair Cunha', 'Yeremi Pino', 'Manuel Ugarte', 'Sávio', 'Kevin', 'Mateus Fernandes', 'Nicolás González', 'Marc Guiu', 'Marquinhos', 'Rodri', 'Rodrigo', 'Josh Sims', 'Bernardo Silva', 'Andrey Santos', 'Alisson', 'João Palhinha', 'Igor Jesus', 'Jota Silva', 'Murillo', 'Matheus Cunha', 'Rúben Dias', 'Estêvão Willian', 'Rodri', 'Vitor Reis', 'Welington', 'Morato', 'André', 'Evanilson', 'Jota Silva', 'Thiago', 'Sávio', 'Beto', 'Ansu Fati', 'Vitinho', 'Benson Manuel', 'Igor', 'Murillo', 'Felipe', 'Tetê', 'João Gomes', 'Danilo', 'Álex Moreno', 'Cafú', 'Matheus Cunha', 'Renan Lodi', 'Lyanco', 'Willian', 'Casemiro', 'Neto', 'Antony', 'Lucas Paquetá', 'Diogo Jota', 'Fabio Carvalho', 'Emerson', 'Bernardo Silva', 'Chiquinho', 'Samir Santos', 'Lyanco', 'Emerson', 'Cucho', 'José Sá', 'Francisco Trincão', 'Kiko Casilla', 'Rúben Dias', 'Raphinha', 'Vitinha', 'Semi Ajayi', 'Ivan Cavaleiro', 'Hélder Costa', 'Allan', 'Bruno Jordão', 'João Pedro', 'Borja Bastón', 'Pepe Reina', 'Roberto', 'Angeliño', 'Rodri', 'Trézéguet', 'Jota', 'Joelinton', 'Xande Silva', 'Fabinho', 'Bernard', 'André Gomes', 'André Gomes', 'Rúben Vinagre', 'Rúben Neves', 'Rui Patrício', 'João Moutinho', 'Jorginho', 'Fred', 'Fabricio', 'Léo Bonatini', 'Bernardo', 'Alisson', 'Freddie Ladapo', 'Lucas Moura', 'João Mário', 'Adrien Silva', 'Danilo', 'Jesé', 'Joselu', 'Ederson', 'Bruno', 'Bernardo Silva', 'Kenedy', 'Gabriel Paulista', 'Fábio', 'Fernando', 'David Luiz', 'Willian', 'Pedro', 'Oscar', 'Nolito', 'Fernandinho', 'Adrián']
                
                    #covid weeks
                    if gw_num > 38:
                        gw_num = gw_num - 9
                        
    
                                  
                    for el in gw.iterrows():
                        
                        # if 'William Saliba' in  el[1]['name']:
                        #     a=dhdhdhdhd
                        #     print(el[0], el[1].element, el[1]['name'])
                        element = el[1].element
                        el_selected =  df_player.element == element
                        
                        #skip if haven't played for the whole season
                        # if df_player.loc[el_selected, 'season_minutes'].iloc[0] == 0:
                        #     continue
                        
                        name_string = clean_string(el[1]['name']) 
                        
                        if name_string in updated_player_names:
                            closest_match = [updated_player_names[name_string]]
                            #print('Previous matching')
                        
                        elif name_string in fantasy_manual_names:
                            #print('List matching')
                            
                            index = fantasy_manual_names.index(name_string)                        
                            closest_match = [fbref_manual_names[index]]
                            
                            updated_player_names[name_string] = closest_match[0]
                            #print('Manual matching')
                        else:
                            
                            results = [sequence_matcher_similarity(prev_name, name_string) for prev_name in fbref_names]
                            max_val = -1
                            for ind, k in enumerate(results):
                                if k[0] > max_val:
                                    max_val =  k[0]
                                    max_ind = ind
                            
                            if max_val > 0.77:
                                closest_match = [fbref_names[max_ind]]
                                #('Seq matching')
                            else:
                                closest_match = difflib.get_close_matches(name_string, fbref_names, n=1)
                                
                            #avoid mixup of historical players
                            if folder in ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']:
                                if closest_match[0] == 'Joshua King':
                                    selected =  season_data.Player == 'Joshua King'
                                    
                                    season_data.loc[selected, 'Player'] = 'XXJoshua King OldXX'
                                    fbref_names[max_ind] = 'XXJoshua King OldXX'
                                    closest_match = ['XXJoshua King OldXX']
                                    
                            #player_name change DURING season
                            
                            # if name_string == 'Jair Paula da Cunha Filho':
                            #     a = fjhkfhf
                            #     print(name_string)

                                
                            if not closest_match:
                                if el[1].minutes > 0:
                                    print('No player matched in fbref', el[1]['name'], name_string)  
                                continue
                                
                            # if not closest_match:
                            #     closest_match = [name_string]
                            #     print('No matches for', name_string)
                            
                            updated_player_names[name_string] = closest_match[0]
                            
                            #print('Automatic matching')
                        
                        
                        

                        
                        
                        #keep this before inventing new names.
                        num_player_selected = fbref_names.count(closest_match[0])
                        
                        if num_player_selected == 0 and el[1].minutes > 0:
                            print('Player played, but not matched', el[0], el[1]['name'], closest_match)
                            continue                 
                            
                        #use names from fbref. same name across seasons
                        gw.loc[el[0], 'name'] = closest_match[0]
    
                        if el[1].minutes == 0:
                            continue                             
                        
                        #for fbref statistics we need to identify the correct match
                        #for 2016-18 we don't have any fbref statistics. for 2025 and after we don't have fbref
                        if folder in ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']:
                            
                            #gameweek doesn't work for double rounds. use date instead
                            utc_time = pd.to_datetime(el[1].kickoff_time)   
                            try:
                                london_time = utc_time.tz_convert('Europe/London')
                            except: 
                                london_time = utc_time + pd.Timedelta(hours=1)
                            
                            player_selected = season_data.Player == closest_match[0]
                                
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
                            
                            if sum(fbref_selected) == 0:
                                print('No fbref games for player', name_string)
                                continue
                            # #also use kickofftime to acocmodate multiple matches
                            # if sum(fbref_selected) > 1:
                            #     utc_time = pd.to_datetime(el[1].kickoff_time)   
                            #     london_time = utc_time.tz_convert('Europe/London')
                            #     fbref_selected = (season_data.kickoff_time == london_time.tz_localize(None)) & (season_data.gameweek == gw_num) & (season_data.Player.values == closest_match)
                                
                            #check if there are two. then merge
                            elif sum(fbref_selected)  > 1:
                               
                                indices = season_data[fbref_selected].index
                                
                                print('Duplicate recordings for', name_string, indices, 'Merge!')
                                
                                
                                season_data.iloc[indices[0]] = season_data.iloc[indices[0]].fillna(season_data.iloc[indices[1]])
                                season_data.iloc[indices[1]] = season_data.iloc[indices[1]].fillna(season_data.iloc[indices[0]])
                                
                                if season_data.iloc[indices[0]].Min > season_data.iloc[indices[1]].Min:
                                    fbref_selected = season_data.index == indices[0]
                                else:
                                    fbref_selected = season_data.index == indices[1]
    
                    
                            #add cbirt points and cbirt data to previous seasons                            
                            
                            
                            
                            if folder in ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']:
                                
                                #find position
                                position = df_player.loc[el_selected].element_type.iloc[0]
                                
                                #defender
                                if position == 2:
                                    cbirt = season_data.loc[fbref_selected, ['Clr', 'Blocks',
                                    'Int', 'Tkl']].sum(axis=1).iloc[0]
                                    
                                    #error in data file :(
                                    # if cbirt > 23:
                                    #     cbirt = season_data.loc[fbref_selected, ['Blocks',
                                    #     'Int', 'Tkl']].sum(axis=1).iloc[0] /3*4
                                        
                                    
                                    if cbirt >= 10:
                                        gw.loc[el[0], 'total_points'] += 2
                                               
                                elif not position == 1:
                                    cbirt = season_data.loc[fbref_selected, ['Clr', 'Blocks',
                                    'Int', 'Tkl', 'Recov']].sum(axis=1).iloc[0]
                                    
                                    #error in data file :(
                                    # if cbirt > 23:
                                    #     cbirt = season_data.loc[fbref_selected, ['Blocks',
                                    #     'Int', 'Tkl']].sum(axis=1).iloc[0]/3*5
                                    
                                    if cbirt >= 12:
                                        gw.loc[el[0], 'total_points'] += 2
                                        
                                else: #keeper
                                    cbirt = season_data.loc[fbref_selected, ['Saves']].sum(axis=1).iloc[0]
                                    
                                #add defcon to the stats
                                gw.loc[el[0], 'defcon'] = cbirt.copy()   
                                    
                            #from fbref
                            if folder in ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22']:
                                xG = season_data.loc[fbref_selected, 'xG'].iloc[0]
                                xA = season_data.loc[fbref_selected, 'xA'].iloc[0]
                                xGC = season_data.loc[fbref_selected, 'xGC'].iloc[0]
                                
                                gw.loc[el[0], 'expected_goals'] = xG.copy()
                                #gw.loc[el[0], 'expected_goal_assists'] = season_data.loc[fbref_selected, 'xAG'].iloc[0]
                                gw.loc[el[0], 'expected_assists'] = xA.copy()
                            
                                #insert xGI
                                #the season data is only correct for players who played 90 min
                                if gw.loc[el[0]].minutes == 90:
                                    gw.loc[el[0], 'expected_goals_conceded']  = xGC.copy()
                            
                            #mulig denne kan gå ut hvis ikke i fantasy
                            #gw.loc[el[0], 'SoT'] = season_data.loc[fbref_selected, 'SoT'].iloc[0]
                        
                        
                    #check
                    # Check values
                    names, num_names = np.unique(gw['name'], return_counts=True)
                    
                    # Create a mask for game weeks with more than 1 entry
                    mask = num_names > 1
                    selected_names = names[mask]
                    selected_counts = num_names[mask]
                    
                    #double check there is a double fixture
                    for name, n in zip(selected_names, selected_counts):
                        
                        # Check if there are double game weeks
                        duplicate_players = gw.loc[gw['name'] == name, 'element']
                        
                        if len(np.unique(duplicate_players)) != 1:
                            print('Check double gameweek for', name, n,  duplicate_players.to_list())    
                            a = hfhfhfhh

                    
                        # # Extract all involved teams in one go using merge
                        # involved_fixtures = fixture_data_fantasy[fixture_data_fantasy['fixture'].isin(duplicate_fixtures)]
                        
                        # # Ensure there is only one unique team involved
                        # unique_teams = np.unique(involved_fixtures['team_h'].to_list() + involved_fixtures['team_a'].to_list())
                    
                        # if len(unique_teams) != 1 + n:
                        #     print(name, n, unique_teams)
                        
                    #remove assistant manager
                    if 'position' in gw.keys():
                        gw = gw.loc[gw['position'] != 'AM']
                    
                    sum_transfers = sum(gw.transfers_in) +  sum(np.abs(gw.transfers_out))
                    
                    if sum_transfers == 0:
                        print(gw_csv, 'no transfers')
                    
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
                        print(gw_csv, 'added')
                        dfs_gw.append(gw)
                        #print('Number of gw player games:', len(gw))
                        

            
            if dfs_gw:
                    
                #FINISHED LOOPING GAMEWWEKS
                df_gw = pd.concat(dfs_gw)
            
    
                df_gw['kickoff_time'] =  pd.to_datetime(df_gw['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
                df_gw = df_gw.sort_values(by='kickoff_time')
    
                df_gw.reset_index(inplace=True)
    
    
                #variables I calculate myself
                if not is_json_file:
                    df_gw['string_opp_team'] = None
                 
                df_gw['points_per_game'] = np.nan       
                df_gw['points_per_played_game'] = np.nan
    
                # Calculate values on my own
                for player in df_gw['element'].unique():
    
                    selected_ind = df_gw['element'] == player
                    player_df = df_gw[selected_ind]
                    player_df.set_index('kickoff_time', inplace=True)
                    
                    
                    opp_team = []
                    
                    if not is_json_file:
                        for team in player_df['opponent_team'].astype(int).values-1:
                            opp_team.append(string_names[team])
                    
                            
                        # own_team = []
                        # for fix in player_df['fixture'].astype(int).values:
                        #     sel_fix = fixture_data_fantasy.fixture == fix
                        #     if was_home:                      
                        #         own_team.append(fixture_data_fantasy[sel_fix].team_h.values[0]-1)
                        #     else:
                        #         own_team.append(fixture_data_fantasy[sel_fix].team_a.values[0]-1)
        
                        df_gw.loc[selected_ind, 'string_opp_team'] = opp_team.copy() 
                        #df_gw.loc[selected_ind, 'string_team'] = own_team.copy() 
    
                    
                    points_per_game =  player_df['total_points'].cumsum() / (player_df['round'])
                    
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
    
                    df_gw.loc[selected_ind, 'points_per_game'] = points_per_game.values.copy()
                    df_gw.loc[selected_ind, 'points_per_played_game'] = result[:-1].copy()
        
                    
                    
                season_df = df_gw[['name', 'minutes', 'string_opp_team', 'transfers_in', 'transfers_out', 'ict_index', 'influence', 'threat', 'creativity', 'bps', 'element', 'fixture', 'total_points', 'round', 'was_home', 'kickoff_time', 'expected_goals', 'expected_assists', 'expected_goals_conceded', 'defcon', 'points_per_game', 'points_per_played_game']]# 'SoT', 'expected_goal_assists' #, 'own_team_points', 'own_wins', 'own_element_points']]
                
                if  folder == '2016-17' or folder == '2017-18':
                    season_df[["team_a_difficulty", "team_h_difficulty"]] = np.nan
                else:
                    season_df = pd.merge(season_df, fixture_data_fantasy[["team_a_difficulty", "team_h_difficulty", "fixture"]], on='fixture')
        
                season_df = pd.merge(season_df, df_player[["element_type", "web_name", "string_team", "element"]], on="element")
    
                season_df["season"] = folder
                
                # for (ind, f) in np.unique(season_df['fixture']):
                #     selected_f = season_df['fixture'] == f
                    
                
                # #apply correct club for those who has transferred
                season_df = season_df.groupby(['fixture', 'string_opp_team'], group_keys=False).apply(correct_string_team)
                
                # def get_majority(series):
                #     return series.mode()[0]  # mode() returns the most common value
                #df['string_team'] = season_df.groupby(['fixture', 'string_opp_team'])['string_team'].transform(get_majority)
                print('Number of player games', len(season_df))
    
    
                season_dfs.append(season_df)
                
season_df = pd.concat(season_dfs)

season_df.to_csv(season_filename, index=False)  # Set index=False to not include row indices

season_df['transfers_in'] = season_df['transfers_in'].astype(float)
season_df['transfers_out'] = season_df['transfers_out'].astype(float)
season_df['points_per_game'] = season_df['points_per_game'].astype(float)


#season_df['names'] = season_df['first_name'] + ' ' + season_df['second_name']



#Remove players who has not played more than 60min



#insert current fantasy names in names.
with open(static_path, 'r') as json_file:
    static = json.load(json_file)  # Load the data from the file
    
    
   
static_df = pd.DataFrame(static['elements'])
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
    

    # if closest_match == ['Jair Paula da Cunha Filho']:
    #     print(name_string, closest_match, ['Jair Paula da Cunha Filho'])

        
    matched_ind = np.where(closest_match[0] == fantasy_names.values)[0][0]        
        
    if matched_ind in matched_inds:
        print('Double matched for', name_string, closest_match)
    
    
        
        
        
    matched_inds.append(matched_ind)

season_df = season_df.reset_index(drop=True)

temp_season = pd.DataFrame(index=season_df.index, columns=['own_team_points', 'own_element_points'])

#turn to datetime
season_df['kickoff_time'] = pd.to_datetime(season_df['kickoff_time'])

#add info about wins, team points, and element points
for team in np.unique(season_df.string_team):
    
    team_df = season_df.loc[season_df.string_team == team]
    
    for match_time in np.unique(team_df.kickoff_time):
        
        match_df = team_df.loc[team_df.kickoff_time == match_time]
        
        temp_season.loc[match_df.index, "own_team_points"] = np.sum(match_df.total_points)       
        
        for element in range(1,5):
            element_point_df = match_df.loc[(match_df.element_type == element) & (match_df.minutes > 60)]
            element_df = match_df.loc[(match_df.element_type == element)]
            temp_season.loc[element_df.index, 'own_element_points'] = np.mean(element_point_df.total_points)
            
season_df = pd.concat([season_df, temp_season], axis=1)


#calculate difficulties
home_diff = season_df["team_h_difficulty"].copy()
away_diff = season_df["team_a_difficulty"].copy()

difficulty_diff = (home_diff - away_diff)

season_df['difficulty'] = difficulty_diff

home = season_df['was_home'] == 1
season_df.loc[home, 'difficulty'] = -season_df.loc[home, 'difficulty']

#for all away matches
season_df['own_difficulty'] = season_df["team_a_difficulty"].copy()
season_df['opp_difficulty'] = season_df["team_h_difficulty"].copy()
#correct home matches
season_df.loc[home, 'own_difficulty'] = season_df.loc[home, "team_h_difficulty"]
season_df.loc[home, 'opp_difficulty'] = season_df.loc[home, "team_a_difficulty"]

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

#ALL VARIABLES:
#how variables are includedvariables
#always included. also for current weak
dynamic_features = ['string_opp_team', 'transfers_in', 'transfers_out',
        'was_home', 'own_difficulty', 'opp_difficulty']#, 'difficulty']

#features that I don't have access to in advance.
#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points','defcon']#, 'opp_element_points']#, 'opp_team_points']
#included once
temporal_single_features = ['points_per_game', 'points_per_played_game']
#total_points, minutes, kickoff time not for prediction
#included once
fixed_features = ['kickoff_time', 'element_type', 'string_team', 'season', 'name']

#categories for dtype
categorical_variables = ['element_type', 'string_team', 'season', 'name']
season_df[categorical_variables] = season_df[categorical_variables].astype('category')


#add nan categories
dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
        'opp_difficulty'] #'difficulty',

season_df[['own_difficulty', 'opp_difficulty', 'difficulty']] = season_df[['own_difficulty', 'opp_difficulty', 'difficulty']].astype('Int64')

int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon'] #, 'opp_team_points']
season_df[int_variables] = season_df[int_variables].astype('Int64')

float_variables = ['transfers_in', 'transfers_out', 'threat', 'own_element_points',  'expected_goals', 'expected_assists',
'expected_goals_conceded', 'creativity', 'ict_index', 'influence'] #, 'opp_element_points']
season_df[float_variables] = season_df[float_variables].astype('float')



# If you want to sort the DataFrame in descending order
# df_sorted = df.sort_values(by='date', ascending=False)

train = pd.DataFrame(season_df[fixed_features + temporal_features])
    

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
opponent_point_names = ['opp_team_points'] + [str(k) + 'opp_team_points' for k in range(temporal_window)]  
opponent_element_names = ['opp_element_points'] + [str(k) + 'opp_element_points' for k in range(temporal_window)]  

temp_train = pd.DataFrame(index=train.index, columns=opponent_point_names + opponent_element_names)
     

club_ind = 1
len_clubs = len(season_df.string_opp_team.unique())

for opponent_club in season_df.string_opp_team.unique():
    
    print(opponent_club, club_ind, '/', len_clubs)
    club_ind += 1
    
    #all that played against this club
    opp_selected = season_df.string_opp_team == opponent_club
    
    #loop through all matches of these matches
    for kickoff in np.unique(season_df.loc[opp_selected, 'kickoff_time']):
        
        #find all matches of the opponent before the current match
        opp_match_selected =  opp_selected & (season_df['kickoff_time'] <= kickoff)
            
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
            opp_elem_selected =  opp_selected & (season_df['kickoff_time'] <= kickoff) & (season_df['element_type'] == element_type)
                
            #find the unique kickoff times
            first_indices = season_df.loc[opp_elem_selected].drop_duplicates(subset='kickoff_time', keep='first').index
            
            #initate
            full_oooep = [np.nan] * len(opponent_element_names)
            
            #get the own points of the opponent for the last X matches
            opponents_of_opponents_elements = season_df.loc[first_indices[-temporal_window:], "own_element_points"]
            
            if len(opponents_of_opponents_points):
                full_oooep[-len(opponents_of_opponents_elements):] = opponents_of_opponents_elements
            
            relevant_elements =  opp_selected & (season_df['kickoff_time'] == kickoff) & (season_df['element_type'] == element_type)
            temp_train.loc[relevant_elements, opponent_element_names] = full_oooep[::-1]
            
            # if np.isnan(full_oooep[0]) and kickoff > np.datetime64('2025-08-20T11:30:00.000000000'):
            #     a = djjdjdjdjjd

#set dtype
for col in opponent_point_names:
    temp_train[col] = temp_train[col].astype('Int64')
    
for col in opponent_element_names:
    temp_train[col] = temp_train[col].astype('float')
    
            
train_data = pd.concat([train, temp_train], axis=1)            


#we need everything in the saved file!!!
# selected = train_data["minutes"] >= 60
# train_data = train_data.loc[selected]
# train_data = train_data.drop(['minutes'], axis=1)


 
train_data.to_pickle(r'M:\model_data.pkl')  # Set index=False to not include row indices



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
model_path = r"M:\model.sav"
try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


optimize = False
continue_optimize = False

method = 'xgboost'
temporal_window = 11 # less than what is used...

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
    objective_X = cv_X[columns_to_keep]   
        
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
    objective_copy['match_ind'] = pd.Series(match_ind[cvs_mask])

    
    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
    
    
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask
    
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
    
    #print('done', val_error)

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


with open(r'M:\model_data.pkl', 'rb') as file:
    train_data = pickle.load(file)                




selected = train_data["minutes"] >= 60
train_data = train_data.loc[selected]

#remove players with few matches
unique_names = train_data.name.unique()

n_tresh = 3

for unique_ind, name in enumerate(unique_names):
    selected = (train_data.name == name)

    if sum(selected) < n_tresh:
        train_data.loc[selected, 'name'] = np.nan


#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon']

train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)
                

                
                
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


# Step 2: Calculate 20% of the unique integers

# unique_integers = list(set(match_ind))

# num_to_select = max(1, int(len(unique_integers) * 0.80))  # Ensure at least one is selected

# Step 3: Randomly select 20% of the unique integers
if optimize:
    #9.38
    random.seed(0)
else:
    random.seed(1)

# train_sample = random.sample(unique_integers, num_to_select)

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
                    algo = atpe.suggest,
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
    import statsmodels.api as sm
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
    
    
    grow_policy = ['depthwise', 'lossguide']
    #include feature search in the hyperparams
    check_features = ['transfers_in', 'transfers_out', 'minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
            'total_points', 'expected_goals', 'expected_assists', 'points_per_played_game', 'was_home', 'season',
            'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon', 'name', 'points_per_game', 'string_opp_team', 'own_difficulty', 'opp_difficulty'] #, 'difficulty']

    do_remove_features= ['names', 'points_per_game', 'points_per_played_game', 'season']
    
    
    
    #old_hyperparams["grow_policy"] = grow_policy[old_hyperparams["grow_policy"]]

    loss = objective_xgboost(old_hyperparams)
    old_loss = loss['loss']

    
    print('Old loss: ', old_loss)
        
    
    # #make sure that there will be data left for evaluation in the final model
    # cv_season =  cv_X.iloc[-1].season
    # selected_cv =  cv_X.season == cv_season
    # cv_fraction = sum(selected_cv) / cv_X.shape[0]   
    
    # current_season =  train_X.iloc[-1].season
    # selected_test =  train_X.season == current_season
    # current_fraction = sum(selected_test) / train_X.shape[0]   
    
    #max_eval_fraction = np.min([cv_fraction, current_fraction])
    
    
    #min_eval_fraction = 1/(len(unique_integers) * 0.80)#len(np.unique(cv_stratify))/cv_X.shape[0]
    #we need at least one match every season
    min_eval_fraction = 1/(380*0.8)
    
    
    space={'max_depth': hp.quniform("max_depth", 1, 6000, 1),
            'min_split_loss': hp.uniform('min_split_loss', 0, 350), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 700),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 400),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 700),
            'learning_rate': hp.uniform('learning_rate', 0, 0.05),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 10, 6000, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.25),
            'n_estimators': hp.quniform('n_estimators', 2, 90000, 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 150),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1
            'max_leaves': hp.quniform('max_leaves', 0, 3000, 1),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(150), 1),
            'temporal_window': hp.quniform('temporal_window', 1, temporal_window+1, 1),
        }
    

    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [True, False]), #111

        
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
                            algo = atpe.suggest,
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
            
        #new_hyperparams["grow_policy"] = grow_policy[new_hyperparams["grow_policy"]]
                           
            
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
            print('Keep old loss')
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
        

        #fit_X, eval_X, fit_y, eval_y, fit_sample_weights, eval_sample_weights = train_test_split(objective_X, train_y, sample_weights, test_size=space['eval_fraction'], stratify=stratify, random_state=42)
        
        # match_ind = pd.factorize(
        #     objective_X[['string_team', 'was_home', 'string_opp_team', 'season']]
        #     .apply(lambda row: '-'.join(row.astype(str)), axis=1)
        # )[0]
            
        
        # #get 20% of those matches
        # # Step 1: Get unique integers using a set
        # unique_integers = list(set(match_ind))
        
        # # Step 2: Calculate 20% of the unique integers
        # num_to_select = int(len(unique_integers) * space['eval_fraction'])
        
        
        
        # Step 3: Randomly select 20% of the unique integers
        # eval_sample = random.sample(unique_integers, num_to_select)
        
        # match_ind_df = pd.Series(match_ind) 
        
        # evals_mask = match_ind_df.isin(eval_sample)  # Mask for cross-validation sample
        # fits_mask = ~evals_mask  # Mask for validation, simply the inverse of cvs_mask
        
        
        # Get the 80% of the first matches every season...
        objective_copy = objective_X.copy()
        objective_copy = objective_copy.reset_index(drop=True)
        objective_copy['match_ind'] = pd.Series(match_ind)

        
        # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
        season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                              .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                              .to_dict())
        
        # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
        # option 1: Unique across all seasons:
        fit_sample = list(set().union(*season_selection.values()))
        
        
        fits_mask =  pd.Series(match_ind_df).isin(fit_sample)  # Mask for cross-validation sample
        evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask
        
        
        
        
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
        
        
import time

#time.sleep(60*60)


my_players = [
    {'web_name': 'Raya', 'selling_price': 59, 'element_type': 1},
    {'web_name': 'Dúbravka', 'selling_price': 40, 'element_type': 1},
    
    {'web_name': 'Keane', 'selling_price': 46, 'element_type': 2},
    {'web_name': 'Mings', 'selling_price': 43, 'element_type': 2},
    {'web_name': 'Chalobah', 'selling_price': 53, 'element_type': 2},
    {'web_name': 'Virgil', 'selling_price': 61, 'element_type': 2},
    {'web_name': "Gabriel", 'selling_price': 71, 'element_type': 2},
    
    {'web_name': 'Semenyo', 'selling_price': 79, 'element_type': 3},
    {'web_name': 'Mac Allister', 'selling_price': 62, 'element_type': 3},
    {'web_name': 'M.Salah', 'selling_price': 140, 'element_type': 3},
    {'web_name': 'B.Fernandes', 'selling_price': 99, 'element_type': 3},
    {'web_name': 'Wilson', 'selling_price': 58, 'element_type': 3},
    
    {'web_name': 'Bowen', 'selling_price': 75, 'element_type': 4},
    {'web_name': 'Raúl', 'selling_price': 62, 'element_type': 4},
    {'web_name': 'Tolu', 'selling_price': 54, 'element_type': 4},
]



bank = 7 #in 10ths of M
free_transfers = 1
save_transfers_for_later = 0 #transfers left at end of last round (no need to put higher than 4)

forward_price_limit = -1 #in millions

minutes_thisyear_treshold = 60
form_treshold = 0.1
points_per_game_treshold = 0.1
running_minutes_threshold = -1

#1: ARS, 8: CRY, 13:MCI, 20: WOL
exclude_team = [1, 8, 13, 20]

exclude_players = ['Keane', 'Collins', 'Bruno G.', 'Havertz', 'M.Bizot', 'Robertson', 'Trossard', 'Jörgensen', 'Gvardiol', 'Digne', 'G.Jesus', 'Foden', 'Merino', 'Estêvão', 'Richarlison', 'Ashley Barnes', 'Nketiah', 'Kostoulas', 'Foster', 'Piroe', 'Nmecha', 'Beto', 'Flemming', 'Awoniyi', 'Callum Wilson', 'Acheampong', 'White', 'Cherki']

include_players = []

do_not_exclude_players = ["Semenyo", "Gabriel", "Raya", "Tolu"]



do_not_transfer_out = ['Mings', 'Keane', 'Wilson']
rounds_to_value = 2
#transfer to evaluate per week
trans_per_week = 2

jump_rounds = 0
#if you also want to evaluate players on the bench. in case of uncertain starters.
number_players_eval = 11

wildcard = False
benchboost = False
skip_gw = [100]

tripple_captain_gw = 100

iterations = 20



midfield_price_limit = -1


#assistant manager in 2024-25 season
assistant_manager_gw = 100
assistant_manager_team = 'CRY'
assistant_manager_price = 0.8 #in millions

addition_of_5_afcon_transfers = 166


force_90 = []

manual_pred = 1

#players
#
#afcon_players = ['Foster', 'Ouattara', 'Agbadou', 'M.Salah', 'Sarr', 'Doucouré', 'Ndіaye', 'Gueye', 'Iwobi', 'Bassey', 'Mbeumo', 'Mazraoui', 'Amad', 'Wissa', 'Aina', 'Boly', 'Sangaré', 'P.M.Sarr', 'Traoré', 'Diouf', 'Wan-Bissaka']
afcon_players = []
manual_blanks = {29: ['Rice', 'Wilson']}

#GW               
manual_blank = {} #{31: {'MCI': ['CRY']}}
manual_double = {}
#manual_double = {26: {'WOL': ['ARS', 4, 2]}}


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
import os
from pandas.api.types import CategoricalDtype
import re

num_jobs = 4


#insert string for team
#old PC
directory = os.path.join(r'C:\Users\jorgels\Github\Fantasy-Premier-League\data', season)
team_path = os.path.join(r'C:\Users\jorgels\Github\Fantasy-Premier-League\data', season, 'teams.csv')
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"

try:
    df_teams = pd.read_csv(team_path)

except:
    #insert string for team
    directory = r'C:\Users\jorgels\Documents\GitHub\Fantasy-Premier-League\data' + '/' + season
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

tripple_captain = []
assistant_manager = []
free_hit = []
add_afcon_transfers= []

if benchboost:
    benchboost_gws = []
else:
    benchboost_gws = [-1]
    
for i in range(jump_rounds, rounds_to_value+jump_rounds):
    this_gw = i + current_gameweek
    
    if benchboost:
        benchboost_gws.append(this_gw)
    
    if this_gw in skip_gw:
        free_hit.append(True)
        skip_free_hit_calc = True
    else:
        print(this_gw)
        free_hit.append(False)

        
    if tripple_captain_gw == this_gw:
        tripple_captain.append(True)
    else:
        tripple_captain.append(False)
        
    if assistant_manager_gw >= this_gw-2 and  assistant_manager_gw <= this_gw:
        assistant_manager.append(True)
    else:
        assistant_manager.append(False)
    
    #this will top up the transfers in this week
    if this_gw == addition_of_5_afcon_transfers:
        add_afcon_transfers.append(5)
    else:
        add_afcon_transfers.append(0)

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


# #add nan categories
# dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
#        'other_difficulty'] #'difficulty',

# int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon', 'SoT']

# float_variables = ['transfers_in', 'transfers_out', 'threat']

# #features that I don't have access to in advance.
# #opp_team_points included because it already calculate in model
# temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
#        'total_points', 'expected_goals', 'expected_assists',
#        'expected_goal_assists', 'expected_goals_conceded', 'own_team_points', 'own_element_points']
#        #'points_per_game', 'points_per_played_game']

# temporal_single_features = ['points_per_game', 'points_per_played_game']


# #total_points, minutes, kickoff time not for prediction
# fixed_features = ['element_type', 'string_team', 'season', 'name']







dynamic_features = ['string_opp_team', 'transfers_in', 'transfers_out',
        'was_home', 'own_difficulty', 'other_difficulty']#, 'difficulty']

#features that I don't have access to in advance.
#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points','defcon', 'opp_element_points', 'opp_team_points']
#included once
temporal_single_features = ['points_per_game', 'points_per_played_game']
#total_points, minutes, kickoff time not for prediction
#included once
fixed_features = ['kickoff_time', 'element_type', 'string_team', 'season', 'name']

#categories for dtype
categorical_variables = ['element_type', 'string_team', 'season', 'name']
#season_df[categorical_variables] = season_df[categorical_variables].astype('category')
#add nan categories
dynamic_categorical_variables = ['string_opp_team', 'own_difficulty',
        'other_difficulty'] #'difficulty',

int_variables = ['minutes', 'total_points', 'was_home', 'bps', 'own_team_points', 'defcon', 'opp_team_points']
#season_df[int_variables] = season_df[int_variables].astype('Int64')

float_variables = ['transfers_in', 'transfers_out', 'threat', 'own_element_points',  'expected_goals', 'expected_assists',
'expected_goals_conceded', 'creativity', 'ict_index', 'influence', 'opp_element_points']
#season_df[float_variables] = season_df[float_variables].astype('float')







#free hit
keep_ind = []
#if rounds_to_value == 1 and wildcard:
for el in [1, 2, 3, 4]:
    selected = slim_elements_df.element_type == el
    min_keeper_price = np.min(slim_elements_df.loc[selected, 'now_cost'])
    keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0][0])
    selected_players.iloc[keep_ind[-1]] = False
    
    # if len(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0]) > 1 and el > 1:  
    #     keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == el))[0][1])
    
# else:
#     selected = slim_elements_df.element_type == 1
#     min_keeper_price = np.min(slim_elements_df.loc[selected, 'now_cost'])
    
#     keep_ind.append(np.where((slim_elements_df['now_cost']==min_keeper_price) & (slim_elements_df.element_type == 1))[0][0])
              
    
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

        # url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id)
        # downloaded = False
        # while not downloaded:
        #     try:
        #         r = requests.get(url)
        #         player = r.json()
        #         downloaded = True
        #     except:
        #         time.sleep(30)

        # player_games = pd.DataFrame(player['history'])

        # player_games['kickoff_time'] =  pd.to_datetime(player_games['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        # player_games = player_games.sort_values(by='kickoff_time')
        # player_games.set_index('kickoff_time', inplace=True)

        # fixtures = pd.DataFrame(player['fixtures'])
        # fixtures['kickoff_time'] =  pd.to_datetime(fixtures['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')
        # fixtures = fixtures.sort_values(by='kickoff_time')

        # should_have_trainingdata = True
        # should_have_database = False
        # past_history  = player["history_past"]
        # if past_history == []:
        #     should_have_trainingdata = False

        # else:
            # last_history = past_history[-1]['season_name']

            # if last_history[:4] == previous_season[:4]:
            #     should_have_database = True
                
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
        
        last_known_row = predicting_df.iloc[-1].copy()

        for game in gws.iterrows():

            #add empty row
            new_row = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in predicting_df.dtypes.items()})

            #add fixed
            #only if we want to rotate the features
            new_row.loc[0, fixed_features] = predicting_df[fixed_features].iloc[-1]
            new_row.loc[0, 'points_per_game'] = predicting_df['points_per_game'].iloc[-1]
            new_row.loc[0, 'points_per_played_game'] = predicting_df['points_per_played_game'].iloc[-1]
            #new_row.loc[0] = predicting_df.iloc[-1]

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
                    

            #new_row['difficulty'] = diff_difficulty[game_idx]
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
        
        #since we will not add more data we can loop the columns!
        for match_row in predicting_df.iterrows():
            for feat in predicting_df.keys():
                
                if match_row[0] > 0:
                
                    #split the feat in integer and string
                    m = re.match(r"^(\d+)(.*)$", feat)
                    if m:
                        num = int(m.group(1))   # 20
                        rest = m.group(2)
                        
                        if rest == 'opp_team_points' or rest == 'opp_element_points' or num >= temporal_window:
                            continue
                        
                        #all temporal features is the last one + 1
                        if num == 0:
                            last_match_feat  = rest     
                        else:
                            last_match_feat  = str(num-1) + rest
                        
                        #print(feat, 'Change', predicting_df.loc[match_row[0], feat], 'with', predicting_df.loc[match_row[0]-1, last_match_feat])
                        
                        #the match we look at is the previous match and a column with one less digit as name.
                        #gives nan for matches that are not yet played
                        #predicting_df.loc[match_row[0], feat] = predicting_df.loc[match_row[0]-1, last_match_feat]
                        
                        #if we want the last played match to be used as the starting point
                        predicting_df.loc[match_row[0], feat] = last_known_row[last_match_feat]
                        
                             
                
            
            
            
            
    
        
        # if temporal_window > 0:
        #     for k in range(int(temporal_window)):
    
    
        #         temporal_names = [str(k) + s for s in temporal_features]
    
        #         if k==0:
                    
        #             dynamic_names = [s for s in dynamic_features]        
        #             temporal_single_names = [s for s in temporal_single_features]
                    
        #             col_names = temporal_single_names + dynamic_names + temporal_names 

        #         else:
        #             dynamic_names = [str(k-1) + s for s in dynamic_features] 
                    
        #             col_names = dynamic_names + temporal_names
                    
        #         #add in empty data
        #         temp_train = pd.DataFrame(index=predicting_df.index, columns=col_names)
                
        #         #loop all data
                
        #         if k==0:
        #             temporal_single_data = predicting_df[temporal_single_features].shift(k+1)
        #             temp_train[temporal_single_names] = temporal_single_data.values
                   
                
    
        #         temporal_data = predicting_df[temporal_features].shift(k+1)
        #         dynamic_data = predicting_df[dynamic_features].shift(k)
    
        #         temp_train.loc[temporal_names] = temporal_data.values
        #         temp_train.loc[dynamic_names] = dynamic_data.values
    
        #         #set dtype
        #         for col in temp_train.columns:
    
        #             col_stem = ''.join([char for char in col if not char.isdigit()])
    
        #             if col_stem in dynamic_categorical_variables:
        #                 temp_train[col] = temp_train[col].astype('category')
        #             elif col_stem in int_variables:
        #                 temp_train[col] = temp_train[col].astype('Int64')
        #             elif col_stem in temporal_features or col_stem in float_variables or col_stem in temporal_single_features:
        #                 temp_train[col] = temp_train[col].astype('float')
        #             else:
        #                 print('CHECK', col)
    
        #         predicting_df = pd.concat([predicting_df, temp_train], axis=1)
        
        if temporal_window > 0:        
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
                    full_ooop[-len(opponents_of_opponents_points):] = opponents_of_opponents_points.values
    
                temp_train.loc[index, opponent_point_names] = full_ooop[::-1].to_list()
                
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
                    
                predicting_df.loc[game[0], temp_train.keys()] = temp_train.values[0].copy()
                    
                
            #set dtype
            for col in predicting_df.columns:
    
                col_stem = ''.join([char for char in col if not char.isdigit()])
    
                if col_stem in dynamic_categorical_variables:
                    predicting_df[col] = predicting_df[col].astype('category')
                elif col_stem in int_variables:
                    predicting_df[col] = predicting_df[col].astype('Int64')
                elif col_stem in temporal_features or col_stem in float_variables or col_stem in temporal_single_features:
                    predicting_df[col] = predicting_df[col].astype('float')
                # else:
                #     print('CHECK', col)
                
                
        
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
    
                # if sum(all_rows.name == name) == 0 and (game_idx == 0):
                #     if should_have_trainingdata:
                #         print(name + ': does not exist in training data. Shoul dbe predicted without name')
                #     #estimated = 0
                if game_idx == 0:
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
                    
                    
                #keep a budget player
                if df_name[0] in keep_ind:
                    if estimated < 0.1:
                        estimated = 0.1
                        
                    if game_idx == 0:
                        print('Including because of low price:', df_name[1].web_name)

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
    deduct_points = 0

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
            #print('a')
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
        
        #add transfers
        if add_afcon_transfers[gw] > 0:
            #subtract one since 1 will be added soon
            free_transfers = add_afcon_transfers[gw] - 1 

        if not unlimited_transfers:

            #if all pred is zero skip week (=free hit)
            if sum(predictions[:, gw]) == 0:
                estimated_points = 0

                all_points.append(0)
            else:
                #add one for the gw
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
                        #subtract one for the transfer
                        free_transfers -=1
                    
                        #pay if negative
                        if free_transfers < 0:
                            deduct_points += -transfer_cost
                            free_transfers += 1
                    
                    #ceil the possible number of transfers. 4 since we add one before next round
                    if free_transfers > 4:
                        free_transfers = 4

                gw_prediction = predictions[team, gw]
                team_positions = slim_elements_df.loc[team, 'element_type'].values

                estimated_points = find_team_points(team_positions, gw_prediction, benchboost[gw], tripple_captain[gw])
                
                captain_bonus = np.max(predictions[team, gw])
                
                all_points.append(np.sum(predictions[team, gw])+captain_bonus)

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
        deduct_points += deduct_transfers*-transfer_cost
        
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

if True: #unlimited_transfers:
    for i in range(gw_iteration):
        free_hit_points.append(0)
else:
    
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
        
        parallel_results = Parallel(n_jobs=num_jobs)(delayed(check_random_transfers)(i, True, free_transfers) for i in range(counter, counter+num_jobs))
        
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
        
        transfered_out = []
        
        for ind, k in enumerate(best_transfer):
            trans = transfers[k]
            if not np.isnan(trans[1]):
                print(slim_elements_df.iloc[trans[0]]['web_name'], 'for', slim_elements_df.iloc[trans[1]]['web_name'])
                transfered_out.append(slim_elements_df.iloc[trans[0]]['web_name'])
                
        for p in slim_elements_df.loc[selected].iterrows():
            if not p[1].web_name in transfered_out:
                print(p[1].web_name)
                
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

    
for benchboost_gw in benchboost_gws[::-1]:
    
    benchboost = []
    
    for i in range(jump_rounds, rounds_to_value+jump_rounds):
        this_gw = i + current_gameweek
        
        if benchboost_gw == this_gw:
            benchboost.append(True)
            print('Benchboost', benchboost_gw)
        else:
            benchboost.append(False)


    print('Check baseline')
    
    
    #check current team
    baseline_point, baseline_price, baseline_all_point = objective(no_transfers, unlimited_transfers, free_transfers)
    
    
    best_points = sum(baseline_point)
    best_all_points = sum(baseline_all_point)
    best_price = baseline_price
    counts = np.ones((len(no_transfers), len(probabilities[0])), dtype='uint32')
    best_transfer = [len(transfers)-1 for _ in range(15)]
    
    best_pitch = baseline_point.copy()
    best_bench = [a - b for a, b in zip(baseline_all_point, baseline_point)]





    try:
        print('Check saved transfers')
        #load saved_transfers
        with open(r'M:\best_transfers.pkl', 'rb') as file:
            saved_transfers = pickle.load(file)
            
        check_transfers = []
        for k in saved_transfers:
            check_transfers.append(transfers[k])
                                
        saved_point, saved_price, saved_all_points = objective(check_transfers, unlimited_transfers, free_transfers)
        
        if np.sum(saved_point) > best_points or (np.sum(saved_point) == best_points and saved_price < best_price) or (np.sum(saved_point) == best_points and saved_price == best_price and np.sum(saved_all_point) > best_all_points):
            best_point = saved_point 
            best_price = saved_price
            best_all_points = saved_all_points
            best_transfer = saved_transfers
            
            best_pitch = best_point.copy()
            best_bench = [a - b for a, b in zip(best_all_points, best_point)]
    except:
        print('Saved transfers did not work')
                
    
    all_evaluated_transfers = [no_transfers]
    
    p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
    prob = (p.T) / np.nansum((p.T), axis=0)
    selected = np.isnan(prob)
    prob[selected] = 0
    
    print('Check guided transfers')
    
    check_guided = True
    
    try:
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
    except:
        print('Not able to guide best transfers')
    
    
    counter = 0
    old_num_teams = 0
    
    
    if rounds_to_value == 1:
        batch_size = 1
    else:
        batch_size = 100000
    
    import time
    
    while counter < iterations:
    
        all_evaluated_transfers = []
    
        if counter > 0:
            print('Start random selections')
        
            p = ((probabilities.T - np.nanmin(probabilities, axis=1)).T / counts)**2 + 1e-6
            prob = (p.T) / np.nansum((p.T), axis=0)
            selected = np.isnan(prob)
            prob[selected] = 0
        
            #guessing part. try random combination followed up by a targeted selection
            print('Getting  teams')
            t1_start = time.time()
            
            parallel_results = Parallel(n_jobs=num_jobs)(delayed(check_random_transfers)(i, unlimited_transfers, free_transfers) for i in range(counter, counter+num_jobs))
            t1_stop = time.time()
            print("Elapsed time:", t1_stop - t1_start)
            print('Interpreting results')
        
        
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
                        
                        best_pitch = par[0][ind_max].copy()
                        best_bench = [a - b for a, b in zip(par[2][ind_max], best_pitch)]
                        
                        counter = 0
                        
                        
                        #save transfers
                        with open(r'M:\best_transfers.pkl', 'wb') as file:
                            pickle.dump(best_transfer, file)
                        
                    
        
                all_evaluated_transfers = all_evaluated_transfers + par[3]
                
                #the first prob of each week is different than the others
                k = 0
                for w in range(rounds_to_value):
                    for t_res in range(trans_per_week):
                        index = w*trans_per_week + t_res
        
                        probabilities[index, :] += par[4][index, :]
                        counts[index, :] += par[5][index, :]
    
    
            
    
        # print('Checked', len(all_evaluated_transfers)-old_num_teams, 'teams')
        # old_num_teams = len(all_evaluated_transfers)
    
        # # Convert each list to a tuple
        # unique_tuples = set(tuple(x) for x in all_evaluated_transfers)
        # # Convert the tuples back to lists
        # all_evaluated_transfers = [list(x) for x in unique_tuples]
    
        # print(len(all_evaluated_transfers), 'unique teams')
    
        counter += 1
        
        if len(best_transfer) == 0:
            print('No acceptable teams')
            continue
    
        #print results
        price = []
        last_gw = 0
        try:
            with open(r"M:\best_transfers" + str(benchboost_gw) + ".txt", 'w') as file:
            
                print('gw', 'free_hit', 'bench', file=file)
                print('gw', 'free_hit', 'bench')
                
                for gw_ind, transfer_ind in enumerate(best_transfer):
            
                    transfer = transfers[transfer_ind]
                    
                    gw = int(1+gw_ind/trans_per_week)
                    
                    if not gw == last_gw and not unlimited_transfers:
                        #print('\n')
                        print('GW', gw, np.round(free_hit_points[gw-1] - best_pitch[gw-1], decimals=1), np.round(best_bench[gw-1], decimals=1), file=file)
                        print('GW', gw, np.round(free_hit_points[gw-1] - best_pitch[gw-1], decimals=1), np.round(best_bench[gw-1], decimals=1))
                        
                        last_gw = gw
                        if np.round(best_bench[gw-1], decimals=1) < 0:
                            a = hfhfhfff
            
                    if not transfer == [np.nan, np.nan]:
                        price.append(slim_elements_df.loc[transfer[1], 'now_cost'])
            
                        if not unlimited_transfers:
                            print( slim_elements_df.loc[transfer[0], 'web_name'], 'for', slim_elements_df.loc[transfer[1], 'web_name'], np.round(prob[transfer_ind, gw_ind], 4), file=file)
                            print( np.round(predictions[transfer[0], :], decimals=1), file=file)
                            print( np.round(predictions[transfer[1], :], decimals=1), file=file)
                            #print(prob[transfer_ind, gw_ind])
                        else:
                            print(int(gw_ind), slim_elements_df.loc[transfer[1], 'web_name'], np.round(predictions[transfer[1], :], 1),  np.round(prob[transfer_ind, gw_ind], 4), file=file)
                        
                        if not unlimited_transfers:
                            print( slim_elements_df.loc[transfer[0], 'web_name'], 'for', slim_elements_df.loc[transfer[1], 'web_name'], np.round(prob[transfer_ind, gw_ind], 4))
                            print( np.round(predictions[transfer[0], :], decimals=1))
                            print( np.round(predictions[transfer[1], :], decimals=1))
                            #print(prob[transfer_ind, gw_ind])
                        else:
                            print(int(gw_ind), slim_elements_df.loc[transfer[1], 'web_name'], np.round(predictions[transfer[1], :], 1),  np.round(prob[transfer_ind, gw_ind], 4))
            
            
                    else:
                        if unlimited_transfers:
                            try:
                                max_ind = np.nanargmax(p[gw_ind, :-1])
                                transfer = transfers[max_ind]
                                print(int(gw_ind), slim_elements_df.loc[transfer[0], 'web_name'], np.round(predictions[transfer[0], :], 1), np.round(prob[transfer_ind, gw_ind], 4))
                                print(int(gw_ind), slim_elements_df.loc[transfer[0], 'web_name'], np.round(predictions[transfer[0], :], 1), np.round(prob[transfer_ind, gw_ind], 4), file=file)
                                price.append(slim_elements_df.loc[transfer[0], 'now_cost'])
                            except:
                                print('Not able to print')
                                print('Not able to print', file=file)
                                    
                        
                print('points: ', np.round(sum(best_pitch), decimals=1), '. diff: ',  np.round(best_points-sum(baseline_point), decimals=1), '. price: ', sum(price), file=file)
                print('points: ', np.round(sum(best_pitch), decimals=1), '. diff: ',  np.round(best_points-sum(baseline_point), decimals=1), '. price: ', sum(price))
                print('\n')
        except:
            print('Not able to open file')



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
model_path = r"M:\model.sav"
try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


optimize = True
continue_optimize = False

method = 'xgboost'
temporal_window = 11 # less than what is used...

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
    objective_X = cv_X[columns_to_keep]   
        
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
    objective_copy['match_ind'] = pd.Series(match_ind[cvs_mask])

    
    # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
    season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                          .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                          .to_dict())
    
    # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
    # option 1: Unique across all seasons:
    fit_sample = list(set().union(*season_selection.values()))
    
    
    fits_mask =  pd.Series(match_ind_df[cvs_mask]).isin(fit_sample)  # Mask for cross-validation sample
    evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask
    
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
    
    #print('done', val_error)

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


with open(r'M:\model_data.pkl', 'rb') as file:
    train_data = pickle.load(file)                




selected = train_data["minutes"] >= 60
train_data = train_data.loc[selected]

#remove players with few matches
unique_names = train_data.name.unique()

n_tresh = 3

for unique_ind, name in enumerate(unique_names):
    selected = (train_data.name == name)

    if sum(selected) < n_tresh:
        train_data.loc[selected, 'name'] = np.nan


#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon']

train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)
                

                
                
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


# Step 2: Calculate 20% of the unique integers

# unique_integers = list(set(match_ind))

# num_to_select = max(1, int(len(unique_integers) * 0.80))  # Ensure at least one is selected

# Step 3: Randomly select 20% of the unique integers
if optimize:
    #9.38
    random.seed(0)
else:
    random.seed(1)

# train_sample = random.sample(unique_integers, num_to_select)

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
                    algo = atpe.suggest,
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
    import statsmodels.api as sm
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
    
    
    grow_policy = ['depthwise', 'lossguide']
    #include feature search in the hyperparams
    check_features = ['transfers_in', 'transfers_out', 'minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
            'total_points', 'expected_goals', 'expected_assists', 'points_per_played_game', 'was_home', 'season',
            'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon', 'name', 'points_per_game', 'string_opp_team', 'own_difficulty', 'opp_difficulty'] #, 'difficulty']

    do_remove_features= ['names', 'points_per_game', 'points_per_played_game', 'season']
    
    
    
    #old_hyperparams["grow_policy"] = grow_policy[old_hyperparams["grow_policy"]]

    loss = objective_xgboost(old_hyperparams)
    old_loss = loss['loss']

    
    print('Old loss: ', old_loss)
        
    
    # #make sure that there will be data left for evaluation in the final model
    # cv_season =  cv_X.iloc[-1].season
    # selected_cv =  cv_X.season == cv_season
    # cv_fraction = sum(selected_cv) / cv_X.shape[0]   
    
    # current_season =  train_X.iloc[-1].season
    # selected_test =  train_X.season == current_season
    # current_fraction = sum(selected_test) / train_X.shape[0]   
    
    #max_eval_fraction = np.min([cv_fraction, current_fraction])
    
    
    #min_eval_fraction = 1/(len(unique_integers) * 0.80)#len(np.unique(cv_stratify))/cv_X.shape[0]
    #we need at least one match every season
    min_eval_fraction = 1/(380*0.8)
    
    
    space={'max_depth': hp.quniform("max_depth", 1, 6000, 1),
            'min_split_loss': hp.uniform('min_split_loss', 0, 350), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 700),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 400),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 700),
            'learning_rate': hp.uniform('learning_rate', 0, 0.05),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 10, 6000, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.25),
            'n_estimators': hp.quniform('n_estimators', 2, 90000, 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 150),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1
            'max_leaves': hp.quniform('max_leaves', 0, 3000, 1),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(150), 1),
            'temporal_window': hp.quniform('temporal_window', 1, temporal_window+1, 1),
        }
    

    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [True, False]), #111

        
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
                            algo = atpe.suggest,
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
            
        #new_hyperparams["grow_policy"] = grow_policy[new_hyperparams["grow_policy"]]
                           
            
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
            print('Keep old loss')
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
        

        #fit_X, eval_X, fit_y, eval_y, fit_sample_weights, eval_sample_weights = train_test_split(objective_X, train_y, sample_weights, test_size=space['eval_fraction'], stratify=stratify, random_state=42)
        
        # match_ind = pd.factorize(
        #     objective_X[['string_team', 'was_home', 'string_opp_team', 'season']]
        #     .apply(lambda row: '-'.join(row.astype(str)), axis=1)
        # )[0]
            
        
        # #get 20% of those matches
        # # Step 1: Get unique integers using a set
        # unique_integers = list(set(match_ind))
        
        # # Step 2: Calculate 20% of the unique integers
        # num_to_select = int(len(unique_integers) * space['eval_fraction'])
        
        
        
        # Step 3: Randomly select 20% of the unique integers
        # eval_sample = random.sample(unique_integers, num_to_select)
        
        # match_ind_df = pd.Series(match_ind) 
        
        # evals_mask = match_ind_df.isin(eval_sample)  # Mask for cross-validation sample
        # fits_mask = ~evals_mask  # Mask for validation, simply the inverse of cvs_mask
        
        
        # Get the 80% of the first matches every season...
        objective_copy = objective_X.copy()
        objective_copy = objective_copy.reset_index(drop=True)
        objective_copy['match_ind'] = pd.Series(match_ind)

        
        # groupby seasons and aggregate into a dictionary: season -> set(of chosen match_inds)
        season_selection = (objective_copy.groupby('season', observed=False)['match_ind']
                              .agg(lambda s: first_Xpct_unique(s.tolist(), 1-space['eval_fraction']))
                              .to_dict())
        
        # If you want a single flat list of all chosen match_inds (unique across seasons or duplicates kept):
        # option 1: Unique across all seasons:
        fit_sample = list(set().union(*season_selection.values()))
        
        
        fits_mask =  pd.Series(match_ind_df).isin(fit_sample)  # Mask for cross-validation sample
        evals_mask = ~fits_mask  # Mask for validation, simply the inverse of cvs_mask
        
        
        
        
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
        
        