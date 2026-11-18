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


season = "2026-27"

build_from_scratch = False
check_last_data = True

temporal_window = 12
    

# Function to correct string_team based on the majority
def correct_string_team(group):
    fixture, string_opp_team = group.name
    group['fixture'] = fixture
    group['string_opp_team'] = string_opp_team
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

static_path = os.path.join(directories, f'{season}\static_{season}.json')

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
                
                filename = os.path.join(directories, f'{season}\players\{player_id[1].id}_{season}.json')
                with open(filename , 'w') as json_file:
                    json.dump(player, json_file)
                    
                #only append when the download succeeded
                players.append(player)
            except:
                print('Error in download')
                time.sleep(30)
                
        
    #download gw
    print('Scrap fixtures and gameweeks')
    for i in range(1, 39):
        url = 'https://fantasy.premierleague.com/api/fixtures' + '?event=' + str(i)
        r = requests.get(url)
        #has the length of matches (10)
        gw = r.json()
        
        # Save to a JSON file
        filename = os.path.join(directories, f'{season}\\fixtures\\{i}_{season}.json') 

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
                            'season': season,
                            'name': static_df.loc[selected_static ]["first_name"].iloc[0] + ' ' + static_df.loc[selected_static ]["second_name"].iloc[0],
                            'clearances_blocks_interceptions': history['clearances_blocks_interceptions'],
                            'recoveries': history['recoveries'],
                            'tackles': history['tackles'],
                            'saves': history["saves"],
                        }
                    
                    gw_data.append(new_row)
        if gw_data:  
            gw_df = pd.DataFrame(gw_data)
            filename = os.path.join(directories, f'{season}\\gws\\gw{i}.json')
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
                
                #those that has changed fbref names               
                selected = season_data.Player == 'Andrew Robertson'
                season_data.loc[selected, 'Player'] = 'Andy Robertson'
                
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
                is_json_gw_file = True
            
            except:
                player_path = directory + '/players_raw.csv'
                df_player = pd.read_csv(player_path)
                
                #insert string for team
                df_teams = pd.read_csv(team_path)
                
                string_names = df_teams["short_name"].values
                
                is_json_file = False
                is_json_gw_file = False
            
            
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

                    if not is_json_gw_file:
                        if folder == '2018-19' or folder == '2016-17' or folder == '2017-18':
                            gw = pd.read_csv(gw_path, encoding='latin1')
                        else:
                            gw = pd.read_csv(gw_path)   
                        
                    else:
                        try:
                            gw = pd.read_json(gw_path)
                        except:
                            gw = pd.read_csv(gw_path)  
                            is_json_gw_file = False                        
                        
                    
                        
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
                            filename = directories + fr'\\{folder}\fixtures\{gw_num}_{folder}.json'
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
                        
                          add_points =  (gw.loc[defenders, 'defcon'] >= 10) * 2
                          gw.loc[defenders, 'total_points'] += add_points
                          
                          add_points =  (gw.loc[attackers, 'defcon'] >= 12) * 2
                          gw.loc[attackers, 'total_points'] += add_points
                        
                    
                    #manually change some names:
                    #if multiple fantasy version over the years, duplicate and put the current one first
                    fantasy_manual_names = ['Juan Angulo', 'Allan Andrade Elias', 'José María Andrés Baixauli', 'Mohamed Belloumi', 'Andrew Robertson', 'Emersonn Correia da Silva', 'João Pedro Loureiro da Costa', 'António João Pereira de Albuquerque Tavares da Silva', 'Felipe Rodrigues da Silva', 'Mike Trésor Ndayishimiye', 'Alysson Edward Franco da Rocha dos Santos', 'João Victor de Souza Menezes', 'Rayan Vitor Simplício Rocha', 'Pedro Cardoso de Lima', 'Pablo Felipe Pereira de Jesus', 'John Victor Maciel Furtado', 'Lucas Pires Silva', 'Jair Paula da Cunha Filho', 'Yéremy Pino Santos', 'Manuel Ugarte Ribeiro', 'Sávio Moreira de Oliveira', 'Kevin Santos Lopes de Macedo', 'Mateus Fernandes', 'Mateus Gonçalo Espanha Fernandes', 'Nico González Iglesias', 'Marc Guiu Paz', 'Marcus Oliveira Alencar', "Rodrigo 'Rodri' Hernandez Cascante", 'Rodrigo Moreno', 'Josh Sims', 'Bernardo Mota Veiga de Carvalho e Silva', 'Andrey Nascimento dos Santos', 'Alisson Becker', 'João Maria Lobo Alves Palhares Costa Palhinha Gonçalves', 'Igor Jesus Maciel da Cruz', 'João Pedro Ferreira da Silva', 'Murillo Costa dos Santos', 'Matheus Santos Carneiro da Cunha', 'Rúben dos Santos Gato Alves Dias', 'Estêvão Almeida de Oliveira Gonçalves', "Rodrigo 'Rodri' Hernandez", 'Vitor de Oliveira Nunes dos Reis', 'Welington Damascena Santos', 'André Trindade da Costa Neto', 'Francisco Evanilson de Lima Barbosa', 'João Pedro Ferreira Silva', 'Igor Thiago Nascimento Rodrigues', "Sávio 'Savinho' Moreira de Oliveira", 'Norberto Bercique Gomes Betuncal', 'Anssumane Fati Vieira', 'Victor da Silva', 'Manuel Benson Hedilazio', 'Igor Julio dos Santos de Paulo', 'Murillo Santiago Costa dos Santos', 'Felipe Augusto de Almeida Monteiro', 'Mateus Cardoso Lemos Martins', 'Joao Gomes', 'João Victor Gomes da Silva', 'Danilo dos Santos de Oliveira', 'Alexandre Moreno Lopera', 'Carlos Ribeiro Dias', 'Matheus Santos Carneiro Da Cunha', 'Renan Augusto Lodi dos Santos', 'Lyanco Silveira Neves Vojnovic', 'Willian Borges da Silva', 'Carlos Henrique Casimiro', 'Norberto Murara Neto', 'Antony Matheus dos Santos', 'Lucas Tolentino Coelho de Lima', 'Diogo Teixeira da Silva','Fábio Freitas Gouveia Carvalho', 'Emerson Leite de Souza Junior', 'Bernardo Veiga de Carvalho e Silva', 'Francisco Jorge Tomás Oliveira', 'Samir Caetano de Souza Santos', 'Lyanco Evangelista Silveira Neves Vojnovic', 'Emerson Aparecido Leite de Souza Junior', 'Juan Camilo Hernández Suárez', 'José Malheiro de Sá', 'Francisco Machado Mota de Castro Trincão', 'Francisco Casilla Cortés', 'Rúben Santos Gato Alves Dias', 'Raphael Dias Belloli', 'Vitor Ferreira', 'Semi Ajayi', 'Oluwasemilogo Adesewo Ibidapo Ajayi', 'Ivan Ricardo Neves Abreu Cavaleiro', 'Hélder Wander Sousa de Azevedo e Costa', 'Allan Marques Loureiro', 'Bruno André Cavaco Jordao', 'João Pedro Junqueira de Jesus', 'Borja González Tomás', 'José Reina', 'Roberto Jimenez Gago', 'José Ángel Esmorís Tasende', 'Rodrigo Hernandez', 'Mahmoud Ahmed Ibrahim Hassan', 'José Ignacio Peleteiro Romallo', 'Joelinton Cássio Apolinário de Lira', 'Alexandre Nascimento Costa Silva', 'Fabio Henrique Tavares', 'Bernard Anício Caldeira Duarte', 'André Filipe Tavares Gomes', 'André Filipe Tavares', 'Rúben Gonçalo Silva Nascimento Vinagre', 'Rúben Diogo da Silva Neves', 'Rui Pedro dos Santos Patrício', 'João Filipe Iria Santos Moutinho', 'Jorge Luiz Frello Filho', 'Frederico Rodrigues de Paula Santos', 'Fabricio Agosto Ramírez', 'Bonatini Lohner Maia Bonatini', 'Bernardo Fernandes da Silva Junior', 'Alisson Ramses Becker', 'Olayinka Fredrick Oladotun Ladapo', 'Lucas Rodrigues Moura da Silva', 'João Mário Naval Costa Eduardo', 'Adrien Sebastian Perruchet Silva', 'Danilo Luiz da Silva', 'Jesé Rodríguez Ruiz', 'Jose Luis Mato Sanmartín', 'Ederson Santana de Moraes', 'Bruno Saltor Grau', 'Bernardo Mota Veiga de Carvalho e Silva', 'Robert Kenedy Nunes do Nascimento', 'Gabriel Armando de Abreu', 'Fabio Pereira da Silva', 'Fernando Francisco Reges', 'David Luiz Moreira Marinho', 'Willian Borges Da Silva', 'Pedro Rodríguez Ledesma', 'Oscar dos Santos Emboaba Junior', 'Manuel Agudo Durán', 'Fernando Luiz Rosa', 'Adrián San Miguel del Castillo']
                    fbref_manual_names = ['Riquelme Angulo', 'Allan', 'Chema', 'Bachir Belloumi', 'Andy Robertson', 'Emersonn', 'Costinha', 'António Silva', 'Morato', 'Mike Trésor', 'Alysson', 'Souza', 'Rayan', 'Pedro Lima', 'Pablo', 'John', 'Lucas', 'Jair', 'Yéremy Pino', 'Manuel Ugarte', 'Sávio', 'Kevin', 'Mateus Fernandes', 'Mateus Fernandes', 'Nicolás González', 'Marc Guiu', 'Marquinhos', 'Rodri', 'Rodrigo', 'Josh Sims', 'Bernardo Silva', 'Andrey Santos', 'Alisson', 'João Palhinha', 'Igor Jesus', 'Jota Silva', 'Murillo', 'Matheus Cunha', 'Rúben Dias', 'Estêvão Willian', 'Rodri', 'Vitor Reis', 'Welington', 'André', 'Evanilson', 'Jota Silva', 'Igor Thiago', 'Sávio', 'Beto', 'Ansu Fati', 'Vitinho', 'Benson Manuel', 'Igor', 'Murillo', 'Felipe', 'Tetê', 'João Gomes', 'João Gomes', 'Danilo', 'Álex Moreno', 'Cafú', 'Matheus Cunha', 'Renan Lodi', 'Lyanco', 'Willian', 'Casemiro', 'Neto', 'Antony', 'Lucas Paquetá', 'Diogo Jota', 'Fabio Carvalho', 'Emerson', 'Bernardo Silva', 'Chiquinho', 'Samir Santos', 'Lyanco', 'Emerson', 'Cucho', 'José Sá', 'Francisco Trincão', 'Kiko Casilla', 'Rúben Dias', 'Raphinha', 'Vitinha', 'Semi Ajayi', 'Semi Ajayi', 'Ivan Cavaleiro', 'Hélder Costa', 'Allan', 'Bruno Jordão', 'João Pedro', 'Borja Bastón', 'Pepe Reina', 'Roberto', 'Angeliño', 'Rodri', 'Trézéguet', 'Jota', 'Joelinton', 'Xande Silva', 'Fabinho', 'Bernard', 'André Gomes', 'André Gomes', 'Rúben Vinagre', 'Rúben Neves', 'Rui Patrício', 'João Moutinho', 'Jorginho', 'Fred', 'Fabricio', 'Léo Bonatini', 'Bernardo', 'Alisson', 'Freddie Ladapo', 'Lucas Moura', 'João Mário', 'Adrien Silva', 'Danilo', 'Jesé', 'Joselu', 'Ederson', 'Bruno', 'Bernardo Silva', 'Kenedy', 'Gabriel Paulista', 'Fábio', 'Fernando', 'David Luiz', 'Willian', 'Pedro', 'Oscar', 'Nolito', 'Fernandinho', 'Adrián']
                     
                    
                    #covid weeks
                    if gw_num > 38:
                        gw_num = gw_num - 9
                        
    
                                  
                    for el in gw.iterrows():
                        
                        # if 'Mohamed-Ali Cho' in  el[1]['name']:
                            
                        #     a=jdjhdhd
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
                                    
                            if folder in ['2020-21', '2021-22', '2022-23']:
                                if closest_match[0] == 'Allan':
                                    selected =  season_data.Player == 'Allan'
                                    
                                    season_data.loc[selected, 'Player'] = 'XXAllan OldXX'
                                    fbref_names[max_ind] = 'XXAllan OldXX'
                                    closest_match = ['XXAllan OldXX']
                                    
                            #player_name change DURING season
                            
                            # if closest_match[0] == 'Mohamed Ali Cho' and not name_string == 'Mohamed-Ali Cho':
                            #     a = fjhkfhf
                            #     print(name_string)

                                
                            if not closest_match:
                                if el[1].minutes > 0:
                                    print('No player matched in fbref', el[1]['name'], name_string)  
                                    a = hhdhdh
                                continue
                                
                            # if not closest_match:
                            #     closest_match = [name_string]
                            #     print('No matches for', name_string)
                            
                            updated_player_names[name_string] = closest_match[0]
                            
                            #print('Automatic matching')
                        
                        #if the name is in the fbref liist
                        #keep this before inventing new names.
                        num_player_selected = fbref_names.count(closest_match[0])
                        
                        if num_player_selected == 0 and el[1].minutes > 0:

                            print('Player played, but not matched', el[0], el[1]['name'], closest_match)
                            a = hfhhfhf
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
                if not is_json_gw_file:
                    df_gw['string_opp_team'] = None
                 
                df_gw['points_per_game'] = np.nan       
                df_gw['points_per_played_game'] = np.nan
    
                # Calculate values on my own
                for player in df_gw['element'].unique():
    
                    selected_ind = df_gw['element'] == player
                    player_df = df_gw[selected_ind]
                    player_df.set_index('kickoff_time', inplace=True)
                    
                    
                    opp_team = []
                    
                    if not is_json_gw_file:
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

                    #points_per_game and points_per_played_game are calculated for the whole
                    #dataset after all seasons are concatenated (see below)
        
                    
                    
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
                #season_df = season_df.groupby(['fixture', 'string_opp_team'], group_keys=False).apply(correct_string_team)
                season_df = ( season_df.groupby(['fixture', 'string_opp_team'], as_index=False, group_keys=False).apply(correct_string_team, include_groups=False))
                
                # def get_majority(series):
                #     return series.mode()[0]  # mode() returns the most common value
                #df['string_team'] = season_df.groupby(['fixture', 'string_opp_team'])['string_team'].transform(get_majority)
                print('Number of player games', len(season_df))
    
    
                season_dfs.append(season_df)
                
season_df = pd.concat(season_dfs)
season_df = season_df.reset_index(drop=True)

#turn to datetime
season_df['kickoff_time'] = pd.to_datetime(season_df['kickoff_time'])

#calculate points_per_game and points_per_played_game on all data.
#when build_from_scratch is False, df_gw only contains the new gameweeks, so these
#have to be calculated after the old and new data are concatenated
season_df = season_df.sort_values(by='kickoff_time', kind='stable')
player_groups = [season_df['season'], season_df['element']]

#points per game: cumulative points (including this match) divided by round
season_df['points_per_game'] = season_df['total_points'].groupby(player_groups).cumsum() / season_df['round']

#points per played game: mean points in previous matches with at least 60 minutes. 0 if none
played_60 = season_df['minutes'] >= 60
points_60 = season_df['total_points'].where(played_60, 0)
previous_points_60 = points_60.groupby(player_groups).cumsum() - points_60
previous_games_60 = played_60.astype(int).groupby(player_groups).cumsum() - played_60.astype(int)
season_df['points_per_played_game'] = (previous_points_60 / previous_games_60.where(previous_games_60 > 0)).fillna(0)

season_df = season_df.sort_index()

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
selected_season = season_df.season == season
matched_inds = []


#these are the fbref names
for name_string in np.unique(season_df.loc[selected_season, 'name']):   
    
    #get the fantasy name
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


 
train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\model_data.pkl')  # Set index=False to not include row indices


import os
import sys
import re
import pickle

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

from hyperopt import Trials, fmin, hp, atpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate

#shared functions for the local and element models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl_model_utils import (GROW_POLICY, normalize_feature_flags, build_match_index,
                             mask_unseen_categories, two_fold_loss, cross_fit_predictions,
                             select_columns, fit_xgboost, season_fit_eval_masks, training_metadata, remove_out_of_range)

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"


try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'

#optimize everything. Typically used when hyperparams ranges change. Does not predict new points
optimize_mode = True

#this will exclude the players with frozen data
exclude_frozen_players = False

current_season = '2026-27'
max_temporal_window = 12


#optimize hyperparameters. uses the data of the current player (set in the loop below)
def objective_xgboost(space):
    
    if space["grow_policy"] == 0 or space["grow_policy"] == 1:
        space["grow_policy"] = GROW_POLICY[space["grow_policy"]]
    
    #fit on the first half and predict the second, and vice versa. penalize slow models
    return two_fold_loss(space, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds,
                         check_features, unknown_features, seconds_per_loss_unit=25)

               
import requests

#get statistics of all players
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
statistics = r.json()

elements_df = pd.DataFrame(statistics['elements'])


names = elements_df.first_name + ' ' + elements_df.second_name
        

with open(r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl", 'rb') as file:
    original_data = pickle.load(file)                

selected = original_data["minutes"] >= 60
train_data = original_data.loc[selected].copy()


#included for all windows, but not current
temporal_features = ['minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
        'total_points', 'expected_goals', 'expected_assists',
        'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon']


#the non digit version of these features will be removed
unknown_features = ['minutes', 'ict_index', 'total_points', 'own_team_points', 'own_element_points', 'defcon']

train_y = train_data['total_points'].astype(int)
train_X = train_data.drop(columns=temporal_features)          


#get one index for each match. players of both teams in a match get the same index
match_ind = build_match_index(train_X)
                
train_X = train_X.drop(['kickoff_time'], axis=1)  




unique_names = train_X.name.unique()

hyper_vals = {'max_depth': [],
            'min_split_loss': [], #log?
            'reg_lambda' : [],
            'reg_alpha': [],
            'min_child_weight' : [],
            'learning_rate': [],
            'early_stopping_rounds': [],
            'n_estimators': [],
            'max_delta_step': [],
            'max_leaves': [],
            'max_bin':  [],
            'temporal_window': [],
            'losses': [],
            'num_matches': [],
        }


train_data['local_predictions'] = np.nan
train_data['local_const_predictions'] = np.nan

original_data['local_predictions'] = np.nan
original_data['local_const_predictions'] = np.nan

last_hyperparams = []

unlikely_values = 0

#train local models
for ind, name in enumerate(unique_names[::-1]):
    
    if ind > 0:
        if best_hyperparams:
            last_hyperparams = best_hyperparams.copy()
    
    
    
    model_path = rf"\\platon.uio.no\med-imb-u1\jorgels\fantasy\local_models\{name}.sav"
    
    num_matches_train = -1
    sel_name = train_X.name == name
    
    print(ind, '/', len(unique_names), name, sum(sel_name))
    
    if optimize_mode:
        search_hyperparams = True
    else:
        search_hyperparams = False
    
    best_hyperparams = []
    old_hyperparams = []
    new_hyperparams = []
    tested_hyperparams = []
    
    #start from an empty summary. otherwise the summary of the previous player is used if loading fails
    summary = {}
    
    if os.path.isfile(model_path):
        try:
            print('Loading model')
            with open(model_path, "rb") as f:
                summary = pickle.load(f)
                
            loading_fail = False
            
            
            # new_hyperparams = {}
            # for field, val in summary["tested_hyperparams"].items():
                
            #     if val == 0 or val==1:
            #         new_hyperparams[field] = int(val)
            #     else:
            #         new_hyperparams[field] = val
                    
            # old_hyperparams = {}
            # for field, val in summary["hyperparams"].items():
                
            #     if val == 0 or val==1:
            #         old_hyperparams[field] = int(val)
            #     else:
            #         old_hyperparams[field] = val
                
            #only compare hyperparams if there are new data
            #do not optimize if not more matches
            try:
                new_hyperparams = summary["tested_hyperparams"]
                try:
                    a = new_hyperparams["num_matches"]
                except:
                    new_hyperparams["num_matches"] = sum(sel_name)
            except:
                new_hyperparams = []


            try:
                old_hyperparams = summary["hyperparams"]
                try:
                    a = old_hyperparams["num_matches"]
                except:
                    old_hyperparams["num_matches"] = sum(sel_name)
                    
            except:
                old_hyperparams = []

            
            #if we have both
            if old_hyperparams and new_hyperparams:
                if (old_hyperparams["num_matches"] == sum(sel_name)) and (new_hyperparams["num_matches"] == sum(sel_name)):
                    print('No new data for old and new hyperparameters. Use new')
                    old_hyperparams = []
                    best_hyperparams = new_hyperparams
                    tested_hyperparams = []
                    
                elif new_hyperparams["num_matches"] == sum(sel_name):
                    print('No new data for new hyperparameters. Use old')
                    new_hyperparams = []
                    best_hyperparams = old_hyperparams
                    tested_hyperparams = new_hyperparams
            #if we have only old
            elif old_hyperparams:
                new_hyperparams = []
                best_hyperparams = old_hyperparams
                tested_hyperparams = []
            #if we have only new
            else:
                new_hyperparams = []
                best_hyperparams = []
                tested_hyperparams = []
                search_hyperparams = True
                    
                    

            
            # #delete later:
            # try:
            #     summary["hyperparams"]["num_matches"] = summary["num_matches"]
            #     num_matches_train = summary["hyperparams"]["num_matches"]
            # except:
            #     num_matches_train = -1
            
            #if not optimizing or no new data (and these players shuld be exlcuded) and we have hyperparams
            if best_hyperparams and (not optimize_mode or (exclude_frozen_players and ((sum(sel_name) == num_matches_train) or (name not in names)))):

                print('Do not search')                        
                search_hyperparams = False
            
        except:
            loading_fail = True
            search_hyperparams = True
            
    else:
        loading_fail = True
        search_hyperparams = True
                
    # Get the 80% of the first matches every season...
    train_copy = train_X.loc[sel_name].copy()
    
    # Identify categorical columns
    categorical_columns = train_copy.select_dtypes(['category']).columns

    # Reset categories for each categorical column
    for column in categorical_columns:
        train_copy[column] = train_copy[column].cat.remove_unused_categories()
        
    empty_category_cols = [col for col in train_copy.select_dtypes(include=["category"]).columns
       if len(train_copy[col].cat.categories) == 0]
    
    train_copy = train_copy.drop(columns=empty_category_cols)
        
        
        
    train_copy = train_copy.reset_index(drop=True)

    #the first half of the matches of every season are used for cv, the second half for validation
    cvs_mask, vals_mask = season_fit_eval_masks(train_copy['season'], match_ind[sel_name], 0.5)
    cvs_mask = pd.Series(cvs_mask)
    vals_mask = pd.Series(vals_mask)
    
    #set the split for later use
    subset_idx = train_data.index[sel_name]        # index of selected rows
    rows_to_update = subset_idx[cvs_mask]  
    train_data.loc[rows_to_update, 'split'] = 1
    rows_to_update = subset_idx[vals_mask]
    train_data.loc[rows_to_update, 'split'] = 0
    
    #at least one train and val
    if sum(sel_name) < 2:
        if sum(sel_name) > 0:
            summary = {'model': [], 'train_features': [], 'hyperparams': [], "local_constant": np.mean(train_y.loc[sel_name])}#, 'all_rows': original_df}
            pickle.dump(summary, open(model_path, 'wb'))
            
        print('Skip name. Only a few matches:', sum(sel_name))
        continue
    
    
    
    cv_X = train_copy.loc[cvs_mask.values].copy()
    val_X =  train_copy.loc[vals_mask.values].copy()
    cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
    val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()
    
    #make sure all categories in val_x and cv_x are present in the player data
    val_X = mask_unseen_categories(val_X, train_copy)
    cv_X = mask_unseen_categories(cv_X, train_copy)
    
    #match index of the two halves
    cv_match_inds = match_ind[sel_name][cvs_mask]
    val_match_inds = match_ind[sel_name][vals_mask]

    mean_cv = np.mean(cv_y)
    
    subset_idx = train_data.index[sel_name]        # index of selected rows
    rows_to_update = subset_idx[vals_mask]         # further filtered rows
    train_data.loc[rows_to_update, 'local_const_predictions'] = mean_cv
    original_data.loc[rows_to_update, 'local_const_predictions'] = mean_cv
    
    mean_val = np.mean(val_y)
    #train_error = np.mean(np.abs((cv_y - mean_cv)**2))
    average_error2 = np.mean(np.abs((cv_y - mean_val)**2))
    
    rows_to_update = subset_idx[cvs_mask]         # further filtered rows
    train_data.loc[rows_to_update, 'local_const_predictions'] = mean_val
    original_data.loc[rows_to_update, 'local_const_predictions'] = mean_val
    
    #min_eval_fraction = 1/(len(unique_integers) * 0.80)#len(np.unique(cv_stratify))/cv_X.shape[0]
    #we need at least one match every season
    
    _, matches_season1 = np.unique(cv_X.season, return_counts=True)
    _, matches_season2 = np.unique(val_X.season, return_counts=True)
    
    if len(matches_season1) == 0  or len(matches_season2) == 0:
        print('Skip name. Only a few matches:', sum(sel_name))
        
        #run one leave out 
        subset_idx = train_data.index[sel_name]        # index of selected rows
        
        for s_ind in range(len(subset_idx)):
            loo_points = np.mean(train_y[subset_idx[:s_ind].to_list() + subset_idx[s_ind+1:].to_list()])
                        
            rows_to_update = subset_idx[s_ind]         # further filtered rows
            train_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
            original_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
            
            summary = {'model': [], 'train_features': [], 'hyperparams': [], "local_constant": np.mean(train_y.loc[sel_name])}#, 'all_rows': original_df}
            pickle.dump(summary, open(model_path, 'wb'))
        
        continue
    
    else:    
    
        min_eval_fraction = 1/min([max(matches_season1), max(matches_season2)])
        
        #at least four matches to make a model...
        if max(matches_season1) <= 3 or max(matches_season2) <= 3:
            
            #run one leave out 
            subset_idx = train_data.index[sel_name]        # index of selected rows
            
            for s_ind in range(len(subset_idx)):
                loo_points = np.mean(train_y[subset_idx[:s_ind].to_list() + subset_idx[s_ind+1:].to_list()])
                            
                rows_to_update = subset_idx[s_ind]         # further filtered rows
                train_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
                original_data.loc[rows_to_update, 'local_const_predictions'] = loo_points
                
            summary = {'model': [], 'train_features': [], 'hyperparams': [], "local_constant": np.mean(train_y.loc[sel_name])}#, 'all_rows': original_df}
            pickle.dump(summary, open(model_path, 'wb'))
                
                
            print('Skip name. Only a few matches:', sum(sel_name))
            continue
    
    
    
    individual_features = ['season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_difficulty', 'ict_index', 'transfers_in', 'transfers_out', 'expected_goals', 'expected_assists',
            'expected_goals_conceded', 'defcon', 'own_element_points', 'season']
    
    #all_features = ['total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'ict_index',  'influence', 'threat', 'creativity', 'bps', 'expected_goals', 'expected_assists',
         #   'expected_goals_conceded', 'defcon', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'own_difficulty'] #, 'difficulty']
    
    
    keep_cols = []
    #remove the features not needed
    for feat in cv_X.keys():
        no_digit_feat = re.sub(r'\d+', '', feat)
        
        if no_digit_feat in individual_features:
            keep_cols.append(feat)

    cv_X = cv_X[keep_cols]
    val_X = val_X[keep_cols]
    train_copy = train_copy[keep_cols]
    
    # #include feature search in the hyperparams
    # check_features = ['transfers_in', 'transfers_out', 'minutes', 'ict_index', 'influence', 'threat', 'creativity', 'bps',
    #         'total_points', 'expected_goals', 'expected_assists', 'points_per_played_game', 'was_home', 'season',
    #         'expected_goals_conceded', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'defcon', 'name', 'points_per_game', 'string_opp_team', 'own_difficulty', 'opp_difficulty'] #, 'difficulty']
    
    check_features = individual_features
    #do_remove_features= ['names', 'points_per_game', 'points_per_played_game', 'season']   
    
    temporal_window = len(train_copy)
    if temporal_window > max_temporal_window:
        temporal_window = max_temporal_window
        
        
    space={'max_depth': hp.qloguniform("max_depth", np.log(1), np.log(90), 1), 
            'min_split_loss': hp.loguniform('min_split_loss', np.log(0.001), np.log(20)), #log?
            'reg_lambda' : hp.uniform('reg_lambda', 0, 650),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 30),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 15),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(5.5)),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 500, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.51),
            'n_estimators': hp.quniform('n_estimators', 2, 300000, 1), #up
            'max_delta_step': hp.uniform('max_delta_step', 0, 10500),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1
            'max_leaves': hp.quniform('max_leaves', 0, 2500, 1), #up
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(40), 1),
            'temporal_window': hp.quniform('temporal_window', 1, temporal_window, 1),
        }

    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [False, True])
    
    #marks hyperparameters where the feature flags work (see normalize_feature_flags)
    space['feature_flags_v2'] = hp.choice('feature_flags_v2', [True])
        
        
    #hyperparameters stored before the feature flag fix keep all features
    old_hyperparams = normalize_feature_flags(old_hyperparams, check_features)
    new_hyperparams = normalize_feature_flags(new_hyperparams, check_features)
    best_hyperparams = normalize_feature_flags(best_hyperparams, check_features)
    last_hyperparams = normalize_feature_flags(last_hyperparams, check_features)
        
    if search_hyperparams:
        
        if ind == 0:
            trials = Trials() 
            
        else:
            if last_hyperparams:
                if last_hyperparams['eval_fraction'] < min_eval_fraction:
                    last_hyperparams['eval_fraction'] = min_eval_fraction
                last_hyperparams["num_matches"] = sum(sel_name)
                
                if last_hyperparams["grow_policy"] == 'lossguide':
                    last_hyperparams["grow_policy"] = 1
                elif last_hyperparams["grow_policy"] == 'depthwise':
                    last_hyperparams["grow_policy"] = 0
                    
                trials = generate_trials_to_calculate([last_hyperparams])
                
            else:
                trials = Trials() 
                print('had to generate new trials')
            
        #trials = Trials() 
        print('Search for hyperparameters')

        loss = np.inf
        tid = len(trials)
        
        while loss > 60 and tid < 200:
    
            #optmimize hyperparameters. use all training data
            new_hyperparams = fmin(fn = objective_xgboost,
                            space = space,
                            algo = atpe.suggest,
                            trials = trials,
                            max_evals=99999999,
                            early_stop_fn=no_progress_loss(20)
                            )
            
            loss = trials.best_trial['result']['loss']
            tid = len(trials)
            
            new_hyperparams["losses"] = trials.best_trial['result']['loss']
            new_hyperparams["num_matches"] = sum(sel_name)
                    
            if optimize_mode:
                best_hyperparams = new_hyperparams
            elif old_hyperparams:
                best_hyperparams = old_hyperparams
            else:
                best_hyperparams = new_hyperparams
    
    else:
        new_loss = np.inf
        old_loss = np.inf
        
        if new_hyperparams and old_hyperparams:
            new_loss = objective_xgboost(new_hyperparams)["loss"]
            old_loss = objective_xgboost(old_hyperparams)["loss"]
            print('Old', old_loss)
            print('New', new_loss)
        elif old_hyperparams:
            old_loss = 0
        elif new_hyperparams:
            new_loss = 0
            
            
        if new_loss < old_loss:
            best_hyperparams = new_hyperparams
            print('Use new')
        else:
            best_hyperparams = old_hyperparams
            print('Use old')
    
    for k in best_hyperparams.keys():
        if k in hyper_vals:
            hyper_vals[k].append(best_hyperparams[k])

    #do not continue if don't have new data
    if optimize_mode:
        summary['tested_hyperparams'] = new_hyperparams
        pickle.dump(summary, open(model_path, 'wb'))
        continue
    else:
        tested_hyperparams = new_hyperparams
    
    

        
        
    print('Predict for hierarchy')
    
    if best_hyperparams['grow_policy'] == 0 or best_hyperparams['grow_policy'] == 1:
        best_hyperparams['grow_policy'] = GROW_POLICY[best_hyperparams['grow_policy']]
        
        
    #get cross-valiated local predictions
    subset_idx = train_data.index[sel_name]        # index of selected rows
    
    for val_pred, y1, y2, test_is_val in cross_fit_predictions(best_hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds, check_features, unknown_features):
        
        val_error = mean_squared_error(y2,  val_pred)
        
        #remove values that are outside of the training range
        unlikely_values += remove_out_of_range(val_pred, y1)
        
        random_error = np.mean(np.abs((y2 - np.mean(y1))**2))
        
        rows_to_update = subset_idx[vals_mask if test_is_val else cvs_mask]
        
        train_data.loc[rows_to_update, 'local_predictions'] = val_pred
        original_data.loc[rows_to_update, 'local_predictions'] = val_pred
        
        print('done', random_error-val_error)
    
    
    player_predictions = train_data[sel_name]
    
    #store the model for the predictions...
    #for the players who are not part of current season we don't need the models...
    #might want to skip for first round(some players might return)
        
    #if not active players
    if not loading_fail and not name in names.values:
        summary = {'model': [], 'train_features': [], 'hyperparams': best_hyperparams, "local_constant": [], "predictions": player_predictions}#, 'all_rows': original_df}
        pickle.dump(summary, open(model_path, 'wb'))
        continue    
    
    #skip if not new hyperparams or data has not changed
    if search_hyperparams or not sum(sel_name) == best_hyperparams["num_matches"]:
        
        print('Train for predictions', name, sum(sel_name))

        X1 = train_copy.reset_index(drop=True)
        
        columns = select_columns(X1, best_hyperparams, check_features, unknown_features)
        
        model = fit_xgboost(X1, train_y[sel_name], match_ind[sel_name], best_hyperparams, columns, mask_eval_categories=True)
                    
        
        summary = {'model': model, 'train_features': training_metadata(X1[columns]), 'hyperparams': best_hyperparams, 'tested_hyperparams': tested_hyperparams, "local_constant": np.mean(train_y.loc[sel_name]), "predictions": player_predictions}#, 'all_rows': original_df}
    
        pickle.dump(summary, open(model_path, 'wb'))
    
    
        

for k in hyper_vals:
    plt.figure()
    #print(k, np.max(np.array(hyper_vals[k])[selected]))
    plt.hist(np.array(hyper_vals[k]))
    plt.title(k)
    
plt.show()

local_error = np.nanmean((train_y - train_data['local_predictions'])**2)
print('Local error:', local_error)
        
train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\model_data.pkl')  # Set index=False to not include row indices


print('Unlikely values set to nan:', unlikely_values, train_data.shape[0])



import os
import sys
import re
import pickle

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings


from matplotlib import pyplot as plt

import xgboost as xgb

from hyperopt import Trials, fmin, hp, atpe
from hyperopt.early_stop import no_progress_loss

#shared functions for the local and element models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl_model_utils import (GROW_POLICY, normalize_feature_flags, build_match_index,
                             mask_unseen_categories, two_fold_loss, cross_fit_predictions,
                             select_columns, fit_xgboost, training_metadata, remove_out_of_range)

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'

update_hyperparams = True
check_hyperparameters = True

try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


method = 'xgboost'

season_dfs = []


#optimize hyperparameters. uses the data of the current position (set in the loop below)
def objective_xgboost(space):
    
    if space["grow_policy"] == 0 or space["grow_policy"] == 1:
        space["grow_policy"] = GROW_POLICY[space["grow_policy"]]
    
    #fit on the first half and predict the second, and vice versa
    return two_fold_loss(space, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds,
                         check_features, unknown_features)

               
with open(r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl", 'rb') as file:
    original_data = pickle.load(file)                

selected = original_data["minutes"] >= 60
train_data = original_data.loc[selected].copy()


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

#get one index for each match. players of both teams in a match get the same index
match_ind = build_match_index(train_X)

train_X = train_X.drop(['kickoff_time'], axis=1)


train_data['element_predictions'] = np.nan
original_data['element_predictions'] = np.nan

unlikely_values = 0

#train local models
for pos in range(1,5):
    
    print('Position:', pos)
    
    sel_name = train_X.element_type == pos  
    
    
    # Get the 80% of the first matches every season...
    train_copy = train_X.loc[sel_name].copy()
    
    # Identify categorical columns
    categorical_columns = train_copy.select_dtypes(['category']).columns

    # Reset categories for each categorical column
    for column in categorical_columns:
        train_copy[column] = train_copy[column].cat.remove_unused_categories()
        
        
        
    train_copy = train_copy.reset_index(drop=True)
    train_copy['match_ind'] = pd.Series(match_ind[sel_name])
   
    
    cvs_mask = train_copy.split == 0  # Mask for cross-validation sample
    #vals_mask =train_copy.split == 1  # Mask for validation, simply the inverse of cvs_mask
    vals_mask = ~cvs_mask       
    
    cv_X = train_copy.loc[cvs_mask.values].copy()
    val_X =  train_copy.loc[vals_mask.values].copy()
    cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
    val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()
    
    #make sure all categories in val_x and cv_x are present in the position data
    val_X = mask_unseen_categories(val_X, train_copy)
    cv_X = mask_unseen_categories(cv_X, train_copy)
    
    #match index of the two halves
    cv_match_inds = match_ind[sel_name][cvs_mask]
    val_match_inds = match_ind[sel_name][vals_mask]
    
    cv_match_count = np.max(train_copy.loc[cvs_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
    val_match_count = np.max(train_copy.loc[vals_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
    min_eval_fraction = 1/np.min([cv_match_count, val_match_count])  
    
    
    #individual_features = ['season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_difficulty', 'ict_index', 'transfers_in', 'transfers_out', 'bps'] #, 'difficulty']
    
    element_features = ['ict_index', 'name', 'season', 'total_points', 'own_element_points', 'opp_element_points', 'own_difficulty', 'opp_difficulty', 'transfers_out', 'transfers_in', 'string_opp_team', 'name', 'points_per_played_game', 'points_per_game', 'local_predictions', 'local_const_predictions', 'minutes', 'was_home']

    # all_features = ['element_type', 'season', 'total_points', 'points_per_played_game', 'points_per_game', 'minutes', 'ict_index',  'influence', 'threat', 'creativity', 'bps', 'expected_goals', 'expected_assists',
    #         'expected_goals_conceded', 'defcon', 'string_opp_team',  'opp_difficulty', 'was_home', 'own_team_points', 'own_element_points', 'opp_team_points', 'opp_element_points', 'own_difficulty', 'transfers_in', 'transfers_out', 'local_predictions', 'local_const_predictions'] #, 'difficulty']


    check_features = element_features
    
    
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
    
    
        
    hyperparam_path = main_directory + f'\models\hyperparams{pos}.pkl'
    with open(hyperparam_path, 'rb') as f:
        old_trials = pickle.load(f)
    
    hyperparams = old_trials.best_trial['misc']['vals']
    #reformat the lists
    old_hyperparams = {}
    for field, val in hyperparams.items():
        old_hyperparams[field] = val[0]
    
    #hyperparameters stored before the feature flag fix keep all features
    old_hyperparams = normalize_feature_flags(old_hyperparams, check_features)
        
    
    if old_hyperparams['eval_fraction'] < min_eval_fraction:
        old_hyperparams['eval_fraction'] = min_eval_fraction
    
    loss = objective_xgboost(old_hyperparams)
    old_loss = loss['loss']
    
    
    print('Old loss: ', old_loss)
    
    space={ 'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 5500, 1),
            'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.45),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(5)),
            'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(175), 1),
            'max_delta_step': hp.uniform('max_delta_step', 0, 9200),
            'max_depth': hp.qloguniform("max_depth", np.log(1), np.log(900), 1), 
            'max_leaves': hp.quniform('max_leaves', 0, 4000, 1),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 3000),
            'min_split_loss': hp.loguniform('min_split_loss', np.log(0.001), np.log(200)), #log?
            'n_estimators': hp.quniform('n_estimators', 2, 65000, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 1500),
            'reg_lambda' : hp.uniform('reg_lambda', 0, 50000),            
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'temporal_window': hp.quniform('temporal_window', 1, 8, 1),
            'grow_policy': hp.choice('grow_policy', [0, 1]), #1  
            
        }

    for feature in check_features:
        # Add a new entry in the dictionary with the feature as the key
        # and hp.quniform('n_estimators', 0, 2, 1) as the value
        space[feature] = hp.choice(feature, [False, True])
    
    #marks hyperparameters where the feature flags work (see normalize_feature_flags)
    space['feature_flags_v2'] = hp.choice('feature_flags_v2', [True])

       
    new_trials = Trials() 
            
    loss = np.inf
    tid = 0
    
    
    tested_hyperparam_path = main_directory + f'\models\\hyperparams_test_{pos}.pkl'
    
    
    #temp_hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
    if update_hyperparams:
        while loss > 60 and tid < 200:
    
            #optmimize hyperparameters. use all training data
            new_hyperparams = fmin(fn = objective_xgboost,
                            space = space,
                            algo = atpe.suggest,
                            trials = new_trials,
                            max_evals=99999999,
                            early_stop_fn=no_progress_loss(200)#(750)
                            )
            
            pickle.dump(new_trials, open(tested_hyperparam_path, "wb"))
            
            loss = new_trials.best_trial['result']['loss']
            tid = len(new_trials)
        
        
        for field, val in new_hyperparams.items():
             #new_hyperparams[field] = val[0]
             print(field, val)
    
        continue        
        
        
    with open(tested_hyperparam_path, 'rb') as f:
        new_trials = pickle.load(f)
        
    hyperparams = new_trials.best_trial['misc']['vals']
    #reformat the lists
    new_hyperparams = {}
    for field, val in hyperparams.items():
        new_hyperparams[field] = val[0]
        
    new_hyperparams = normalize_feature_flags(new_hyperparams, check_features)
        
    if new_hyperparams['eval_fraction'] < min_eval_fraction:
        new_hyperparams['eval_fraction'] = min_eval_fraction
        
    loss = objective_xgboost(new_hyperparams)
    new_loss = loss['loss']
     
    print('New loss: ', new_loss)


    if new_loss < old_loss and check_hyperparameters:
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
        
    best_hyperparams = normalize_feature_flags(best_hyperparams, check_features)
    
    
    if best_hyperparams['eval_fraction'] < min_eval_fraction:
        best_hyperparams['eval_fraction'] = min_eval_fraction
    
    random_error = []
    local_const_error = []
    local_error = []
    element_error = []   
    
    subset_idx = train_data.index[sel_name]        # index of selected rows
    
    for val_pred, y1, y2, test_is_val in cross_fit_predictions(best_hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds, check_features, unknown_features):
        
        #remove values that are outside of the training range
        unlikely_values += remove_out_of_range(val_pred, y1)
        
        random_error.append(np.mean(np.abs((y2 - np.mean(y1))**2)))
        
        rows_to_update = subset_idx[vals_mask if test_is_val else cvs_mask]
        
        original_data.loc[rows_to_update, 'element_predictions'] = val_pred
        train_data.loc[rows_to_update, 'element_predictions'] = val_pred
        
        local_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_predictions'])**2))
        
        local_const_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_const_predictions'])**2))
        
        #nanmean since predictions outside of the training range are set to nan
        element_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'element_predictions'])**2))

        
    print('Mean element error:', np.mean(random_error))
    print('Local constant error:', np.mean(local_const_error))
    print('Local error:', np.mean(local_error))
    print('ELement error:', np.mean(element_error))
    
    X1 = train_copy.reset_index(drop=True)
    
    columns = select_columns(X1, best_hyperparams, check_features, unknown_features)
    
    model = fit_xgboost(X1, train_y[sel_name], match_ind[sel_name], best_hyperparams, columns, mask_eval_categories=True)

    xgb.plot_importance(model, importance_type='gain',
                    max_num_features=20, show_values=False)
    plt.show()
    
    


    summary = {'model': model, 'train_features': training_metadata(X1[columns]), 'hyperparameters': best_hyperparams}#, 'all_rows': original_df}
    
    model_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy"  + f'\element_model_{pos}.sav'
    
    pickle.dump(summary, open(model_path, 'wb'))
        
        
if not update_hyperparams:
    train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\model_data.pkl')  # Set index=False to not include row indices
    
    
print('Unlikely values set to nan:', unlikely_values, train_data.shape[0])



import os
import sys
import re
import pickle

import pandas as pd
import numpy as np

#to make hyperopt work
import warnings
np.warnings = warnings

from matplotlib import pyplot as plt

import xgboost as xgb

from hyperopt import Trials, fmin, hp, atpe
from hyperopt.early_stop import no_progress_loss

#shared functions for the local, element and all models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl_model_utils import (GROW_POLICY, normalize_feature_flags, build_match_index,
                             mask_unseen_categories, two_fold_loss, cross_fit_predictions,
                             select_columns, fit_xgboost, training_metadata, remove_out_of_range)

#runs the update
update_hyperparams = True
#retrain but does not check the updated hyperparams
check_hyperparameters = True

directories = r'C:\Users\jorgels\Github\Fantasy-Premier-League\data'
model_path = r"\\platon.uio.no\med-imb-u1\jorgels\model.sav"


try:
    folders = os.listdir(directories)
    main_directory = r'C:\Users\jorgels\Github\Fantasy-Premier-League'
except:
    main_directory = r'C:\Users\jorgels\Git\Fantasy-Premier-League'


method = 'xgboost'

season_dfs = []


#optimize hyperparameters. uses the data of all players (set below)
def objective_xgboost(space):
    
    if space["grow_policy"] == 0 or space["grow_policy"] == 1:
        space["grow_policy"] = GROW_POLICY[space["grow_policy"]]
    
    #fit on the first half and predict the second, and vice versa
    return two_fold_loss(space, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds,
                         check_features, unknown_features)

               
with open(r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\model_data.pkl", 'rb') as file:
    original_data = pickle.load(file)                

selected = original_data["minutes"] >= 60
train_data = original_data.loc[selected].copy()


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




#get one index for each match. players of both teams in a match get the same index
match_ind = build_match_index(train_X)

train_X = train_X.drop(['kickoff_time'], axis=1)

#include all
sel_name = train_X.element_type == -1  
sel_name = ~sel_name
 
# Get the 80% of the first matches every season...
train_copy = train_X.loc[sel_name].copy()        
        

# Identify categorical columns
categorical_columns = train_copy.select_dtypes(['category']).columns

# Reset categories for each categorical column
for column in categorical_columns:
    train_copy[column] = train_copy[column].cat.remove_unused_categories()
        
train_copy = train_copy.reset_index(drop=True)
train_copy['match_ind'] = pd.Series(match_ind[sel_name])



cvs_mask = train_copy.split == 0  # Mask for cross-validation sample
#vals_mask =train_copy.split == 1  # Mask for validation, simply the inverse of cvs_mask
vals_mask = ~cvs_mask  


 
 
cv_X = train_copy.loc[cvs_mask.values].copy()
val_X =  train_copy.loc[vals_mask.values].copy()
cv_y =  train_y.loc[sel_name].loc[cvs_mask.values].copy()
val_y = train_y.loc[sel_name].loc[vals_mask.values].copy()

#make sure all categories in val_x and cv_x are present in the data
val_X = mask_unseen_categories(val_X, train_copy)
cv_X = mask_unseen_categories(cv_X, train_copy)

#match index of the two halves
cv_match_inds = match_ind[sel_name][cvs_mask]
val_match_inds = match_ind[sel_name][vals_mask]

cv_match_count = np.max(train_copy.loc[cvs_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
val_match_count = np.max(train_copy.loc[vals_mask].groupby('season', observed=False)['match_ind'].agg(lambda s: len(np.unique(s))))
min_eval_fraction = 1/np.min([cv_match_count, val_match_count])  


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

#hyperparameters stored before the feature flag fix keep all features
old_hyperparams = normalize_feature_flags(old_hyperparams, check_features)
    

if old_hyperparams['eval_fraction'] < min_eval_fraction:
    old_hyperparams['eval_fraction'] = min_eval_fraction
 
loss = objective_xgboost(old_hyperparams)
old_loss = loss['loss']


print('Old loss: ', old_loss)

space={'early_stopping_rounds': hp.quniform("early_stopping_rounds", 1, 4000, 1),
       'max_bin':  hp.qloguniform('max_bin', np.log(2), np.log(130), 1),
       'max_delta_step': hp.uniform('max_delta_step', 0, 2000),
       'max_depth': hp.qloguniform("max_depth", np.log(1), np.log(300), 1),
       'min_child_weight' : hp.uniform('min_child_weight', 0, 800),
       'max_leaves': hp.quniform('max_leaves', 0, 400, 1),
        'min_split_loss': hp.loguniform('min_split_loss', np.log(0.001), np.log(300)), #log?
        'n_estimators': hp.quniform('n_estimators', 2, 40000, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 1500),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 6000),
        'eval_fraction': hp.uniform('eval_fraction', min_eval_fraction, 0.45),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(6)),
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
    space[feature] = hp.choice(feature, [False, True])

#marks hyperparameters where the feature flags work (see normalize_feature_flags)
space['feature_flags_v2'] = hp.choice('feature_flags_v2', [True])

   
trials = Trials() 
        
loss = np.inf
tid = 0

tested_hyperparam_path = main_directory + f'\models\\tested_hyperparams.pkl'


#temp_hyperparam_path = main_directory + '\models\hyperparams_temp.pkl'
if update_hyperparams:
    
    while loss > 60 and tid < 200:
    
        #optmimize hyperparameters. use all training data
        new_hyperparams = fmin(fn = objective_xgboost,
                        space = space,
                        algo = atpe.suggest,
                        trials = trials,
                        max_evals=99999999,
                        early_stop_fn=no_progress_loss(750)
                        )
        
        pickle.dump(trials, open(tested_hyperparam_path, "wb"))
        
        loss = trials.best_trial['result']['loss']
        tid = len(trials)
    
    
    for field, val in new_hyperparams.items():
         #new_hyperparams[field] = val[0]
         print(field, val)
    
    
else:

    
    
    with open(tested_hyperparam_path, 'rb') as f:
        new_trials = pickle.load(f)
        
    hyperparams = new_trials.best_trial['misc']['vals']
    #reformat the lists
    new_hyperparams = {}
    for field, val in hyperparams.items():
        new_hyperparams[field] = val[0]
        
    new_hyperparams = normalize_feature_flags(new_hyperparams, check_features)
        
    
    if new_hyperparams['eval_fraction'] < min_eval_fraction:
        new_hyperparams['eval_fraction'] = min_eval_fraction
        
    loss = objective_xgboost(new_hyperparams)
    new_loss = loss['loss']
     
    print('New loss: ', new_loss)
    
    
    if new_loss < old_loss and check_hyperparameters:
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
        
    best_hyperparams = normalize_feature_flags(best_hyperparams, check_features)
        
    
    
     
    if best_hyperparams['eval_fraction'] < min_eval_fraction:
        best_hyperparams['eval_fraction'] = min_eval_fraction
    
    random_error = []
    local_const_error = []
    local_error = []
    element_error = []
    all_error = []
    
    unlikely_values = 0
    
     
    subset_idx = train_data.index[sel_name]        # index of selected rows
    
    for val_pred, y1, y2, test_is_val in cross_fit_predictions(best_hyperparams, cv_X, cv_y, cv_match_inds, val_X, val_y, val_match_inds, check_features, unknown_features):
        
        #remove values that are outside of the training range
        unlikely_values += remove_out_of_range(val_pred, y1)
        
        random_error.append(np.mean(np.abs((y2 - np.mean(y1))**2)))
        
        rows_to_update = subset_idx[vals_mask if test_is_val else cvs_mask]
        
        train_data.loc[rows_to_update, 'all_predictions'] = val_pred
        
        local_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_predictions'])**2))
        
        local_const_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'local_const_predictions'])**2))
        
        element_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'element_predictions'])**2))
        
        #nanmean since predictions outside of the training range are set to nan
        all_error.append(np.nanmean((y2 - train_data.loc[rows_to_update, 'all_predictions'])**2))
         
    print('Mean error:', np.mean(random_error))
    print('Local constant error:', np.mean(local_const_error))
    print('Local error:', np.mean(local_error))
    print('Element error:', np.mean(element_error))
    print('All error:', np.mean(all_error))
            
                     
    #train_data.to_pickle(r'\\platon.uio.no\med-imb-u1\jorgels\\fantasy\\model_data.pkl')  # Set index=False to not include row indices
    
    X1 = train_copy.reset_index(drop=True)
    
    columns = select_columns(X1, best_hyperparams, check_features, unknown_features)
    
    model = fit_xgboost(X1, train_y[sel_name], match_ind[sel_name], best_hyperparams, columns, mask_eval_categories=True)
    
    xgb.plot_importance(model, importance_type='gain',
                    max_num_features=20, show_values=False)
    plt.show()
    
    
    
    
    
    summary = {'model': model, 'train_features': training_metadata(X1[columns]), 'hyperparameters': best_hyperparams}#, 'all_rows': original_df}
    
    model_path = r"\\platon.uio.no\med-imb-u1\jorgels\fantasy\all_model.sav"
    
    pickle.dump(summary, open(model_path, 'wb'))
    
    print('Unlikely values set to nan:', unlikely_values, train_data.shape[0])
        
    
    
    
    


