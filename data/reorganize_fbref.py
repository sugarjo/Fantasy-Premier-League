# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:32:43 2025

@author: jorgels
"""

import pandas as pd
import os
import numpy as np
#2023
#folder = '2023-24'


for folder in ['2024-25', '2023-24', '2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']:
    print(folder)
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
    
    
    directory = os.path.join(directories, folder)
    fixture_csv = os.path.join(directory, 'fixtures.csv')
    
    season_data = pd.read_csv(directory + '\\fbref/' + folder[:-2] + '20' + folder[-2:] + '_player_data.csv')
    fixture_data_fbref = pd.read_csv(directory + '\\fbref/'  + folder[:-2] + '20' + folder[-2:] + '_fixture_data.csv')
    
    selected = np.where((pd.isna(season_data.Pos)) & (~pd.isna(season_data.Nation)))[0]
    
    if len(selected) > 0:
        #checked for 2016-2024
        print('Remove players beacuse they played both as keeper and outfield. Double check!')
        
    season_data = season_data.drop(selected)   
    
      
    #correct assign  game_id. Assume they are ordered
    for id, ind in enumerate(np.unique(fixture_data_fbref.game_id)):
        selected_id = fixture_data_fbref.game_id == ind
        fixture_data_fbref.loc[selected_id, 'game_id'] = id
      
    #fixture_data_fbref.loc[selected_id, 'fantasy_id'] = None
      
    fixture_data_fbref['kickoff_time'] = pd.to_datetime(fixture_data_fbref['Date'] + ' ' + fixture_data_fbref['Time'])
     
    season_data = pd.merge(season_data, fixture_data_fbref.loc[:, ['Wk', 'game_id', 'kickoff_time']], on='game_id', how='left')
    season_data.rename(columns={'Wk': 'gameweek'}, inplace=True)
      
    fbref_names = season_data.Player.to_list()
    
    order = ['NaN', 'GK', 'RB', 'CB', 'DF', 'LB', 'WB', 'DM', 'RM', 'CM', 'MF', 'LM', 'AM', 'RW', 'LW', 'FW'][::-1]
    
    #[RB, RM] after #[LB, CB]
    
    
    
    order_map = {pos: i for i, pos in enumerate(order)}
    
    df  = season_data.copy()
    df = df.fillna('NaN')
    
    def sort_positions(positions):
        # Split positions and sort according to order_map
        pos_list = positions.split(',') if isinstance(positions, str) else [np.nan]
        
        if len(pos_list)==1:
            pos_list.append(pos_list[0])
            
        if len(pos_list)==2:
            pos_list.append(pos_list[1])
            
        # Sort using the order_map, default to high value if not found
        return pos_list#, key=lambda x: order_map.get(x.strip(), float('inf'))
    
    # Create a column of sorted positions
    df['Sorted_Pos'] = df['Pos'].apply(sort_positions)
    
    
    def find_double_substitution(group_sorted):
    
    
        num_substitutions = int(group_sorted.shape[0] -1 - 11)
        
        #sort by ascending minutes
        min_sort_iloc_ind = np.argsort(group_sorted.Min)
        min_sort_ind = group_sorted.index[min_sort_iloc_ind]
        
        #first match those that can in minutes
        for group_ind in min_sort_ind:
            
            #check if possible substituted player has red card (cannot add to 90). check if already subed in
            if group_sorted.loc[group_ind].CrdR == 1 or sum(group_sorted['sub_in_for']==group_ind) > 0 or group_sorted.loc[group_ind].Min == 90:
                continue
            
            min_in = group_sorted.loc[group_ind].Min
        
            if np.any(min_in + group_sorted.Min == 90):
                #remove complement player so that he is not repicked
                poss_out = np.where(min_in + group_sorted.Min == 90)[0]
                
    
                #position of player1
                sub_in_pos = group_sorted.loc[group_ind, 'Sorted_Pos']
                
                #get the one with most similar position for player2 options
                pos_dists = []
                
                #loop the potential players
                for out_iloc_ind in poss_out:
                    
                    out_ind = int(group_sorted.index[out_iloc_ind])
                    
                    #check if player is same (45min). check if player has red card (cannot add to 90).  check if already subed out
                    if out_ind==group_ind or group_sorted.loc[out_ind].CrdR == 1 or group_sorted.loc[out_ind, 'sub_in_for'] or sum(group_sorted['sub_in_for']==out_ind) > 0 or (group_sorted.loc[group_ind, 'sub_in_for']):
                        pos_dists.append(np.inf)
                    else:
                        #get the position
                        sub_out_pos = group_sorted.loc[out_ind, 'Sorted_Pos']
                        
                        d = 0
                        
                        for p1, p2 in zip(sub_in_pos, sub_out_pos):
                            order1 = order.index(p1)
                            order2 = order.index(p2)
                            d += abs(order1-order2)
                            
                        pos_dists.append(d)
                        
                if np.min(pos_dists) == np.inf:
                    continue
                
                out_iloc_ind = poss_out[np.argmin(pos_dists)]
                out_ind = int(group_sorted.index[out_iloc_ind])
                
                
                group_sorted.loc[group_ind, 'sub_in_for'] = out_ind
                
        return group_sorted
    
    
    def find_tripple_substitution(group_sorted):
        
    
        #check for tripple_substitutions
        #first match those that can in minutes
        
        
        num_substitutions = int(group_sorted.shape[0] -1 - 11)
        
        #sort by ascending minutes
        min_sort_iloc_ind = np.argsort(group_sorted.Min)
        min_sort_ind = group_sorted.index[min_sort_iloc_ind]
        
        
    
        
        re_check = True
        
        while re_check:
            re_check = False
            
            #get the one with most similar position for player2 options
            pos_dists = []
            
            for group_ind in min_sort_ind:
                
                if group_sorted.loc[group_ind, 'sub_in_for'] or (sum(group_sorted['sub_in_for'] == group_ind) > 0) or group_sorted.loc[group_ind].CrdR == 1:
                    continue
                
                for group_ind1 in min_sort_ind:
                    
                    if group_sorted.loc[group_ind1, 'sub_in_for'] or (sum(group_sorted['sub_in_for'] == group_ind1) > 0) or group_sorted.loc[group_ind1].CrdR == 1 or (group_ind1 == group_ind):
                        continue
                         
                    for group_ind2 in min_sort_ind:
                        if group_sorted.loc[group_ind2, 'sub_in_for'] or  (sum(group_sorted['sub_in_for'] == group_ind2) > 0) or group_sorted.loc[group_ind2].CrdR == 1 or (group_ind2 == group_ind) or (group_ind2 == group_ind1):
                            continue
                        
                        min_in = group_sorted.loc[group_ind].Min
                        min_in1 = group_sorted.loc[group_ind1].Min
                        min_in2 = group_sorted.loc[group_ind2].Min
                         
                        if (min_in + min_in1 + min_in2) == 90:
                            
                            sub_in_pos = group_sorted.loc[group_ind, 'Sorted_Pos']
                            sub_in_pos1 = group_sorted.loc[group_ind1, 'Sorted_Pos']
                            sub_in_pos2 = group_sorted.loc[group_ind2, 'Sorted_Pos']
                            
                            d = 0
                            
                            for p1, p2 in zip(sub_in_pos, sub_in_pos1):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            for p1, p2 in zip(sub_in_pos, sub_in_pos2):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            for p1, p2 in zip(sub_in_pos2, sub_in_pos1):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            pos_dists.append(d)
                            
                            
            for group_ind in min_sort_ind:
                
                if group_sorted.loc[group_ind, 'sub_in_for'] or (sum(group_sorted['sub_in_for'] == group_ind) > 0) or group_sorted.loc[group_ind].CrdR == 1 or re_check:
                    continue
                
                for group_ind1 in min_sort_ind:
                    
                    if group_sorted.loc[group_ind1, 'sub_in_for'] or (sum(group_sorted['sub_in_for'] == group_ind1) > 0) or group_sorted.loc[group_ind1].CrdR == 1 or (group_ind1 == group_ind) or re_check:
                        continue
                         
                    for group_ind2 in min_sort_ind:
                        if group_sorted.loc[group_ind2, 'sub_in_for'] or  (sum(group_sorted['sub_in_for'] == group_ind2) > 0) or group_sorted.loc[group_ind2].CrdR == 1 or (group_ind2 == group_ind) or (group_ind2 == group_ind1) or re_check:
                            continue
    
                        
                        min_in = group_sorted.loc[group_ind].Min
                        min_in1 = group_sorted.loc[group_ind1].Min
                        min_in2 = group_sorted.loc[group_ind2].Min
                         
                        if (min_in + min_in1 + min_in2) == 90:
                            
                            sub_in_pos = group_sorted.loc[group_ind, 'Sorted_Pos']
                            sub_in_pos1 = group_sorted.loc[group_ind1, 'Sorted_Pos']
                            sub_in_pos2 = group_sorted.loc[group_ind2, 'Sorted_Pos']
                            
                            d = 0
                            
                            for p1, p2 in zip(sub_in_pos, sub_in_pos1):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            for p1, p2 in zip(sub_in_pos, sub_in_pos2):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            for p1, p2 in zip(sub_in_pos2, sub_in_pos1):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            
                            if min(pos_dists) == d:                                             
                        
                                min_val = min([min_in, min_in1, min_in2])
                                max_val = max([min_in, min_in1, min_in2])
                                
                                #assume non has played identical minutes :(
                                
                                #assume that the most played is the middle most played
                                if min_in == min_val and (not num_substitutions == sum(group_sorted['sub_in_for'] > 0)):                                                    
                                    
                                    #assume:
                                    #second is the most
                                    #third is the least
                                    if min_in1 == max_val and ((sum(group_sorted['sub_in_for'] == group_ind1) == 0) and (sum(group_sorted['sub_in_for'] == group_ind) == 0)):
                                        group_sorted.loc[group_ind, 'sub_in_for'] = group_ind1
                                        group_sorted.loc[group_ind1, 'sub_in_for'] = group_ind2
                                        
                                        if num_substitutions - sum(group_sorted['sub_in_for'] > 0) > 1:
                                            re_check = True
                                        
                                        
                                    elif (sum(group_sorted['sub_in_for'] == group_ind2) == 0) and (sum(group_sorted['sub_in_for'] == group_ind) == 0):
                                        group_sorted.loc[group_ind, 'sub_in_for'] = group_ind2
                                        group_sorted.loc[group_ind2, 'sub_in_for'] = group_ind1
                                        if num_substitutions - sum(group_sorted['sub_in_for'] > 0) > 1:
                                            re_check = True
                                
        return group_sorted
    
    
    def find_red_card_substitution(group_sorted):
        
        num_substitutions = int(group_sorted.shape[0] -1 - 11)
        
        #sort by ascending minutes
        min_sort_iloc_ind = np.argsort(group_sorted.Min)
        min_sort_ind = group_sorted.index[min_sort_iloc_ind]
        
        
        #check for red card on substitute out
        for group_ind in min_sort_ind:
            
            min_in = group_sorted.loc[group_ind].Min
            
            #check if possible substituted player has red card (cannot add to 90)
            if group_sorted.loc[group_ind].CrdR == 1:
                if np.any(min_in + group_sorted.Min < 90):
                    #remove complement player so that he is not repicked
                    poss_out = np.where(min_in + group_sorted.Min < 90)[0]
                    
                    #position of player1
                    sub_in_pos = group_sorted.loc[group_ind, 'Sorted_Pos']
                    
                    #get the one with most similar position for player2 options
                    pos_dists = []
                    
                    #loop the potential players
                    for out_iloc_ind in poss_out:
                        
                        out_ind = int(group_sorted.index[out_iloc_ind])
                        
                        #check if red card (cannout go out), same player, if player that goes out already has been subed in (this can happen but will then be a tripple), and if player that goes out already is taken out.
                        if group_sorted.loc[out_ind].CrdR == 1 or out_ind==group_ind or group_sorted.loc[out_ind, 'sub_in_for'] or np.any(group_sorted['sub_in_for'] == out_ind):
                            pos_dists.append(np.inf)
                        else:
                            #get the position
                            sub_out_pos = group_sorted.loc[out_ind, 'Sorted_Pos']
                            
                            d = 0
                            
                            for p1, p2 in zip(sub_in_pos, sub_out_pos):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            pos_dists.append(d)
                    
                    out_iloc_ind = poss_out[np.argmin(pos_dists)]
                    out_ind = int(group_sorted.index[out_iloc_ind])
                    
                    #check if player has red card (cannot add to 90). check if the same (in case of 45min). check if already subed out
                    if group_sorted.loc[out_ind].CrdR == 1 or out_ind==group_ind or group_sorted.loc[out_ind, 'sub_in_for']:
                        continue
                    elif not (group_sorted.loc[group_ind, 'sub_in_for']):
                        group_sorted.loc[group_ind, 'sub_in_for'] = out_ind
                        break
                    
        return group_sorted
    
    
    def find_injured(group_sorted):
        
        
        
        num_substitutions = int(group_sorted.shape[0] -1 - 11)
        
        #sort by ascending minutes
        min_sort_iloc_ind = np.argsort(group_sorted.Min)
        min_sort_ind = group_sorted.index[min_sort_iloc_ind]
        
        
        #check for red card on substitute out
        for group_ind in min_sort_ind:
            
            min_in = group_sorted.loc[group_ind].Min
            
            #check if possible substituted player has no red card (cannot add to 90)
            if group_sorted.loc[group_ind].CrdR == 0:
                if np.any(min_in + group_sorted.Min < 90):
                    #remove complement player so that he is not repicked
                    poss_out = np.where(min_in + group_sorted.Min < 90)[0]
                    
                    #position of player1
                    sub_in_pos = group_sorted.loc[group_ind, 'Sorted_Pos']
                    
                    #get the one with most similar position for player2 options
                    pos_dists = []
                    
                    #loop the potential players
                    for out_iloc_ind in poss_out:
                        
                        out_ind = int(group_sorted.index[out_iloc_ind])
                        
                        #check if red card (cannout go out), same player, if player that goes out already has been subed in (this can happen but will then be a tripple), and if player that goes out already is taken out.
                        if group_sorted.loc[out_ind].CrdR == 1 or out_ind==group_ind or group_sorted.loc[out_ind, 'sub_in_for'] or np.any(group_sorted['sub_in_for'] == out_ind):
                            pos_dists.append(np.inf)
                        else:
                            #get the position
                            sub_out_pos = group_sorted.loc[out_ind, 'Sorted_Pos']
                            
                            d = 0
                            
                            for p1, p2 in zip(sub_in_pos, sub_out_pos):
                                order1 = order.index(p1)
                                order2 = order.index(p2)
                                d += abs(order1-order2)
                                
                            pos_dists.append(d)
                    
                    out_iloc_ind = poss_out[np.argmin(pos_dists)]
                    out_ind = int(group_sorted.index[out_iloc_ind])
                    
                    #check if player has red card (cannot add to 90). check if the same (in case of 45min). check if already subed out
                    if group_sorted.loc[out_ind].CrdR == 1 or out_ind==group_ind or group_sorted.loc[out_ind, 'sub_in_for']:
                        continue
                    elif not (group_sorted.loc[group_ind, 'sub_in_for']):
                        group_sorted.loc[group_ind, 'sub_in_for'] = out_ind
                        print('Injury on last sub:', group_sorted.iloc[0]['kickoff_time'], group_sorted.loc[group_ind, 'Player'])
                        break
                    
        return group_sorted
    
    
                        
                        
                        
    
    # Function to find pairs that sum to 90
    def identify_subs(group_sorted):
        
        group_sorted['sub_in_for'] = None
        
        
        num_substitutions = int(group_sorted.shape[0] -1 - 11)
        
        group_sorted = find_double_substitution(group_sorted)    
        #if not all substitutions
        if num_substitutions - sum(group_sorted['sub_in_for'] > 0) > 1:
            #group_sorted['sub_in_for'] = None
            group_sorted = find_tripple_substitution(group_sorted) 
            #group_sorted = find_double_substitution(group_sorted)  
                            
        if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
            group_sorted = find_red_card_substitution(group_sorted) 
            
                      
        #check if a triple confounds a double
        if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
            
            print('Tripple confounded:', group_sorted.iloc[0]['kickoff_time'], group_sorted.iloc[1]['Player'])
    
            group_sorted['sub_in_for'] = None
            
            group_sorted = find_double_substitution(group_sorted) 
            #remove one
            selected = np.where(group_sorted['sub_in_for'] > 0)[0]
            for rem_iloc in selected:
                rem = group_sorted.index[rem_iloc]
                group_sorted.loc[rem, 'sub_in_for'] = None
            
            
                group_sorted = find_tripple_substitution(group_sorted) 
                
                #if not all substitutions
                if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
                    group_sorted['sub_in_for'] = None
                    
                    group_sorted = find_double_substitution(group_sorted) 
                else:
                    #print(rem_iloc)
                    break
               
                                
            if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
                group_sorted = find_red_card_substitution(group_sorted) 
                
        
        if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
            #check if there is an unsubstituted player
            group_sorted['sub_in_for'] = None
            
            
            group_sorted = find_double_substitution(group_sorted)    
            #if not all substitutions
            if num_substitutions - sum(group_sorted['sub_in_for'] > 0) > 1:
                group_sorted = find_tripple_substitution(group_sorted) 
                                
            if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
                group_sorted = find_red_card_substitution(group_sorted) 
            if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
                group_sorted = find_injured(group_sorted)     
                
                
                            
        if not num_substitutions == sum(group_sorted['sub_in_for'] > 0):
            a = hhfhfhff
            
        return group_sorted
             
             
    
    
    
    def remove_unique_pos(group_sub):
        
        unique_pos = []
        
        for k in group_sub.iterrows():
            if not k[1]['sub_in_for']:
                if len(np.unique(k[1]['Sorted_Pos'])) == 1:
                    unique_pos.append(np.unique(k[1]['Sorted_Pos'])[0])
         
        for k in group_sub.iterrows():
            if not k[1]['sub_in_for']:
                if len(np.unique(k[1]['Sorted_Pos'])) > 1:
                    
                    place_holder = None           
                    
                    for ind, p in enumerate(k[1]['Sorted_Pos']):
                         if p in unique_pos and (p[0] == 'L' or p[0] == 'R'):
                             group_sub.loc[k[0], 'Sorted_Pos'][ind] = 'exchange'
                             
                         elif not place_holder:
                             place_holder = p
                             
                    for ind, p in enumerate(k[1]['Sorted_Pos']):
                         if p == 'exchange':
                             group_sub.loc[k[0], 'Sorted_Pos'][ind] = place_holder                     
                         
        return group_sub
                             
                            
                        
    def rearrange_pairs(group_sorted):    
        
        indices = []
        
        for k in group_sorted.iterrows():
            if not k[1]['sub_in_for']:
                indices.append(k[0])
                
                selected = group_sorted['sub_in_for'] == k[0]
                
                while sum(selected == 1):
                    
                    sub_in_ind = group_sorted.loc[selected].index[0]
                    
                    indices.append(sub_in_ind)
                    
                    selected = group_sorted['sub_in_for'] == sub_in_ind
                    
        sub_rearranged = group_sorted.loc[indices]
        
        if not (sub_rearranged.shape == group_sorted.shape):
            a = fhjfkjfkdja
                                     
        return sub_rearranged                 
                        
                    
    # Group by teams and sort each group
    team_groups = []
    for team, group in df.groupby((df['Player'].str.contains('Players')).cumsum()):
        group_sub = identify_subs(group)
        unique_pos = remove_unique_pos(group_sub)
        
        group_sorted = group.sort_values(by=['Sorted_Pos'], key=lambda x: x.map(lambda pos_list: [order_map.get(p, float('inf')) for p in pos_list]))
        sub_rearranged = rearrange_pairs(group_sorted)
        team_groups.append(sub_rearranged.reset_index(drop=True))
    
    # Concatenate all the sorted groups back into a single DataFrame
    sorted_df = pd.concat(team_groups, ignore_index=True)
    sorted_df = sorted_df.drop(columns=['Sorted_Pos'])
    
    sorted_df[['Clr', 'Recov', 'xA', 'xGC']] = df[['Clr', 'Recov', 'xA', 'xGC']]
    
    
    #correct xGC
    selected = sorted_df.Nation == 'NaN'
    sorted_df.loc[~selected, 'xGC'] = sorted_df.loc[~selected, 'xGC'].copy() /2
    
    sorted_df.to_csv(directory + '\\fbref/' + folder[:-2] + '20' + folder[-2:] + '_player_data.csv', index=False)
    
    
    
    
    
    
    
    
    



        
        
