from collections import defaultdict
import numpy as np

def solve_knapsack(weights, values, names, max_weight, num_players, teams=[], max_players_from_one_team=3, global_num_teams_in_path=defaultdict(int)):
    '''
        Args
            weights - array of length n 
            values - array of length n
            names - array of length n 
            max_weight - max weight that can be put in knapsack 
            num_players - max num of players
            teams - array of length n teams to which players belong to 
            max_players_from_one_team - max number of players from one team
        Returns
            subset of values whose sum is maximal and its weight being under max_weight
        No path should have more than 3 players from the same team
    '''
    dp = np.zeros((len(weights) + 1, max_weight + 1))
    paths = []
    teams_in_path = []
    for i in range(dp.shape[0]):
        path = []
        team_in_path = []
        for j in range(dp.shape[1]):
            path.append([])
            team_in_path.append([])
        paths.append(path)
        teams_in_path.append(team_in_path)

    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            is_valid = True 
            num_teams_in_path = defaultdict(int)
            for team in teams_in_path[i-1][j-weights[i-1]]:
                num_teams_in_path[team] += 1
            is_valid = num_teams_in_path[teams[i-1]] < max_players_from_one_team and global_num_teams_in_path[teams[i-1]] < max_players_from_one_team


            if is_valid and j >= weights[i-1] and (values[i-1] + dp[i-1][j-weights[i-1]] >= dp[i-1][j]) and len(paths[i-1][j-weights[i-1]]) < num_players:
                dp[i][j] = values[i-1] + dp[i-1][j-weights[i-1]]
                paths[i][j].extend(paths[i-1][j-weights[i-1]] + [names[i-1]])
                teams_in_path[i][j].extend(teams_in_path[i-1][j-weights[i-1]] + [teams[i-1]])
            else:
                dp[i][j] = dp[i-1][j]
                paths[i][j] = list(paths[i-1][j])
                teams_in_path[i][j] = list(teams_in_path[i-1][j])
    

    best_path = dp[-1].argmax()    
    
    indices = [ names.index(name) for name in paths[-1][best_path] ]
    best_weights = [weights[index] for index in indices]
    best_values = [values[index] for index in indices] 
    for team in teams_in_path[-1][best_path]:
        global_num_teams_in_path[team] += 1
    return paths[-1][best_path], best_weights, best_values, global_num_teams_in_path

