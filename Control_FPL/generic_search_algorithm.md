Select squad with maximal value following the budget and formation constraints

Constraints:
    total cost <= 10000
    num_forwards = 3, num_midfielders = 5, num_defenders = 5, num_goalkeepers = 2

Each player has a {points_per_game, now_cost}

Id Cost Value
0 5 3
1 4 2
2 6 7

http://www.es.ele.tue.nl/education/5MC10/Solutions/knapsack.pdf

Formal specification
    weights -> w1, w2, .. wn
    values -> v1, v2 ... vn
    Find subset T such that maximal value and sum of subset < W

dp array of size N * N, dp[i, j] indicates maximum value that can be obtained by using elements 1 to i with a weight j. 

Example : 
    values = [4, 5, 7]
    weights = [2, 1, 4]
    W = 3
    Answer should be [4, 5], sum 9. 

dp[i, j] = max(dp[i-1, j], values[i] + dp[i-1, j - weights[i]])

dp
    0   1   2   3   4
0   0   0   0   0   0
1   0   0   4   0   0   
2   0   5   4   9   0
3   0   5   4   9   7

split the budget equally and do 4 knapsacks
Build answer path array. 
If we are using element to update max value, add element to path. 

    if values[i] + dp[i-1, j - weights[i] > dp[i-1, j]:
        paths[i, j] = paths[i-1, j - weights[i]] + [values[i]]
    else
        paths[i, j] = paths[i-1, j] 
    




