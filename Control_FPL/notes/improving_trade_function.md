## Why
- Profiler highlighted the `get_optimal_parallel_double_trade` as being the slowest with taking 120 seconds. 

## Problem 
- Choose one player from squad table to throw away and select one player from non squad to bring in such that (player_in-player_out) >= 0 and cost is optimized. This looks like a variation of knapsack except that cost is a dynamic variable. 

- Example
    - Squad 
        | Player | Value | Cost| 
        |----|----|----|
        |P1|2|1|
        |P2|3|1|
        |P3|4|1|
    - NonSquad
        | Player | Value | Cost| 
        |----|----|----|
        |NP1|6|1|
        |NP2|3|1|
- Will always kicking out the least valuable player work ? No 
    - Squad 
        | Player | Value | Cost| 
        |----|----|----|
        |P1|2|1|
        |P2|3|2|
        |P3|4|2|
    - NonSquad
        | Player | Value | Cost| 
        |----|----|----|
        |NP1|3|1|
        |NP2|9|2|
    - For this example, swap(P1, NP1) gives profit 1 but swap(NP2, P2) gives 6. 

- Naive solution for one swap
    ```
    max_profit = 0, best_swap = None
    for player_out in current_squad:
        for player_in in non_squad:
            profit = player_in.value - player_out.value
            affordable = player_in.cost - player_out.cost 
            if profit > max_profit and affordable:
                profit = max_profit
                best_swap = (player_out, player_in)
    ```
- Problem
    - List of non squad is quite long(700), so one swap is (700 $\times$ 15) comparisons. Number of comparisons when we do 2 swaps at once 700 $\times$ 700 $\times$ 15 $\times$ 15 = 110250000. It takes about 2 minutes to run in my laptop. It is hardly the end of the world, perhaps my time is better spent elsewhere, but I will go deeper here because I have time to kill and my ego is too big. 
- Observation
    - We do not need to look at every non squad player for a given player_out. We just need the most_valuable from non_squad under a given budget. If we precompute get_most_valuable_player_under_cost(non_squad_player, cost) and store it, we can simply look up the most valuable player for a given cost. 
    - This will bring down number of computations to 15 $\times$ 15 $\times$ O(get_most_valuable_player_under_cost(non_squad_player, cost)) $\times$ O(get_most_valuable_player_under_cost(non_squad_player, cost)). If O(get_most_valuable_player_under_cost(non_squad_player, cost)) can be done in less than linear time, we will have improved upon signifcantly. 
- Reduced Problem
    - Find the most valuable player under a given budget in less than linear time. 
        - NonSquad
            | Player | Value | Cost| 
            |----|----|----|
            |NP1|2|1|
            |NP2|5|1|
            |NP3|1|3|
            |NP2|2|4|
    - Solution - build lookups for MVP based on cost keys. 
        - Build MVP_lookup_under_cost
            | Cost|Value|
            | -|-|
            | 1 | (5, NP2)|
            | 3 | (1, NP3)|
            | 4 | (2, NP4)|
            ```
            mvp_under_cost = [None*100] 
            for player in non_squad:
                if mvp_under_cost[player.cost] < player.value:
                    mvp_under_cost[player.cost] = player
            ```
        - This involves a precomputation cost of O(700) and each subsequent call to MVP_lookup_under_cost(cost) takes O(1) time if we store in an array. So O(get_most_valuable_player_under_cost(non_squad_player, cost)) can be done in O(1) and this will bring our computations down to 15 $\times$ 15 + 700.
        - Now the trade is straight forward. 
        ```
        max_profit = 0, best_swap = None
        for player_out in current_squad:
            player_in = mvp_under_cost(player_out.cost)
            profit = player_in.value - player_out.value
            if profit > max_profit:
                profit = max_profit 
                best_swap = (player_out, player_in)
        ```
        

- `get_optimal_double_trade` takes 6 seconds to run now.