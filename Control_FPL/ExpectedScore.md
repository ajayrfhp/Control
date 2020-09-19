# Goal
- Use historical performances to provide squad with highest expected reward over the season. 
- Squad selection is done by a heuristic search / sampling with constraints
  
# Formulation
- View it as a constrained packing problem
- Pack squad to give most reward while meeting budget and player position constraints
- Sort available players that can be bought with remaining budget by some heuristic, pick one of k and repeat
- **Greedy random**
    - Heuristic - points per 90 / cost, random selection from options
- **Greedy random with superstar**
    - Pick some superstars first with heuristic points per 90 and then do greedy random for rest
- **Greedy random soft**
    - Heuristic - weighted average of points per 90 / cost and points per 90
    - weights determined from sampling a skewed normal. 


# Misc
- Build nearest neighbour logic for every player using player stats and fill missing performance values of players.
- Distinguish injuries from players being not in the league

