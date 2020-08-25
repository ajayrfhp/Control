# Goal
- Use historical performances to predict performance for each player over time
- Plot all player performances over time and choose a squad to provide highest expected reward over the course of season.
- Squad selection is done by a heuristic search with constraints
- Have uncertainity in the model
  
# Model
- Use data $Score(P_{ij})$ to denote score by player i in game j last season. 
- Model $Score(P_{ij})$ to be sampled from a probability distribution, for each $P_i$, use scores in previous games to predict expected score with confidence interval in the next game
- Naturally supports an online learning scenario where we can keep adding games from this season and update for form. 


# Misc
- Build nearest neighbour logic for every player using player stats and fill missing performance values of players.
- Distinguish injuries from players being not in the league

