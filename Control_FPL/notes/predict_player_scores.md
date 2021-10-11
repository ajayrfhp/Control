# Goal
- Build a model to predict player score in next game based on high frequency input signals
- Use system to select captains and do weekly transfers

# Signals
- Player performances in previous window of games (4, 5)
    - EPL stat signals
    - FPL points signals
    - Previous year signals, previous league signals
- Opponent characteristics - form, talent and style of play
- Player should be free of injuries


# Models

### Model 1
- Some kind of time series model
    - For each player, get s1, s2.. s38
    - Use st to predict s1..t
- S is signal describing player performance game
    - Start with scalar s number of fpl points

- Models
    - median model, average model, linear model 
- window_size = 3
- Features
    - Baseline feature set
        - total points won scalar
    - Real signal feature set
        - 'total_points', 'yellow_cards', 'assists', 'ict_index', 'saves', 'goals_scored', 'goals_conceded'