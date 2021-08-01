### 28 Nov 12:42 
- Order of magnitude improvement with learnt models
- Hierarchial model 
  - Tested to better than linear model
- Randomize prediction target between w and w+1 per player
  - Was implemented in data_processor, perhaps making it pytorch on the fly would provide inifite data
- Non linear model seems to marginally better
- Where to find next order of magnitude improvement ?
  - How to increase number of prediction samples ?
    - Rewrite the get_training_data function to make augmentation on the fly
    - Instead of predicting total_points on one timestep predict in random timesteps in pytorch
    - Use previous year datasets. 
  
### 03 Dec 12:30
- Build on success of opponent feature by reducing them to 2 opponent form numbers and build a large player performance matrix N * D * L
  - Build the giant N * D * L matrix in numpy, do the windowed prediction in pytorch
- Do classic autoregressive modelling on this dataset
  - RNNs
  - Transformers

### 15 Dec 15:46
  - Add total points scored this season as a feature to make predictions smooth

### 10 Jan 15:47
  - Clean up get_training_dataloader
  - Log changes to disk
  - Add captain as dude with most points in season

## 11 Jan 23:27
  - Go back to toy datasets that you can visualize easily for autoregressive modelling. Choose input dimension and one output dimension
  - Input P1, P2, P3, P4 .... Pt. Predict Pt using previous things. 
  - Bayesian linear model, unit tests, clean code.

## 23 Feb 23:47
  - Input P1, P2, P3, P4 .... Pt. Predict Pt using previous things. 
    - Throw a RNN and see what happens
  - Bayesian linear model
    - Predict player scores along with uncertainity 

# 7 march 13:45 
  - Threw a RNN and made it work :) 
  - Augment dataset by using window sizes 3, 4, 5
  - Play around with RNN architectures to see if it helps. 
  - Throw a transformer at it. 
  - Bayesian linear model
    - Predict player scores along with uncertainity 
  
# 11 March 15:31
  - Augment dataset by using window sizes 3, 4, 5
    - Build input maxtrix in train loader
    - Do dynamic masking to determine output in pytorch batch
      - Infinite data augmentation
      - Flexible input size
  - Compare RNN with dynamic augmention with RNN without
  - Play around with RNN architectures to see if it helps. 
  -  Throw a transformer at it. 
  - Bayesian linear model
    - Predict player scores along with uncertainity 

# 12 March 11:12
  - Make them gpu friendly
  - Plot loss curves for models
  - Data augmentation different inputs and different outputs
  - RNN hyper parameter tuning
  - Throw a transformer
  - Bayesian linear model
    - Predict player scores along with uncertainity 
  
# 13 March 21:29
  - Add opponent feature to see if it helps
  - Data augmentation different inputs and different outputs
  - RNN hyper parameter tuning
  - Bayesian linear model
    - Predict player scores along with uncertainity 

# 18 April 13:27
  - Code refactoring
    - Rewrite model interface 
    - Link new data and model interface with agent.py and agent.ipynb 
  - Agent selection should optimize for best 11 and not best 15.
  - Bayesian linear model
    - Predict player scores along with uncertainity 

# 10 Jul 17:57
  - Code refactoring
    - Add wildcard capability in a standalone file.
  - Agent selection should optimize for best 11 and not best 15
  - Bayesian linear model 
    - Predict player scores with uncertainity

# 17 July 11:24
  - Code refactoring
    - Test model train function
    - Test optimal trade function
    - Test performance prediction function
    - Add wildcard capability in a standalone file. Wildcard should choose cheapest gk and defender first 
    - Add decent training visualization
  - Bayesian linear model 
    - Predict player scores with uncertainity
  
# 28 Jul 17:15
  - Code refactoring
    - Test optimal trade function, playing 11 function
    - Add wildcard capability in a standalone file. Wildcard should choose cheapest gk and defender first 
    - Add decent training visualization
  - Log current squad, new squad and top players. 
  - Bayesian linear model 
    - Predict player scores with uncertainity


# 1 Aug 18:20
- Code refactoring
  - Test playing 11 function
  - Add wildcard capability in a standaline file. Wildcard should choose cheapest gk and defender first 
  - Add decent training viz 
- Log changes, current squad, new_squad and top players with agent.ipynb
  - Generate static agent_GAMEWEEK_ID.html and publish 
