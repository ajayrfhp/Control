# Notes
## Motivation
    - Challenges when deploying the model - Scale, version control, model reproducibility, aligning stakeholders
    - Data, Model and code - Change in one changes everything
        - After deploying Model into system, we would need to retrain model on new data. 
        - We might reformulate the problem and prepare new datasets
        - We might go back to collect more data 
        - We might change model assumptions
        - We might change the model itself. 
    - Common issues
        - Data quality is very important
        - Model decay - performance of model in production can decay over time if new real life data keeps coming in
        - Model may not generalize well. 
    - Extend devops idea for ML to include data and model. 
        - Docker and Kubernetes allow deployent of model in a scalable way. 

## Some Tools
    - Abstract data and computation steps and have a pipeline. This will allow for model reproducability
    - DataVersionControl library - Version control for datasets
    - Model DB, PachyDerm - s
    - Model orchestration - Seldon, MLEAP, DeepDetect. 

# Resources 
- [MLOPS](https://ml-ops.org/)
- [Awesome MLE](https://github.com/EthicalML/awesome-production-machine-learning)