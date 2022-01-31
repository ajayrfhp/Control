## Why ?
- Others should be able to run and test models easily by importing our dataset library. 

## How ?
- Build Wrapper around varav's fpl data and builds a pytorch contextual prediction dataset
- Build prediction dataset.
    - `trainset, testset = get_epl_dataset(feature_names, prediction_names, window_size, batch_size, years)`
    - Dataset has input batches of shape (N, InD, L) and output batches of shape (N, OutD, L)
- Show demo on colab
    - Use epl_dataset.py to benchmark some prediction tasks in colab