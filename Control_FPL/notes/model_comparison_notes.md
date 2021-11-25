# Goal
Compare different model types, feature sets and window sizes

## Things to do 
- Compare different feature sets
    - Does adding was_home feature help ?
- Compare different window sizes
    - Is 4 the optimal window size
- Compare different models
    - Linear model vs RNN 
- Does dynamic augmentation help ?

### How to model comparison with pytorch lightning
- Current workflow
    - I want to test performance of my model on a few different feature sets and visualize losses of different features on the same plot in tensorboard.
    ```
    class LightningWrapper(pl.LightningModule):
        def__init__(self, features):
            self.features = features 

        def training_step(self, batch, batch_idx):
            ....
            .....
            self.logger.experiment.add_scalars("version_0", { f"{self.features}" : loss})

    trainer = pl.Trainer()
    for feature_set in potential_feature_sets:
        model = LightningWrapper(feature_set)
        trainer.fit(model)
    ```
    - But this is creating different versions inside tensorboard logging. Does each call to Lightning module create a new version ? Is this because the loggers are not shared across feature sets in my current design ? How do I share logger across different feature sets or different models ?