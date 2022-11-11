import argparse
from box import Box
import yaml

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model_logreg_mvn import ModelLogisicRegressionMvn
from dataset_npz import DataModuleFromNPZ


def main():

    pl.seed_everything(2202)
    dm = DataModuleFromNPZ(
        data_dir="data_logistic_regression_2d",
        feature_labels=["inputs", "targets"],
        batch_size=-1,
        num_workers=4,
        shuffle_training=False
    )
    if False:
        import pdb
        pdb.set_trace()

    dm.prepare_data()
    dm.setup(stage="fit")

    model   = ModelLogisicRegressionMvn(
                2,
                dm.size_train(),
                scale_prior=10.0,
                optimizer_name="RMSprop", 
                optimizer_lr=0.1)
    
    trainer = Trainer(max_epochs=50)
    
    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__ == "__main__":
    main()
