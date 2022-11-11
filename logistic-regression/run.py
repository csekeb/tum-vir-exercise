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
        batch_size=64,
        num_workers=4,
        shuffle_training=False
    )

    model   = ModelLogisicRegressionMvn(
                dim=2,
                scale_prior=10.0,
                type_loss="stochastic_local",
                optimizer_name="RMSprop", 
                optimizer_lr=0.1)
    
    trainer = Trainer(max_epochs=10)
    
    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__ == "__main__":
    main()
