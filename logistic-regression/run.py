import argparse
from box import Box
import yaml

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model_logreg_dn import ModelLogisicRegressionMvn
from dataset_npz import DataModuleFromNPZ
import torch.nn as nn

def main():

    pl.seed_everything(2202)
    dm = DataModuleFromNPZ(
        data_dir="data_logistic_two_moons",
        feature_labels=["inputs", "targets"],
        batch_size=1024,
        num_workers=1,
        shuffle_training=True
    )
    if False:
        import pdb
        pdb.set_trace()

    dm.prepare_data()
    dm.setup(stage="fit")

   
    #
    # training a diagonal model
    #

    feature_map = nn.Sequential(nn.Linear(2,256), nn.LeakyReLU(), nn.Linear(256,2))
    model_diag = ModelLogisicRegressionMvn(
                2,
                dm.size_train(),
                feature_map=feature_map,
                is_diagonal=True,
                scale_prior=10.0,
                optimizer_name="RMSprop", 
                optimizer_lr=0.1,
                save_path="runs/models/diagonal")


    trainer = Trainer(max_epochs=200)
    
    trainer.fit(model_diag, dm)
    trainer.test(model_diag, dm)


if __name__ == "__main__":
    main()
