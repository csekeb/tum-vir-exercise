import argparse
from box import Box
import yaml

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model_vae import ModelVae
from data_mnist import MNISTDataModule


def main(config):

    #cudnn.deterministic = True
    #cudnn.benchmark = False
    pl.seed_everything(config.model.seed_everything)
    
    dm      = MNISTDataModule(**config.data)
    model   = ModelVae(config.model,config.optimizers, config.saving, config_to_save=config)
    trainer = Trainer(**config.trainer)
    

    trainer.fit(model, dm)
    #trainer.test(dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training/testing  model from config")
    parser.add_argument("--config", "-c",
                        dest="filename",
                        metavar="FILE",
                        help="path to the config file")
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    with open(args.filename, "r") as file:
        config = yaml.safe_load(file)

    main(Box(config))
