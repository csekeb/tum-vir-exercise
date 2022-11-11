import numpy as np
import torch
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DataModuleFromNPZ(LightningDataModule):
    def __init__(
        self,
        data_dir,
        file_ext: str = 'npz',
        feature_labels=None,
        batch_size: int = 256,
        num_workers: int = 4,
        shuffle_training: bool = True
    ):
        super().__init__()
        self.data_dir       = data_dir
        self.file_ext       = file_ext
        self.feature_labels = feature_labels

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()

    def prepare_data(self):
        # download
        exists_all = [os.path.exists(os.path.join(self.data_dir, s))
                      for s in ["train", "test", "validate"]]

        if not all(exists_all):
            Exception(f"Data not exist at specified location {self.data_dir}")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.data_train    = DatasetFromNPZ(os.path.join(self.data_dir, "train"),
                                                feature_labels=self.feature_labels,
                                                file_ext=self.file_ext)

            self.data_validate = DatasetFromNPZ(os.path.join(self.data_dir, "validate"),
                                                feature_labels=self.feature_labels,
                                                file_ext=self.file_ext)
            if self.batch_size == -1:
                self.batch_size = int(np.max([len(self.data_train), len(self.data_validate)]))


        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = DatasetFromNPZ(os.path.join(self.data_dir, "test"),
                                            feature_labels=self.feature_labels,
                                            file_ext=self.file_ext)
            if self.batch_size == -1:
                self.batch_size = len(self.data_test)
    
    def size_train(self):
        return len(self.data_train)

    def size_validate(self):
        return len(self.data_validate)

    def size_test(self):
        return len(self.data_test)

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle_training)

    def val_dataloader(self):
        return DataLoader(self.data_validate,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


class DatasetFromNPZ(Dataset):
    def __init__(self, data_dir, feature_labels=None, transform=None, target_transform=None, file_ext=None):

        file_list      = [os.path.join(data_dir, f)
                          for f in os.listdir(data_dir) if f.endswith("." + file_ext)]
        data_dict_list = [np.load(data_file, allow_pickle=True) for data_file in file_list]

        if feature_labels is not None:
            self.labels = feature_labels
        else:
            self.labels = list(data_dict_list[0].keys())

        self.data   = {key: torch.tensor(np.concatenate([d[key]
                       for d in data_dict_list], axis=0), dtype=torch.float32) for key in self.labels}
        self.shapes = [v[0].shape for _, v in self.data.items()]
        self.len    = len(self.data[self.labels[0]])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return tuple([self.data[key][idx] for key in self.labels])


def main():
    dm = DataModuleFromNPZ(
        data_dir="data_logistic_regression_2d",
        feature_labels=["inputs", "targets"],
        batch_size=-1,
        num_workers=4,
        shuffle_training=False
    )

    dm.prepare_data()
    dm.setup()

    for i, batch in enumerate(dm.train_dataloader()):
        if i < 100:
            print([torch.min(batch[0]), torch.max(batch[0])])
        else:
            break

if __name__ == "__main__":
    main()
