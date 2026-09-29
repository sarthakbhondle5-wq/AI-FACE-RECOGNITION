import pytorch_lightning as pl
from abc import abstractmethod

class BaseDataLoader(pl.LightningDataModule):
    """
    Class co ban cho tat ca cac data loader (DataModule) trong project.
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_workers = num_workers

    @abstractmethod
    def setup(self, stage=None):
        """
        Chuan bi va thiet lap cac dataset (train, val, test splits).
        """
        pass

    @abstractmethod
    def train_dataloader(self):
        """
        Tra ve training dataloader.
        """
        pass

    @abstractmethod
    def val_dataloader(self):
        """
        Tra ve validation dataloader.
        """
        pass

    @abstractmethod
    def test_dataloader(self):
        """
        Tra ve test dataloader.
        """
        pass
