import os
from abc import abstractmethod

class BaseTrainer:
    """
    Class trainer co ban de thiet lap training cho PyTorch Lightning.
    """
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        # cau hinh thu muc luu tru
        self.save_dir = config.get('save_dir', 'saved/models')
        self.log_dir = config.get('log_dir', 'saved/log')
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        """
        Dinh nghia vong lap train.
        """
        pass
