import torch
import pytorch_lightning as pl
from abc import abstractmethod

class BaseModel(pl.LightningModule):
    """
    Class co ban cho tat ca cac model trong project.
    Dinh nghia cau truc chuan cho cac buoc train, validation va test.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x):
        """
        Forward pass cua model.
        """
        pass

    def training_step(self, batch, batch_idx):
        """
        Buoc train cua PyTorch Lightning.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """
        Buoc validation cua PyTorch Lightning.
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """
        Cau hinh optimizers.
        """
        raise NotImplementedError
        
    def __str__(self):
        """
        In thong tin chi tiet model.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.numel(p) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'
