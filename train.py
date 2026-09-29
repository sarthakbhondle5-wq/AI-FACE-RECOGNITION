import argparse
import pytorch_lightning as pl
from parse_config import ConfigParser
from data_loader.data_loaders import FaceDataModule
from model.model import FaceNetModel
from trainer.trainer import Trainer

import torch

def main(config):
    # set seed de dam bao ket qua dong nhat
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    
    # khoi tao datamodule
    dl_cfg = config.get("data_loader", {})
    data_module = FaceDataModule(
        data_dir=dl_cfg.get("data_dir", "data"),
        batch_size=dl_cfg.get("batch_size", 8),
        shuffle=dl_cfg.get("shuffle", True),
        validation_split=dl_cfg.get("validation_split", 0.1),
        num_workers=dl_cfg.get("num_workers", 0),
        img_size=dl_cfg.get("img_size", 160),
        num_synthetic_classes=dl_cfg.get("num_synthetic_classes", 10)
    )
    
    # tu dong quet data de lay so luong class thuc te
    data_module.setup(stage='fit')
    actual_num_classes = data_module.num_classes
    print(f"[Train] Tu dong dieu chinh classifier head thanh {actual_num_classes} classes.")
    
    # cau hinh model facenet
    model_config = {
        "embedding_dim": config.get("embedding_dim", 128),
        "learning_rate": config.get("learning_rate", 0.001),
        "weight_decay": config.get("weight_decay", 0.0005),
        "num_classes": actual_num_classes,
        "loss": config.get("loss", {}),
    }
    
    model = FaceNetModel(model_config)
    
    # khoi tao trainer de quan ly checkpoint va log tensorboard
    trainer = Trainer(model, data_module, config.config)
    
    # bat dau qua trinh train
    best_checkpoint = trainer.train()
    
    print(f"\n[Train] Train xong! Checkpoint tot nhat luu tai: {best_checkpoint}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Face Recognition Training')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='device run (default: auto)')
    args.add_argument('-e', '--epochs', default=None, type=int,
                      help='override epochs')
    args.add_argument('-b', '--batch_size', default=None, type=int,
                      help='override batch size')
    args.add_argument('--lr', default=None, type=float,
                      help='override learning rate')
    args.add_argument('--run_id', default=None, type=str,
                      help='run identifier')
                      
    config = ConfigParser.from_args(args)
    main(config)
