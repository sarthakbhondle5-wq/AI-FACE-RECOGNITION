import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from base.base_model import BaseModel
from model.loss import TripletLoss
from model.metric import triplet_accuracy, mean_pos_dist, mean_neg_dist

class FaceNetModel(BaseModel):
    """
    Mo hinh FaceNet su dung PyTorch Lightning.
    Dung ResNet-18 lam backbone, trich xuat embedding 128 chieu,
    dung Triplet Loss ket hop Cross-Entropy de train nhanh hon.
    """
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()
        
        # lay tham so tu config
        self.embedding_dim = config.get("embedding_dim", 128)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.0005)
        self.num_classes = config.get("num_classes", 10)
        self.margin = config.get("loss", {}).get("margin", 0.5)
        self.cls_weight = config.get("loss", {}).get("cls_weight", 0.5)
        
        # khoi tao backbone resnet-18
        print(f"[Model] Khoi tao FaceNet voi backbone ResNet-18...")
        try:
            # load pre-trained weights tu torchvision
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            print("[Model] Da load pre-trained weights cho ResNet-18.")
        except Exception as e:
            print(f"[Model] Warning: Khong load duoc pre-trained weights ({e}). Khoi tao ngau nhien.")
            self.backbone = models.resnet18(weights=None)
            
        # bo di lop fc cu cua resnet
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # lop linear chieu sang khong gian 128 chieu
        self.embedding_layer = nn.Linear(in_features, self.embedding_dim)
        
        # classifier head ho tro qua trinh hoi tu
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        
        # cac ham loss
        self.criterion_triplet = TripletLoss(margin=self.margin)
        self.criterion_cls = nn.CrossEntropyLoss()

    def forward(self, x):
        # trich xuat embedding va normalize L2
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

    def training_step(self, batch, batch_idx):
        anchor, positive, negative, labels = batch
        
        # trich xuat embedding cua cac anh trong triplet
        anchor_emb = self.forward(anchor)
        pos_emb = self.forward(positive)
        neg_emb = self.forward(negative)
        
        # 1. tinh triplet loss
        loss_triplet = self.criterion_triplet(anchor_emb, pos_emb, neg_emb)
        
        # 2. tinh cross-entropy loss cho anchor de ho tro train
        logits = self.classifier(anchor_emb)
        loss_cls = self.criterion_cls(logits, labels)
        
        # tong loss
        total_loss = loss_triplet + self.cls_weight * loss_cls
        
        # tinh metrics de theo doi
        acc = triplet_accuracy(anchor_emb, pos_emb, neg_emb)
        pos_d = mean_pos_dist(anchor_emb, pos_emb)
        neg_d = mean_neg_dist(anchor_emb, neg_emb)
        
        # ghi log
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_triplet_loss", loss_triplet, on_epoch=True)
        self.log("train_cls_loss", loss_cls, on_epoch=True)
        self.log("train_triplet_acc", acc, on_epoch=True, prog_bar=True)
        self.log("train_pos_dist", pos_d, on_epoch=True)
        self.log("train_neg_dist", neg_d, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative, labels = batch
        
        anchor_emb = self.forward(anchor)
        pos_emb = self.forward(positive)
        neg_emb = self.forward(negative)
        
        loss_triplet = self.criterion_triplet(anchor_emb, pos_emb, neg_emb)
        
        # loss phan loai danh tinh
        logits = self.classifier(anchor_emb)
        loss_cls = self.criterion_cls(logits, labels)
        
        total_loss = loss_triplet + self.cls_weight * loss_cls
        
        acc = triplet_accuracy(anchor_emb, pos_emb, neg_emb)
        pos_d = mean_pos_dist(anchor_emb, pos_emb)
        neg_d = mean_neg_dist(anchor_emb, neg_emb)
        
        # ghi log
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val_triplet_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_pos_dist", pos_d, on_epoch=True)
        self.log("val_neg_dist", neg_d, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        # dung ReduceLROnPlateau de giam lr khi val_loss khong cai thien
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min",
            factor=0.5,
            patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
