import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from base.base_data_loader import BaseDataLoader
from utils.util import generate_synthetic_faces

class TripletFaceDataset(Dataset):
    """
    Dataset tra ve cac triplet (Anchor, Positive, Negative, Label)
    dung de train Triplet Loss.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        
        # gom nhom index theo tung class de de chon Positive va Negative
        self.label_to_indices = {}
        is_subset = isinstance(self.base_dataset, Subset)
        for idx in range(len(self.base_dataset)):
            if is_subset:
                original_idx = self.base_dataset.indices[idx]
                label = self.base_dataset.dataset.samples[original_idx][1]
            else:
                label = self.base_dataset.samples[idx][1]
                
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        
        # canh bao neu co class nao co it hon 2 anh (in gộp để tránh làm rác terminal)
        few_samples_classes = [label for label, idxs in self.label_to_indices.items() if len(idxs) < 2]
        if few_samples_classes:
            print(f"[Warning] Co {len(few_samples_classes)}/{len(self.labels)} classes chi co 1 anh. Voi cac class nay, Positive se trung voi Anchor.")

    def __getitem__(self, index):
        # lay anh anchor va label cua no
        anchor_img, label = self.base_dataset[index]
        
        # chon ngau nhien mot anh cung class de lam positive
        pos_indices = self.label_to_indices[label]
        if len(pos_indices) > 1:
            pos_idx = random.choice([idx for idx in pos_indices if idx != index])
        else:
            pos_idx = index
        pos_img, _ = self.base_dataset[pos_idx]

        # chon ngau nhien mot anh khac class de lam negative
        neg_labels = [l for l in self.labels if l != label]
        if not neg_labels:
            neg_img, _ = self.base_dataset[index]
        else:
            neg_label = random.choice(neg_labels)
            neg_idx = random.choice(self.label_to_indices[neg_label])
            neg_img, _ = self.base_dataset[neg_idx]

        # apply transform neu co
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img, label

    def __len__(self):
        return len(self.base_dataset)


class FaceDataModule(BaseDataLoader):
    """
    DataModule quan ly load, split train/val, va batching.
    """
    def __init__(self, data_dir="data", batch_size=16, shuffle=True, validation_split=0.2, 
                 num_workers=0, img_size=160, num_synthetic_classes=10):
        super().__init__(data_dir, batch_size, shuffle, validation_split, num_workers)
        self.img_size = img_size
        self.num_synthetic_classes = num_synthetic_classes
        
        # augmentation cho tap train
        self.train_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # transform cho tap val/test (khong augment)
        self.val_test_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 0

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        casia_dir = os.path.join(self.data_dir, "CASIA-WebFace")
        
        # tu dong sinh data gia lap neu khong co san data that
        if not os.path.exists(casia_dir) or len(os.listdir(casia_dir)) == 0:
            print(f"[Dataset] Khong tim thay CASIA-WebFace tai '{casia_dir}'.")
            generate_synthetic_faces(self.data_dir, num_classes=self.num_synthetic_classes, images_per_class=20)
            
        # load dataset dang ImageFolder
        base_dataset = ImageFolder(root=casia_dir)
        self.num_classes = len(base_dataset.classes)
        print(f"[Dataset] Da load ImageFolder tu '{casia_dir}' voi {self.num_classes} classes va {len(base_dataset)} anh.")
        
        # chia train/val theo tung class de dam bao cac class deu co mat
        label_to_indices = {}
        for idx, (_, label) in enumerate(base_dataset.samples):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
            
        train_indices = []
        val_indices = []
        
        for label, idxs in label_to_indices.items():
            random.seed(42)
            random.shuffle(idxs)
            split_idx = int(len(idxs) * self.validation_split)
            if split_idx == 0 and len(idxs) >= 2:
                split_idx = 1
            
            val_indices.extend(idxs[:split_idx])
            train_indices.extend(idxs[split_idx:])
            
        # tao cac subset
        train_subset = Subset(base_dataset, train_indices)
        val_subset = Subset(base_dataset, val_indices)
        
        # boc vao Triplet dataset
        self.train_dataset = TripletFaceDataset(train_subset, transform=self.train_transform)
        self.val_dataset = TripletFaceDataset(val_subset, transform=self.val_test_transform)
        self.test_dataset = TripletFaceDataset(val_subset, transform=self.val_test_transform)
        
        print(f"[Dataset] Thong tin chia: Train = {len(self.train_dataset)}, Val/Test = {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
