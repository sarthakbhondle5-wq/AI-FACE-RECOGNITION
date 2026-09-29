import torch

def triplet_accuracy(anchor, positive, negative):
    """
    Tinh toan do chinh xac cua thu tu triplet.
    Dung neu khoang cach (anchor, positive) < (anchor, negative).
    """
    with torch.no_grad():
        pos_dist = torch.pow(anchor - positive, 2).sum(dim=1)
        neg_dist = torch.pow(anchor - negative, 2).sum(dim=1)
        
        # so luong triplet dung
        correct = (pos_dist < neg_dist).float()
        return correct.mean()

def mean_pos_dist(anchor, positive):
    """
    Tinh khoang cach L2 trung binh cho cac cap positive.
    """
    with torch.no_grad():
        dist = torch.sqrt(torch.pow(anchor - positive, 2).sum(dim=1) + 1e-8)
        return dist.mean()

def mean_neg_dist(anchor, negative):
    """
    Tinh khoang cach L2 trung binh cho cac cap negative.
    """
    with torch.no_grad():
        dist = torch.sqrt(torch.pow(anchor - negative, 2).sum(dim=1) + 1e-8)
        return dist.mean()
