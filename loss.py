import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Triplet loss voi Margin.
    L = max(0, d(a, p) - d(a, n) + margin)
    Su dung MarginRankingLoss de tinh gradient on dinh hon.
    """
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        # tinh khoang cach binh phuong L2
        pos_dist = torch.pow(anchor - positive, 2).sum(dim=1)
        neg_dist = torch.pow(anchor - negative, 2).sum(dim=1)
        
        # muon neg_dist > pos_dist + margin -> dung MarginRankingLoss
        y = torch.ones_like(pos_dist)
        loss = self.ranking_loss(neg_dist, pos_dist, y)
        return loss
