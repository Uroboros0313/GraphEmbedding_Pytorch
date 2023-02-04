import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
class UnSupSageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def __max_margin_loss(self):
        pass
    
    def __sage_loss(self):
        pass
    

class NegativeSamplingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pos_u, pos_v, neg_v):
        batch_size, embedding_dim = pos_u.shape
        
        pos_u = pos_u.view(batch_size, embedding_dim, 1)
        pos_v = pos_v.view(batch_size, 1, embedding_dim)
        pos_loss = torch.log(torch.sigmoid(torch.bmm(pos_v, pos_u)))
        neg_loss = torch.log(torch.sigmoid(torch.bmm(neg_v.neg(), pos_u)))
        neg_loss = neg_loss.squeeze().sum(dim=1)
        
        return - (pos_loss + neg_loss).mean()