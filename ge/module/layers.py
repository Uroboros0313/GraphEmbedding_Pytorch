import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.U = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.V = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.ns_loss = NegativeSamplingLoss()
        
    def forward(self, pos_u_idxs, pos_v_idxs, neg_v_idxs):
        pos_u = self.get_u_vecs(pos_u_idxs)
        pos_v = self.get_v_vecs(pos_v_idxs)
        neg_v = self.get_v_vecs(neg_v_idxs)
        loss = self.ns_loss(pos_u, pos_v, neg_v)
        return loss

    def get_v_vecs(self, idxs):
        return self.V(idxs)
    
    def get_u_vecs(self, idxs):
        return self.U(idxs)


class WeightedWord2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,):
        pass
    
        
        
        
        
        