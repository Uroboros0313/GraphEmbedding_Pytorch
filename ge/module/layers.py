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
    
    
class LineModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        order='first',):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.order = order
        self.ns_loss = NegativeSamplingLoss()
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.node_embeddings.weight,
                         -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight,
                         -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        
    def forward(self, pos_nodes_idxs, pos_neighs_idxs, neg_neighs_idxs):
        
        pos_nodes = self.node_embeddings(pos_nodes_idxs)
        pos_neighs = self.node_embeddings(pos_neighs_idxs)
        neg_neighs = self.node_embeddings(neg_neighs_idxs)
        
        pos_context = self.context_embeddings(pos_neighs_idxs)
        neg_context = self.context_embeddings(neg_neighs_idxs)
        
        if self.order == 'first':
            return self.ns_loss(pos_nodes, pos_neighs, neg_neighs)
        elif self.order == 'second':
            return self.ns_loss(pos_nodes, pos_context, neg_context)
            

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
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.U.weight,
                         -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        nn.init.uniform_(self.V.weight,
                         -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
            
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


class Word2VecWithSideInfo(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,):
        pass
    
        
        
        
        
        