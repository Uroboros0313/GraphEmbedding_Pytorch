import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import NegativeSamplingLoss

class SageConv(nn.Module):
    def __init__(self):
        super().__init__()
        
        
class GrpahSageModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    
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
    def __init__(
        self,
        vocab_size,
        info_sizes,
        embedding_dim,
        use_embedding_weight=True) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W0 = nn.Embedding(vocab_size, embedding_dim)
        self.WSs = nn.ModuleList(
            [nn.Embedding(info_size, embedding_dim) for info_size in info_sizes]
            )
        self.V = nn.Embedding(vocab_size, embedding_dim)
        self.ns_loss = NegativeSamplingLoss()
        
        if use_embedding_weight == True:
            self.alpha = nn.parameter.Parameter(torch.randn((len(info_sizes) + 1), 1), requires_grad=True)
            nn.init.xavier_uniform_(self.alpha, gain=nn.init.calculate_gain('sigmoid'))
        else:
            self.alpha = nn.parameter.Parameter(torch.ones((len(info_sizes) + 1), 1), requires_grad=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.W0.weight,
                         -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        for WS in self.WSs:
            nn.init.uniform_(WS.weight,
                         -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
            
        nn.init.uniform_(self.V.weight,
                         -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        
            
    def forward(self, pos_u_idxs, pos_v_idxs, neg_v_idxs, pos_s_idxs):
        pos_u = self.get_u_vecs(pos_u_idxs)
        pos_s = self.get_u_side_vecs(pos_s_idxs)
        
        batch_size, embedding_dim = pos_u.shape
        pos_u = pos_u.view(batch_size, 1, embedding_dim)
        
        pos_ws = torch.cat([pos_u, pos_s], dim=1)
        pos_ws = torch.mul(pos_ws, self.alpha.exp() / self.alpha.exp().sum())
        pos_ws = torch.mean(pos_ws, dim=1,).squeeze(1)
        
        
        pos_v = self.get_v_vecs(pos_v_idxs)
        neg_v = self.get_v_vecs(neg_v_idxs)
        loss = self.ns_loss(pos_ws, pos_v, neg_v)
        return loss

    def get_v_vecs(self, idxs):
        return self.V(idxs)
    
    def get_u_vecs(self, idxs):
        return self.W0(idxs)
    
    def get_u_side_vecs(self, idxs):
        batch_size, info_size = idxs.shape
        info_embeds = []
        for i in range(info_size):
            info_embed = self.WSs[i](idxs[:, i])
            _, embedding_dim = info_embed.shape
            info_embeds.append(
                info_embed.view(batch_size, 1, embedding_dim))
            
        ws = torch.cat(info_embeds, dim=1)
        return ws
    
    
    
        
        
        
        
        