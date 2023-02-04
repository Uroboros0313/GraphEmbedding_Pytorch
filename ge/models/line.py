import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..module.layers import LineModel
from ..module.sampler import AliasSampler
from ..module.data import PosPairDataset


class LINE():
    # TODO: 修复处理有向边和无向边的Alias方法
    def __init__(
        self,
        G,
        order='first',
        lr=0.01,
        embedding_dim=128,
        epochs=5,
        batch_size=256,
        num_neg=5,
        ns_exponent=0.75,
        device=torch.device('cpu')) -> None:
        
        self.G = G
        self.order = order
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.ns_exponent = ns_exponent
        self.device = device
        
        
        self.edges = list(G.edges)
        if not G.is_directed():
            inv_edges = [(v, t) for t, v in self.edges]
        self.edges.extend(inv_edges)
        
        self.nodes = list(G.nodes)
        self.node2idx = dict([(node, idx) for idx, node in enumerate(self.nodes)])
        self.idx2node = dict([(idx, node) for idx, node in enumerate(self.nodes)])
        self.line_model = LineModel(len(G.nodes), embedding_dim, order=order).to(device)
        self.optimizer = optim.SparseAdam(self.line_model.parameters(), lr=lr, )
        
        self.node_sampler = None
        self.edge_sampler = None
    
    def init_alias(self):
        '''
            初始化负采样Alias和边采样Alias
        '''
        self.__init_node_alias()
        self.__init_edge_alias()
    
    def train(self):
        self.init_alias()
        
        self.line_model.train()
        for epoch in range(self.epochs):
            pairs = self.__gen_positive_pairs()
            dataset = PosPairDataset(pairs)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=False,)
            
            epoch_loss = 0.0
            for pos_nodes, pos_neighs in tqdm(dataloader):
                
                batch_size = pos_nodes.shape[0]
                neg_neighs = self.__gen_negative_samples(batch_size, self.num_neg)
                    
                pos_nodes, pos_neighs, neg_neighs =\
                    pos_nodes.to(self.device), pos_neighs.to(self.device), neg_neighs.to(self.device) 
                ns_loss = self.__train(pos_nodes, pos_neighs, neg_neighs)
                epoch_loss += ns_loss
            
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print("Epoch: {}, Loss: {:.4f}, memory usage: {:.4f} MiB".\
                    format(epoch + 1, epoch_loss, gpu_mem_alloc))
    
    def __train(self, pos_nodes, pos_neighs, neg_neighs):
        ns_loss = self.line_model(pos_nodes, pos_neighs, neg_neighs)
        self.optimizer.zero_grad()
        ns_loss.backward()
        self.optimizer.step()
        
        return ns_loss.item()
    
    def __gen_positive_pairs(self):
        edges = self.edges[: ]
        pairs = []
        shuffle_index = np.random.permutation(np.arange(len(edges)))
        result_list = self.edge_sampler.alias_list('edge', shuffle_index, )
        for edge_idx in result_list:
            t, v = edges[edge_idx]
            pairs.append((self.node2idx[t], self.node2idx[v]))
            
        return pairs   
    
    def __gen_negative_samples(self, batch_size, num_neg):
        batch_negs = []
        for i in range(batch_size):
            negs = []
            for j in range(num_neg):
                negs.append(self.node_sampler.alias_sample('node'))
            batch_negs.append(negs)
            
        return torch.LongTensor(batch_negs)
                
    def __init_edge_alias(self):
        G = self.G
        alias_sampler = AliasSampler()
        
        weights = np.asarray([G[t][v].get('weight', 1.0) for t, v in self.edges])
        norm_weights = weights / np.sum(weights)
        
        alias_sampler.add_alias_table('edge', norm_weights)
        self.edge_sampler = alias_sampler
    
    def __init_node_alias(self):
        G = self.G
        nodes = list(G.nodes)
        nodes_degree = np.zeros(len(nodes), dtype=np.float)
        alias_sampler = AliasSampler()
        
        for t, v in G.edges:
            nodes_degree[self.node2idx[t]] += G[t][v].get('weight', 1)
            if not G.is_directed():
                # 防止无向图出现某些节点出度为0导致负采样采不到的情况
                nodes_degree[self.node2idx[v]] += G[t][v].get('weight', 1)
        
        nodes_degree /= np.sum(nodes_degree)
        nodes_degree **= self.ns_exponent
        norm_degree = nodes_degree / np.sum(nodes_degree)
        
        alias_sampler.add_alias_table('node', norm_degree)
        self.node_sampler = alias_sampler
        
    def save_model(self, file_path):
        torch.save(self.line_model.state_dict(), file_path)
        print('Saved line_model in {}...'.format(file_path))
        
    def load_model(self, file_path):
        self.line_model.load_state_dict(torch.load(file_path))
        print('Loaded line_model in {}...'.format(file_path))
    
    def get_embedding(self, node):
        self.line_model.eval()
        with torch.no_grad():
            if self.order == 'first':
                embedding = self.line_model.node_embeddings.weight[self.node2idx[node], :]
            elif self.order == 'second':
                embedding = self.line_model.context_embeddings.weight[self.node2idx[node], :]
            
            return embedding.cpu().numpy()
    
    def get_all_embeddings(self, ):
        embedding_dict = {}
        nodes = self.nodes
        for node in nodes:
            embedding_dict[node] = self.get_embedding(node)
            
        return embedding_dict
            
            