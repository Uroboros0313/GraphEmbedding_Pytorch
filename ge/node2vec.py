import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .module.walker import RandomWalker
from .module.layers import Word2Vec
from .module.data import PosPairDataset
        
    
class Node2Vec:
    def __init__(
        self,
        G,
        walk_len=10,
        num_walk=30,
        walk_type='biased',
        p=0.5,
        q=2,
        lr=0.01,
        embedding_dim=128,
        window_size=5,
        epochs=5,
        batch_size=10000,
        num_neg=5,
        use_noise_dist=True,
        ns_exponent=0.75,
        device=torch.device('cpu')
        ) -> None:
        
        self.G = G
        self.walk_len = walk_len
        self.num_walk = num_walk
        self.walk_type = walk_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_neg = num_neg
        self.use_noise_dist = use_noise_dist
        self.ns_exponent = ns_exponent
        self.device = device
        
        self.nodes = list(G.nodes)
        self.idx2node = dict([(idx, node) for idx, node in enumerate(list(G.nodes))])
        self.node2idx = dict([(node, idx) for idx, node in self.idx2node.items()])
        self.w2v_model = Word2Vec(vocab_size=len(self.nodes), embedding_dim=embedding_dim).to(device)
        self.random_walker = RandomWalker(G, walk_len, num_walk, walk_type=walk_type, p=p, q=q)
        self.optimizer = optim.SparseAdam(self.w2v_model.parameters(), lr=lr, )
        
    def train(self):
        sentences = self.random_walker.gen_walks()
        
        if self.use_noise_dist == True:
            self.__init_noise_weights('frequency', sentences)
        else:
            self.noise_weights = torch.ones(len(self.nodes)).view(1, )
        
        print('INFO: Generating Positive Pairs...')
        pairs = self.__gen_positive_pairs(sentences)
        
        dataset = PosPairDataset(pairs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, pin_memory=False, )
        
        print('INFO: Start Training Word2Vec...')
        self.w2v_model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for pos_u, pos_v in tqdm(dataloader):
                batch_size = pos_u.shape[0]
                neg_vs = self.__gen_negative_samples(batch_size, self.num_neg)
                
                pos_u, pos_v, neg_vs =\
                    pos_u.to(self.device), pos_v.to(self.device), neg_vs.to(self.device) 
                ns_loss = self.__train(pos_u, pos_v, neg_vs)
                epoch_loss += ns_loss
            
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print("Epoch: {}, Loss: {:.4f}, memory usage: {:.4f} MiB".\
                    format(epoch + 1, epoch_loss, gpu_mem_alloc))
    
    def __gen_positive_pairs(self, sentences):
        pairs = []
        for sent in tqdm(sentences):
            pairs.extend(self.__gen_sentence_pairs(sent))
        return pairs
    
    def save_model(self, file_path):
        torch.save(self.w2v_model.state_dict(), file_path)
        print('Saved w2v_model in {}...'.format(file_path))
        
    def load_model(self, file_path):
        self.w2v_model.load_state_dict(torch.load(file_path))
        print('Loaded w2v_model in {}...'.format(file_path))
    
    def get_embedding(self, node):
        self.w2v_model.eval()
        with torch.no_grad():
            embedding = self.w2v_model.U.weight[self.node2idx[node], :]
            embedding = embedding.cpu().numpy()
        return embedding
    
    def get_all_embeddings(self, ):
        embedding_dict = {}
        nodes = self.nodes
        for node in nodes:
            embedding_dict[node] = self.get_embedding(node)
            
        return embedding_dict
        
    def __train(self, pos_u, pos_v, neg_vs):
        ns_loss = self.w2v_model(pos_u, pos_v, neg_vs)
        self.optimizer.zero_grad()
        ns_loss.backward()
        self.optimizer.step()
        
        return ns_loss.item()
    
    def __gen_sentence_pairs(self, sent):
        pairs = []
        sent_len = len(sent)
        if sent_len <= 1:
            return []
        
        for tg_idx in range(sent_len):
            tg_word = sent[tg_idx]
            for cont_idx in range(max(0, tg_idx - self.window_size), 
                                  min(sent_len, tg_idx + self.window_size)):
                if sent[cont_idx] != tg_word:
                    pairs.append((
                            self.node2idx[tg_word], 
                            self.node2idx[sent[cont_idx]]))
        return pairs
    
    def __gen_negative_samples(self, batch_size, num_neg):
        neg_vs = torch.multinomial(
            self.noise_weights, num_neg * batch_size, replacement=True)\
                .view(batch_size, num_neg)
        return neg_vs
    
    def __init_noise_weights(self, method='degree', sentences=[]):
        '''
        节点度作为weight和frequency作为weight的方法
        '''
        noise_weights = []
        if method == 'degree':
            for node in self.nodes:
                noise_weights.append(self.G.degree(node))
                noise_weights = np.asarray(noise_weights).astype(np.float)
                
        elif method == 'frequency':
            if sentences == None or len(sentences) == 0:
                raise ValueError('sentences is not existing.')
            
            noise_weights = np.asarray([0] * len(self.nodes)).astype(np.float)
            num_sent = len(sentences)
            if num_sent > 10000:
                sampled_sentences = random.sample(sentences, 10000)
            else:
                sampled_sentences = sentences
            
            for sent in sampled_sentences:
                for node in sent:
                    noise_weights[self.node2idx[node]] += 1
                noise_weights[noise_weights==0] = 1

        noise_weights /= np.sum(noise_weights)
        noise_weights = noise_weights ** self.ns_exponent
        noise_weights /= np.sum(noise_weights)
        
        self.noise_weights = torch.from_numpy(noise_weights)
        
            
        
                
                
                
                
                
                
    
    
