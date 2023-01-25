import random
from itertools import chain

import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel

from .sampler import AliasSampler

            
class RandomWalker():
    def __init__(
        self,
        G,
        walk_len,
        num_walk,
        p=1,
        q=1,
        walk_type='random',
        ) -> None:
        
        self.G = G
        self.walk_len = walk_len
        self.num_walk = num_walk
        self.p = p
        self.q = q
        self.walk_type = walk_type # ['random', 'weighted', 'biased', 'rejection']
        
        self.nodes_alias_sampler = None
        self.edges_alias_sampler = None
    
    def gen_walks(self):
        self.init_alias()
        nodes = list(self.G.nodes)
        
        if self.walk_type == 'random':
            opt = self.__random_walk
        elif self.walk_type == 'weighted':
            opt = self.__weighted_random_walk
        elif self.walk_type == 'biased':
            opt = self.__biased_walk
        elif self.walk_type == 'rejection':
            opt = self.__reject_sampling_walk
        
        print('INFO: Generating Walks...')
        walks = []
        for node in tqdm(nodes):
            walks.extend(opt(node))  
        
        '''
        results = Parallel(n_jobs = 10, require='sharedmem')(
            delayed(opt)(node) for node in tqdm(nodes))
        walks = list(chain(*results))
        '''
        return walks

    def __random_walk(self, node):
        node_walks = []
        for i in range(self.num_walk): 
            walk = [node]
            cur = node
            for j in range(self.walk_len):
                neighs = list(self.G.neighbors(cur))
                if len(neighs) <= 0: break
                
                next = random.choice(neighs)
                walk.append(next)
                cur = next
            node_walks.append(walk)
            
        return node_walks
    
    def __weighted_random_walk(self, node):
        node_walks = []
        for i in range(self.num_walk): 
            walk = [node]
            cur = node
            for j in range(self.walk_len):
                neighs = list(self.G.neighbors(cur))
                next_idx = self.nodes_alias_sampler.alias_sample(cur)
                next = neighs[next_idx]
                walk.append(next)
                cur = next
            node_walks.append(walk)
            
        return node_walks
    
    def __biased_walk(self, node):
        node_walks = []
        for i in range(self.num_walk): 
            walk = [node]
            cur = node
            for j in range(self.walk_len):
                neighs = list(self.G.neighbors(cur))
                if len(walk) == 1:
                    next_idx = self.nodes_alias_sampler.alias_sample(cur)
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_idx = self.edges_alias_sampler.alias_sample(edge)
                next = neighs[next_idx]
                walk.append(next)    
                cur = next
                
            node_walks.append(walk)
        
        return node_walks
    
    def __reject_sampling_walk(self, node):
        pass
    
    def init_alias(self):
        if self.walk_type == 'random':
            return
        
        elif self.walk_type == 'weighted':
            self.__init_nodes_alias()
            
        elif self.walk_type == 'rejection':
            self.__init_nodes_alias()
            
        elif self.walk_type == 'biased':
            self.__init_nodes_alias()
            self.__init_edges_alias()
            
    def __init_nodes_alias(self):
        '''
            生成nodes的Alias采样表
        '''
        def _get_node_alias(node):
            '''
                单个node的alias采样表生成函数
            '''
            probs = [G[node][neigh].get('weight', 1) \
                for neigh in G.neighbors(node)]
            p_sum = sum(probs)
            norm_probs = [float(prob) / p_sum for prob in probs]
            alias_sampler.add_alias_table(node, norm_probs)
            return
            
        G = self.G
        nodes = list(G.nodes)
        alias_sampler = AliasSampler()
        
        print('INFO: Initialize node alias tables...')
        for node in tqdm(nodes):
            _get_node_alias(node)
            
        self.nodes_alias_sampler = alias_sampler
        
    def __init_edges_alias(self):
        '''
            生成edges的Alias采样表
        '''
        def _get_edge_alias(t, v):
            '''
                单条edge的alias采样表生成函数
            '''
            probs = []
            for x in G.neighbors(v):
                weight = G[v][x].get('weight', 1)
                if x == t:
                    probs.append(weight / p)
                elif G.has_edge(x, t):
                    probs.append(weight)
                else:
                    probs.append(weight / q)
            p_sum = sum(probs)
            norm_probs = [float(prob) / p_sum \
                for prob in probs]
            alias_sampler.add_alias_table((t, v), norm_probs)
            return
        
        G = self.G
        edges = G.edges
        p = self.p
        q = self.q
        alias_sampler = AliasSampler()
        
        print('INFO: Initialize edge alias tables...')
        for t, v in tqdm(edges):
            _get_edge_alias(t, v)
            if not G.is_directed():
                _get_edge_alias(v, t)
        
        self.edges_alias_sampler = alias_sampler
