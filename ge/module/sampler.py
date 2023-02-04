# -*- encoding: utf-8 -*-
#@File    :   sampler.py
#@Time    :   2023/01/24 04:38:06
#@Author  :   Li Suchi 
#@Email   :   lsuchi@126.com

import numpy as np


class AliasSampler():
    '''
    Alias 算法, 建立一个accept和alias列表
        accept: 选择当前index的阈值概率
        alias: 超出当前index阈值概率时选择的index
    
    '''
    def __init__(self) -> None:
        self.alias_tables = {}
    
    def get_alias_table(self, key):
        return self.alias_tables[key]
    
    def alias_list(self, key, index_list):
        accept, alias = self.alias_tables[key]
        
        result_list = []
        for idx in index_list:
            if np.random.random() < accept[idx]:
                result_list.append(idx)
            else:
                result_list.append(alias[idx])
                
        return result_list
    
    def alias_sample(self, key):
        accept, alias = self.alias_tables[key]
        
        N = len(accept)
        idx, rd = int(np.random.random() * N), np.random.random()
        
        if len(accept) == 0:
            return None
        
        if rd < accept[idx]:
            return idx
        else:
            return alias[idx]
    
    def add_alias_tables_from(self, probs_dict):
        '''
            create alias from dict
        '''
        for key, probs in probs_dict.items():
            self.add_alias_table(key, probs)
            
    def add_alias_table(self, key, probs):
        '''
            add a alias pair from key and probs
        '''
        N = len(probs)
        probs_ = np.asarray(probs) * N
        
        # gen accept and alias candidates
        small, large = [], []
        for i, prob in enumerate(probs_):
            if prob < 1.0: 
                small.append(i)
            else:
                large.append(i)
        
        # gen accept and alias
        accept, alias = [0] * N, [0] * N
        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = probs_[small_idx]
            alias[small_idx] = large_idx
            
            probs_[large_idx] -= (1 - probs_[small_idx])
            if probs_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)
                
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1.0
        
        while large:
            large_idx = large.pop()
            accept[large_idx] = 1.0
        
        self.alias_tables[key] = (accept, alias)
        
        
class NeighborSampler():
    def __init__(self) -> None:
        super().__init__()
    
    
    
    