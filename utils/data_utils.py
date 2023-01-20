from pathlib import Path

import networkx as nx

def load_data(data_dir, dataset='wiki'):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
        
    if dataset == 'wiki':
        G = nx.Graph()
        with open(data_dir / 'wiki\Wiki_edgelist.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            edge_list = [tuple(edge_info.strip().split(' ')) for edge_info in lines]
            G.add_edges_from(edge_list)
        
        with open(data_dir / 'wiki\Wiki_labels.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for label_info in lines:
                node, label = label_info.strip().split(' ')
                G.nodes[node]['label'] = label
                
    elif dataset == 'jdata':
        pass
    elif dataset == 'cora':
        pass
    
    return G      
            
                
                
                
                
        
        
    