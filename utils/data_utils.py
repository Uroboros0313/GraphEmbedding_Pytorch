from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_data(data_dir, dataset='wiki'):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
        
    if dataset == 'wiki':
        G = nx.Graph()
        with open(data_dir / 'wiki/Wiki_edgelist.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            edge_list = [tuple(edge_info.strip().split(' ')) for edge_info in lines]
            G.add_edges_from(edge_list)
        
        with open(data_dir / 'wiki/wiki_labels.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for label_info in lines:
                node, label = label_info.strip().split(' ')
                G.nodes[node]['label'] = label

    elif dataset == 'cosearch':
        G = nx.Graph()
        with open(data_dir / 'cosearch/co_search_2016.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            edge_list = [tuple(edge_info.strip().split(' ')) for edge_info in lines]
            G.add_weighted_edges_from(edge_list)
              
    elif dataset == 'jdata':
        pass
    
    elif dataset == 'cora':
        G = nx.Graph()
        with open(data_dir / 'cora/cora.cites', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            edge_list = [tuple(edge_info.strip().split(' ')) for edge_info in lines]
            G.add_edges_from(edge_list)
        
        with open(data_dir / 'cora/cora.content', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            nodes = []
            feats = []
            labels = []
            for feature_info in lines:
                features = feature_info.strip().split(' ')
                nodes.append(features[0])
                feats.append([int(col_feat) for col_feat in features[1: -1]])
                labels.append(features[-1])
        
        unique_labels = list(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}   
        
        for node, feat, label in zip(nodes, feats, labels):
            G.nodes[node]['feature'] = feat
            G.nodes[node]['label'] = label_map[label]
    
    return G

def visualize_embeddings(embedding_dict, save_path, title:str=None, colors=None):
    
    print("INFO: Visualize embeddings...")
    embeddings = np.asarray(list(embedding_dict.values()))
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embedding_pos = tsne.fit_transform(embeddings)
    
    xs, ys = [], []
    for x, y in embedding_pos:
        xs.append(x)
        ys.append(y)
    
    plt.figure(figsize=(10, 10)) 
    plt.scatter(xs, ys, s=1, c=colors)
    plt.title(title)
    plt.savefig(save_path)
    
    
        
            
                
                
                
                
        
        
    
