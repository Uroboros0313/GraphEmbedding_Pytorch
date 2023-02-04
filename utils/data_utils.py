import os
import pickle as pkl
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_data(data_dir, dataset='wiki'):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    
    ################################# wiki数据集预处理 ###################################
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
    
    ################################# cosearch数据集预处理 ###################################
    elif dataset == 'cosearch':
        G = nx.Graph()
        with open(data_dir / 'cosearch/co_search_2016.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            edge_list = []
            for edge_info in lines:
                sc, tg, w = edge_info.strip().split(' ')
                edge_list.append((sc, tg, int(w)))
            G.add_weighted_edges_from(edge_list)
            
    ################################# jdata数据集预处理 ###################################          
    
    elif dataset == 'jdata':
        tmp_path = data_dir / 'tmp/jdata/jdata_graph.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                G = pkl.load(f)
        else:
            window_size = 5
            G = nx.Graph()

            user_actions = pd.read_csv(data_dir/'jdata/action_head.csv', parse_dates=['action_time'], header=0)
            user_actions = user_actions[['user_id', 'sku_id', 'action_time']]

            items = pd.read_csv(data_dir/'jdata/jdata_product.csv', header=0)
            items = items[['sku_id','brand','shop_id','cate']]

            user_actions = user_actions.merge(items, on='sku_id', how='inner')
            items = user_actions[['sku_id','brand','shop_id','cate']].drop_duplicates()
            user_actions = user_actions[['user_id', 'sku_id']]

            sessions = user_actions.groupby('user_id').agg(list)
            for vis_info in sessions.itertuples():
                sess = vis_info[1]
                if len(sess) == 1:
                    G.add_node(sess[0])
                    continue

                for i in range(len(sess)):
                    for j in range(max(0, i - window_size), min(len(sess), i + window_size)):
                        if i==j:
                            continue
                        t, v = sess[i], sess[j]
                        if t not in G.nodes or v not in G.nodes:
                            G.add_edge(t, v, weight=1)
                        elif v not in G[t]:
                            G.add_edge(t, v, weight=1)
                        else:
                            G[t][v]['weight'] += 1
                            
            for node_info in items.itertuples():
                G.nodes[node_info.sku_id]['brand'] = node_info.brand
                G.nodes[node_info.sku_id]['shop_id'] = node_info.shop_id
                G.nodes[node_info.sku_id]['cate'] = node_info.cate
            
            os.mkdir(tmp_path.parent)
            with open(tmp_path, 'wb') as f:
                pkl.dump(G, f)
                
    ################################# cora数据集预处理 ###################################
    elif dataset == 'cora':
        G = nx.Graph()
        with open(data_dir / 'cora/cora.cites', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            edge_list = [tuple(edge_info.strip().split('\t')) for edge_info in lines]
            G.add_edges_from(edge_list)
        
        with open(data_dir / 'cora/cora.content', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            nodes = []
            feats = []
            labels = []
            for feature_info in lines:
                features = feature_info.strip().split('\t')
                nodes.append(features[0])
                feats.append([int(col_feat) for col_feat in features[1: -1]])
                labels.append(features[-1])
        
        unique_labels = list(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}   
        
        for node, feat, label in zip(nodes, feats, labels):
            G.nodes[node]['feature'] = np.asarray(feat)
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
    
    plt.figure(figsize=(10, 5)) 
    plt.scatter(xs, ys, s=4, c=colors)
    plt.title(title)
    plt.savefig(save_path)
    
    
        
            
                
                
                
                
        
        
    
