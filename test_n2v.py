import os
import pathlib
import warnings

import torch
from easydict import EasyDict

from utils.data_utils import *
from utils.option_utils import *
from ge.node2vec import Node2Vec


warnings.filterwarnings('ignore')

n2v_params = EasyDict()
n2v_params.WALK_LEN = 10
n2v_params.NUM_WALK = 50
n2v_params.P = 0.5
n2v_params.Q = 2
n2v_params.WALK_TYPE = 'biased'
n2v_params.LR = 0.025
n2v_params.EMBEDDING_DIM = 128
n2v_params.WINDOW_SIZE = 5
n2v_params.EPOCHS = 5
n2v_params.BATCH_SIZE = 10000
n2v_params.NUM_NEG = 5
n2v_params.SEED = 2023

DATASET = 'cora'

FILE_NAME = ['_'.join([par.lower(), str(val)]) for par, val in n2v_params.items()]
FILE_NAME = '-'.join([f'ds_{DATASET}'] + FILE_NAME)

MODEL_PATH = pathlib.Path('./checkpoints/node2vec/{}.pt'.format(FILE_NAME))
PICS_PATH = pathlib.Path('./pics/node2vec/{}.png'.format(FILE_NAME))

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
G = load_data(data_dir='./data', dataset=DATASET)


if __name__=="__main__":
    set_seed(n2v_params.SEED)
    n2v = Node2Vec(G,
                walk_len=n2v_params.WALK_LEN,
                num_walk=n2v_params.NUM_WALK,
                walk_type=n2v_params.WALK_TYPE,
                lr=n2v_params.LR,
                p=n2v_params.P,
                q=n2v_params.Q,
                embedding_dim=n2v_params.EMBEDDING_DIM,
                window_size=n2v_params.WINDOW_SIZE,
                epochs=n2v_params.EPOCHS,
                batch_size=n2v_params.BATCH_SIZE,
                num_neg=n2v_params.NUM_NEG,
                device=DEVICE,)
    
    if not os.path.exists(MODEL_PATH):
        n2v.train()
        if not os.path.exists(MODEL_PATH.parent):
            os.makedirs(MODEL_PATH.parent, exist_ok=True)
        n2v.save_model(MODEL_PATH)
    else:
        n2v.load_model(MODEL_PATH)
    
    if not os.path.exists(PICS_PATH.parent):
        os.makedirs(PICS_PATH.parent, exist_ok=True)
        
    embedding_dict = n2v.get_all_embeddings()
    colors = []
    for node in embedding_dict:
        try:
            c = int(G.nodes[node]['label'])
            colors.append(c)
        except:
            colors = None
            break
    
    visualize_embeddings(embedding_dict, PICS_PATH, FILE_NAME, colors)
