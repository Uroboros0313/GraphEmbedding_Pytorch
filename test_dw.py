import os
import pathlib
import warnings

import torch
from easydict import EasyDict

from utils.data_utils import *
from utils.option_utils import *
from ge.models import DeepWalk


warnings.filterwarnings('ignore')

dw_params = EasyDict()
dw_params.WALK_LEN = 10
dw_params.NUM_WALK = 50
dw_params.WALK_TYPE = 'random'
dw_params.LR = 0.025
dw_params.EMBEDDING_DIM = 128
dw_params.WINDOW_SIZE = 5
dw_params.EPOCHS = 5
dw_params.BATCH_SIZE = 10000
dw_params.NUM_NEG = 5
dw_params.SEED = 2023

DATASET = 'cora'

FILE_NAME = ['_'.join([par.lower(), str(val)]) for par, val in dw_params.items()]
FILE_NAME = '-'.join([f'ds_{DATASET}'] + FILE_NAME)

MODEL_PATH = pathlib.Path('./checkpoints/deepwalk/{}.pt'.format(FILE_NAME))
PICS_PATH = pathlib.Path('./pics/deepwalk/{}.png'.format(FILE_NAME))

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
G = load_data(data_dir='./data', dataset=DATASET)


if __name__=="__main__":
    set_seed(dw_params.SEED)
    dw = DeepWalk(G,
                walk_len=dw_params.WALK_LEN,
                num_walk=dw_params.NUM_WALK,
                walk_type=dw_params.WALK_TYPE,
                lr=dw_params.LR,
                embedding_dim=dw_params.EMBEDDING_DIM,
                window_size=dw_params.WINDOW_SIZE,
                epochs=dw_params.EPOCHS,
                batch_size=dw_params.BATCH_SIZE,
                num_neg=dw_params.NUM_NEG,
                device=DEVICE,)
    
    if not os.path.exists(MODEL_PATH):
        dw.train()
        if not os.path.exists(MODEL_PATH.parent):
            os.makedirs(MODEL_PATH.parent, exist_ok=True)
        dw.save_model(MODEL_PATH)
    else:
        dw.load_model(MODEL_PATH)
    
    if not os.path.exists(PICS_PATH.parent):
        os.makedirs(PICS_PATH.parent, exist_ok=True)
        
    embedding_dict = dw.get_all_embeddings()
    colors = []
    for node in embedding_dict:
        try:
            c = int(G.nodes[node]['label'])
            colors.append(c)
        except:
            colors = None
            break
    
    visualize_embeddings(embedding_dict, PICS_PATH, FILE_NAME, colors)
