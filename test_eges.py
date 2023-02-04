import os
import pathlib
import warnings

import torch
from easydict import EasyDict

from utils.data_utils import *
from utils.option_utils import *
from ge.models import EGES


warnings.filterwarnings('ignore')

eges_params = EasyDict()
eges_params.WALK_LEN = 5
eges_params.NUM_WALK = 10
eges_params.WALK_TYPE = 'weighted'
eges_params.LR = 0.025
eges_params.EMBEDDING_DIM = 128
eges_params.WINDOW_SIZE = 5
eges_params.EPOCHS = 5
eges_params.BATCH_SIZE = 10000
eges_params.NUM_NEG = 5
eges_params.SEED = 2023

DATASET = 'jdata'

FILE_NAME = ['_'.join([par.lower(), str(val)]) for par, val in eges_params.items()]
FILE_NAME = '-'.join([f'ds_{DATASET}'] + FILE_NAME)

MODEL_PATH = pathlib.Path('./checkpoints/eges/{}.pt'.format(FILE_NAME))
PICS_PATH = pathlib.Path('./pics/eges/{}.png'.format(FILE_NAME))

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
G = load_data(data_dir='./data', dataset=DATASET)


if __name__=="__main__":
    set_seed(eges_params.SEED)
    eges = EGES(G,
                walk_len=eges_params.WALK_LEN,
                num_walk=eges_params.NUM_WALK,
                walk_type=eges_params.WALK_TYPE,
                lr=eges_params.LR,
                embedding_dim=eges_params.EMBEDDING_DIM,
                window_size=eges_params.WINDOW_SIZE,
                epochs=eges_params.EPOCHS,
                batch_size=eges_params.BATCH_SIZE,
                num_neg=eges_params.NUM_NEG,
                device=DEVICE,)
    
    if not os.path.exists(MODEL_PATH):
        eges.train()
        if not os.path.exists(MODEL_PATH.parent):
            os.makedirs(MODEL_PATH.parent, exist_ok=True)
        eges.save_model(MODEL_PATH)
    else:
        eges.load_model(MODEL_PATH)
    
    if not os.path.exists(PICS_PATH.parent):
        os.makedirs(PICS_PATH.parent, exist_ok=True)
    
    
    embedding_dict = eges.get_all_embeddings()
    colors = []
    for node in embedding_dict:
        try:
            c = int(G.nodes[node]['label'])
            colors.append(c)
        except:
            colors.append(1)
    
    cold_feats = [7163, 8952, 7]
    cold_embedding = eges.get_cold_start_embedding(cold_feats)
    embedding_dict['cold_start'] = cold_embedding
    colors.append(2)
    
    visualize_embeddings(embedding_dict, PICS_PATH, FILE_NAME, colors)
