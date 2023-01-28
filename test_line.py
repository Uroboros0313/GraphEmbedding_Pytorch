import os
import pathlib
import warnings

import torch
from easydict import EasyDict

from utils.data_utils import *
from utils.option_utils import *
from ge.line import LINE


warnings.filterwarnings('ignore')
        
line_params = EasyDict()
line_params.LR = 0.025
line_params.ORDER = 'second'
line_params.EMBEDDING_DIM = 128
line_params.EPOCHS = 50
line_params.BATCH_SIZE = 2048
line_params.NUM_NEG = 5
line_params.SEED = 2023

DATASET = 'wiki'

FILE_NAME = ['_'.join([par.lower(), str(val)]) for par, val in line_params.items()]
FILE_NAME = '-'.join([f'ds_{DATASET}'] + FILE_NAME)

MODEL_PATH = pathlib.Path('./checkpoints/line/{}.pt'.format(FILE_NAME))
PICS_PATH = pathlib.Path('./pics/line/{}.png'.format(FILE_NAME))

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
G = load_data(data_dir='./data', dataset=DATASET)


if __name__=="__main__":
    set_seed(line_params.SEED)
    line = LINE(
        G,
        order=line_params.ORDER, 
        lr=line_params.LR,
        embedding_dim=line_params.EMBEDDING_DIM, 
        epochs=line_params.EPOCHS,
        batch_size=line_params.BATCH_SIZE,
        num_neg=line_params.NUM_NEG,
        device=DEVICE,)
    
    if not os.path.exists(MODEL_PATH):
        line.train()
        if not os.path.exists(MODEL_PATH.parent):
            os.makedirs(MODEL_PATH.parent, exist_ok=True)
        line.save_model(MODEL_PATH)
    else:
        line.load_model(MODEL_PATH)
    
    if not os.path.exists(PICS_PATH.parent):
        os.makedirs(PICS_PATH.parent, exist_ok=True)
        
    embedding_dict = line.get_all_embeddings()
    colors = []
    for node in embedding_dict:
        try:
            c = int(G.nodes[node]['label'])
            colors.append(c)
        except:
            colors = None
            break
    
    visualize_embeddings(embedding_dict, PICS_PATH, FILE_NAME, colors)

