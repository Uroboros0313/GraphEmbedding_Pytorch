import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .module.walker import RandomWalker
from .module.word2vec import Word2Vec, WeightedWord2Vec