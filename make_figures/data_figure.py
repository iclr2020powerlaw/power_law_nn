import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy.random as npr
import sys
sys.path.append('../actual_regularization/')
from models import MLP, CNN
import utils
# In[]
"Load in architectures"
data = np.load('data/easypeesy/computed_data_analysis.npy', allow_pickle=True)[()]