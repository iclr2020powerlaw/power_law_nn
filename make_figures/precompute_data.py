import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
import copy
import sys
sys.path.append('../actual_regularization/')
from models import MLP, CNN
import utils
import torch.nn as nn
from adversaries import Adversary
# In[]
"Load in all single layer feed-forward architectures"
noReals = 10
activations = ['relu', 'tanh']  # 2 different types of relu
seeds = [int(i) for i in range(noReals)]  # seeds for RNGs
singleArchitectures = [[(28 * 28, 100), (100, 10)],
                         [(28 * 28, 500), (500, 10)],
                         [(28 * 28, 1000), (1000, 10)],
                         [(28 * 28, 10000), (10000, 10)]]
pathDir = 'data/models/data_analysis/mlp/mlp_'
singleMlp = []
singleMlpModels = []
for activation in activations:
    architectMlp = []
    architectMlpModel = []
    pathBase = pathDir + activation + '_hidden=('
    for architect in singleArchitectures:
        pathSecond = pathBase + str(architect[0][-1]) + ',10)_lr=0.001_alpha=0.0_batch_size=200_seed='
        model = MLP(dims=architect, activation=activation)  # create an object
        seedMlp = []
        seedMlpModel = []
        for seed in seeds:
            pathT = pathSecond + str(seed) + '_epochs=10000'
            modelTemp = torch.load(pathT)
            model.load_state_dict(modelTemp['parameters'])
            seedMlp.append(copy.deepcopy(modelTemp))
            seedMlpModel.append(copy.deepcopy(model))
        architectMlp.append(copy.deepcopy(seedMlp))
        architectMlpModel.append(copy.deepcopy(seedMlpModel))
    singleMlp.append(copy.deepcopy(architectMlp))
    singleMlpModels.append(copy.deepcopy(architectMlpModel))

# In[]
"Load in all two layers feed-forward architectures"
doubleArchitectures = [[(28 * 28, 100), (100, 100), (100, 10)],
                         [(28 * 28, 200), (200, 200), (200, 10)]]
pathDir = 'data/models/data_analysis/mlp/mlp_'
doubleMlp = []
doubleMlpModels = []
for activation in activations:
    architectMlp = []
    architectMlpModel = []
    pathBase = pathDir + activation + '_hidden=('
    for architect in doubleArchitectures:
        pathSecond = pathBase + str(architect[0][-1]) + ',' + str(architect[1][-1]) +\
                     ',10)_lr=0.001_alpha=0.0_batch_size=200_seed='
        model = MLP(dims=architect, activation=activation)  # create an object
        seedMlp = []
        seedMlpModel = []
        for seed in seeds:
            pathT = pathSecond + str(seed) + '_epochs=10000'
            modelTemp = torch.load(pathT)
            model.load_state_dict(modelTemp['parameters'])
            seedMlp.append(copy.deepcopy(modelTemp))
            seedMlpModel.append(copy.deepcopy(model))
        architectMlp.append(copy.deepcopy(seedMlp))
        architectMlpModel.append(copy.deepcopy(seedMlpModel))
    doubleMlp.append(copy.deepcopy(architectMlp))
    doubleMlpModels.append(copy.deepcopy(architectMlpModel))

# In[]
"Load in all CNNs"
cnnArchitectures = [[1, (1, 12), (1728, 128), (128, 10)],
                    [1, (1, 28), (4032, 128), (128, 10)],
                    [2, (1, 32), (32, 64), (1024, 1024), (1024, 10)]]
pathDir = 'data/models/data_analysis/cnn/cnn_'
cnn = []
cnnModels = []
for activation in activations:
    architectCNN = []
    architectCNNModel = []
    pathBase = pathDir + activation + '_hidden=('
    for architect in cnnArchitectures:
        architectT = architect[1:]
        pathSecond = pathBase
        for idx in range(len(architectT) - 1):
            pathSecond += str(architectT[idx][-1]) + ','
        pathSecond += '10)_lr=0.001_alpha=tensor([0.], device=\'cuda:0\')_batch_size=200_seed='
        model = CNN(dims=architect, activation=activation)
        seedCNN = []
        seedCNNModel = []
        for seed in seeds:
            pathT = pathSecond + str(seed) + '_epochs=10000'
            modelTemp = torch.load(pathT, map_location=torch.device('cpu'))
            model.load_state_dict(modelTemp['parameters'])
            seedCNN.append(copy.deepcopy(modelTemp))
            seedCNNModel.append(copy.deepcopy(model))
        architectCNN.append(copy.deepcopy(seedCNN))
        architectCNNModel.append(copy.deepcopy(seedCNNModel))
    cnn.append(copy.deepcopy(architectCNN))
    cnnModels.append(copy.deepcopy(architectCNNModel))

# In[]
"Load in data"
mnist_data = np.load('../simple_regularization/data/mnist.npy')[()]
trainData, trainLabels = mnist_data['train']  # training set
testData, testLabels = mnist_data['test']
totData = torch.cat((trainData, testData), 0)
totLabels = torch.cat((trainLabels, testLabels))

# In[]
"Compute the eigenspectra of single layer mlp"
singleMlpSpectra = []
for activation in range(len(singleMlpModels)):
    activationSpectra = []
    for arch in range(len(singleMlpModels[activation])):
        realizationSpectra = []
        for model in singleMlpModels[activation][arch]:
            hidden, _ = model.bothOutputs(totData)
            tempSpectra = []
            for h in hidden:
                h = h.detach().numpy()
                h = h - np.mean(h, 0)
                cov = h.T @ h / h[:, 0].size
                cov = (cov + cov.T) / 2
                temp, _ = np.linalg.eigh(cov)
                tempSpectra.append(copy.deepcopy(temp[::-1]))
            realizationSpectra.append(copy.deepcopy(tempSpectra[::-1]))
        activationSpectra.append(copy.deepcopy(realizationSpectra))
    singleMlpSpectra.append(copy.deepcopy(activationSpectra))

# In[]
"Compute the eigenspectra of double layer mlp"
doubleMlpSpectra = []
for activation in range(len(doubleMlpModels)):
    activationSpectra = []
    for arch in range(len(doubleMlpModels[activation])):
        realizationSpectra = []
        for model in doubleMlpModels[activation][arch]:
            hidden, _ = model.bothOutputs(totData)
            tempSpectra = []
            for h in hidden:
                h = h.detach().numpy()
                h = h - np.mean(h, 0)
                cov = h.T @ h / h[:, 0].size
                cov = (cov + cov.T) / 2
                temp, _ = np.linalg.eigh(cov)
                tempSpectra.append(copy.deepcopy(temp[::-1]))
            realizationSpectra.append(copy.deepcopy(tempSpectra[::-1]))
        activationSpectra.append(copy.deepcopy(realizationSpectra))
    doubleMlpSpectra.append(copy.deepcopy(activationSpectra))

# In[]
"Compute the eigenspectra of cnn"
cnnSpectra = []
for activation in range(len(cnnModels)):
    activationSpectra = []
    for arch in range(len(cnnModels[activation])):
        realizationSpectra = []
        for model in cnnModels[activation][arch]:
            hidden, _ = model.bothOutputs(totData)
            tempSpectra = []
            for h in hidden:
                h = h.detach().numpy()
                h = h - np.mean(h, 0)
                cov = h.T @ h / h[:, 0].size
                cov = (cov + cov.T) / 2
                temp, _ = np.linalg.eigh(cov)
                tempSpectra.append(copy.deepcopy(temp[::-1]))
            realizationSpectra.append(copy.deepcopy(tempSpectra))
        activationSpectra.append(copy.deepcopy(realizationSpectra))
    cnnSpectra.append(copy.deepcopy(activationSpectra))


# In[]
"Save computed stuff so don't have to waste time again"
computedData = {'single': (singleMlp, singleMlpModels, singleMlpSpectra, singleArchitectures),
                'double': (doubleMlp, doubleMlpModels, doubleMlpSpectra, doubleArchitectures),
                'cnn': (cnn, cnnModels, cnnSpectra, cnnSpectra)}
np.save('data/easypeesy/computed_data_analysis', computedData, allow_pickle=True)
