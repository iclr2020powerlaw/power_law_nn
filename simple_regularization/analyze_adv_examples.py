import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn
from simple_architectures import MLP, twoLayerMLP
import copy
torch.set_default_dtype(torch.float64)

# In[]
"Load in adversial examples"
advData = torch.load('data/adv_mnist')
advImages = advData['images']
advLabels = advData['labels']

# In[]
"Plot some images"
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(221)
ax.imshow(advImages[0, :].detach().numpy().reshape(28, 28))
ax = fig.add_subplot(222)
ax.imshow(advImages[5, :].detach().numpy().reshape(28, 28))
ax = fig.add_subplot(223)
ax.imshow(advImages[7, :].detach().numpy().reshape(28, 28))
ax = fig.add_subplot(224)
ax.imshow(advImages[10, :].detach().numpy().reshape(28, 28))
fig.show()

# In[]
"Load in models"
allModels = []
noReals = 25
seeds = [int(i) for i in range(noReals)]
hidden = [100, 300, 500]
two_hidden = [100, 200]

# Load in single layer MLPs first
baseFile = 'models/mnist/_hidden='
lr = 1e-3
for hu in hidden:
    fileNameT = baseFile + str(hu) + '_lr=' + str(lr) + '_seed='
    tempModels = []
    for seed in seeds:
        fileName = fileNameT + str(seed)
        params = torch.load(fileName)['parameters']
        model = MLP(28 * 28, hu, 10)
        model.load_state_dict(params)  # load in parameters
        allModels.append(copy.deepcopy(model))

baseFile = 'models/mnist/two_hidden='
lr = 1e-3
for hu in two_hidden:
    fileNameT = baseFile + str(hu) + '_lr=' + str(lr) + '_seed='
    tempModels = []
    for seed in seeds:
        fileName = fileNameT + str(seed)
        params = torch.load(fileName)['parameters']
        model = twoLayerMLP(28 * 28, hu, 10)
        model.load_state_dict(params)  # load in parameters
        allModels.append(copy.deepcopy(model))

del fileName, fileNameT, tempModels, model

# In[]
softMax = torch.nn.Softmax(dim=1)
totNum = advLabels.shape[0] * 1.0
testResults = np.zeros(len(allModels))
for idx in range(len(allModels)):
    outputs = allModels[idx](advImages)  # do forward pass
    probs = softMax(outputs)
    numCorrect = torch.sum(torch.argmax(probs, dim=1) == advLabels).detach().numpy() * 1.0
    testResults[idx] = numCorrect / totNum
    # testResults[idx, j] = numCorrect / totNum