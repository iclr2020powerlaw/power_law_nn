from simple_architectures import MLP, twoLayerMLP
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import fire
import copy
from utils import get_jacobian


def generate_adversial_mlp(noReals=25, hidden=[100], two_hidden=[100], eps=256/4, numAdv=1000, cuda=False, seed=0):
    """
    Given trained networks, generate adversial examples :D
    :param noReals: number of networks to take from a given subclass
    :param hiddens: list of hidden units to look at for single hidden layer MLP
    :param two_hiddens: list of hidden units to look at for double hidden layer MLP
    :param eps: epsilon to add to sign(Jacobian)
    :param numAdv: number of adversial images to produce for each network
    :param cuda: boolean variable whether to do computations on GPU
    :return: nothing lol, save to a file
    """
    npr.seed(seed)
    # In[]
    "Load in training data"
    mnist_data = np.load('data/mnist.npy')[()]
    trainData, trainLabels = mnist_data['test']  # training set
    trainData = trainData.double()  # convert from integer to floating point

    # In[]
    "Load in models"
    allModels = []
    # Load in single layer MLPs first
    seeds = [int(i) for i in range(noReals)]
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
    idx = npr.randint(low=0, high=trainLabels.shape[0],
                      size=numAdv)  # choose a whole bunch of images for creating adversarial examples
    labels = trainLabels[idx]
    images = trainData[idx, :]
    advImages = torch.zeros(numAdv, 28 * 28, len(allModels))  # for storing adversarial examples
    lossFunction = nn.CrossEntropyLoss()
    for j in tqdm(range(len(allModels))):
        # Get jacobian
        jacobian = get_jacobian(allModels[j], images, labels, lossFunction, cuda=cuda)
        advImages[:, :, j] = images + eps * torch.sign(jacobian)  # eps * jacobian to original image to create adversial examples
        # jacobian = get_jacobian(allModels[j], advImages[[j], :], advLabels[j].unsqueeze(0), lossFunction, cuda=cuda)
        # advImages[j, :] += eps * torch.sign(jacobian)  # eps * jacobian to original image to create adversial examples

    # In[]
    data = {'images': images,
            'labels': labels,
            'advImages': advImages}
    torch.save(data, 'data/adv_mnist')


if __name__ == '__main__':
    fire.Fire(generate_adversial_mlp)
    print("something")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("something")
