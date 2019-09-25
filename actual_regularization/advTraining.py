from models import MLP, CNN
from adversaries import Adversary
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from utils import create_batches, eigen_val_regulate, computeEigVectors, compute_loss, split_mnist
from itertools import count

## Piotr's edits
import os
import shutil
import uuid
from sacred import Experiment
from sacred.observers import MongoObserver
from random_words import RandomWords
import random
rw = RandomWords()
ex = Experiment(name=rw.random_word() + str(random.randint(0,100)))
# ex.observers.append(MongoObserver.create(db_name='isonetry-hyperparams2'))  #TODO: fix this so it can connect to
# Ackermann
ex.observers.append(MongoObserver.create(
        url='mongodb://powerLawNN:***@***.***.com/admin?authMechanism=SCRAM-SHA-1',
        db_name='powerLawExpts'))
##
@ex.config
def cfg():
    batch_size = 200
    lr = 1e-3
    dims = [1, (1, 28), (4032, 128), (128, 10)]
    numEpochs = 100_000
    eps = 0.3
    alpha = 0.01
    gradSteps = 20
    noRestarts = 1
    cuda = True
    pathLoad = '../simple_regularization/data/mnist.npy'
    pathSave = os.path.join('trained_models/adv/mlp/', 'tmp' + str(uuid.uuid4())[:8])
    epochSave = 10
    activation = 'relu'
    modelType = 'cnn'
    computeEigVectorsOnline = False
    regularizerFcn = 'default'

@ex.main
def advTrain(batch_size, lr, dims, numEpochs, eps, alpha, gradSteps, noRestarts, cuda, pathLoad, pathSave,
             epochSave,  activation, modelType, computeEigVectorsOnline, regularizerFcn,  _seed, _run):
    """
    Function for creating and training NNs on MNIST using adversarial training.
    :param regularizerFcn
    :param computeEigVectorsOnline:
    :param batch_size: specifies batch size
    :param lr: learning rate of stochastic optimizer
    :param dims: A list of N tuples that specifies the input and output sizes for the FC layers. where the last layer
     is the output layer
    :param numEpochs: number of epochs to train the network for
    :param eps: radius of l infinity ball
    :param alpha: learning rate for projected gradient descent. If alpha is 0, then use FGSM
    :param gradSteps: number of gradient steps to take when doing pgd
    :param noRestarts: number of restarts for pgd
    :param cuda: boolean variable that will specify whether to use the GPU or nt
    :param pathLoad: path to where MNIST lives
    :param pathSave: path specifying where to save the models
    :param epochSave: integer specfying how often to save loss
    :param activation: string that specified whether to use relu or not
    :param _seed: seed for RNG
    :param _run: Sacred object that logs the relevant data and stores them to a database
    """
    device = 'cuda' if cuda == True else 'cpu'
    os.makedirs(pathSave, exist_ok=True)
    npr.seed(_seed)
    torch.manual_seed(_seed + 1)
    alpha = alpha * torch.ones(1,device=device)

    "Load in MNIST"
    fracVal = 0.1
    train, val, test = split_mnist(pathLoad, fracVal)
    trainData, trainLabels = train[0], train[1]
    valData, valLabels = val[0], val[1]
    testData, testLabels = test[0], test[1]
    numSamples = trainData.shape[0]

    # In[]
    if modelType == 'mlp':
        mlp = MLP(dims,  activation=activation)  # create a mlp object
    elif modelType == 'cnn':
        mlp = CNN(dims, activation=activation)  # create a CNN object
    else:
        print('WOAHHHHH RELAX')


    mlp.to(device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=lr)


    "Create adversary"
    adv = Adversary(eps=eps, alpha=alpha, gradSteps=gradSteps, noRestarts=noRestarts, cuda=cuda)

    "Objects used to store performance metrics while network is training"
    trainLoss = []  # store the training loss (reported at the end of each epoch on the last batch)

    "Get initial value of loss function"
    tempIdx = npr.randint(numSamples, size=batch_size)
    _, lossAdv = adv.generateAdvImages(trainData[tempIdx, :].to(device), trainLabels[tempIdx].to(device), mlp, lossFunction)
    trainLoss.append(lossAdv)
    _run.log_scalar("trainLoss", float(lossAdv))
    "Train that bad boy!"
    counter = count(0)
    for epoch in tqdm(range(numEpochs), desc="Epochs", ascii=True, position=0, leave=False):
        batches = create_batches(batch_size=batch_size, numSamples=numSamples)  # create indices for batches
        for batch in tqdm(batches, desc='Train Batches', ascii=True, position=1, leave=False):
            "Compute a forward pass through the network"
            # Create adversarial images
            xAdv, _ = adv.generateAdvImages(trainData[batch, :].to(device), trainLabels[batch].to(device), mlp,
                                            lossFunction)
            optimizer.zero_grad()
            outputs = mlp(xAdv)  # feed data forward
            loss = lossFunction(outputs, trainLabels.to(device)[batch])  # compute loss
            loss.backward()  # backprop!
            optimizer.step()  # take a gradient step

            if (epoch + 1) % epochSave == 0:
                trainLoss.append(loss.item())  # store training loss
                _run.log_scalar("trainLoss", loss.item())


    "Check accuracy on test set"
    outputs = mlp(testData.to(device))
    softMax = nn.Softmax(dim=1)
    probs = softMax(outputs.cpu())
    numCorrect = torch.sum(torch.argmax(probs, dim=1) == testLabels).detach().numpy() * 1.0
    testResult = numCorrect / testData.shape[0] * 100

    "Collect accuracy on validation set"
    outputs = mlp(valData.to(device))

    softMax = nn.Softmax(dim=1)
    probs = softMax(outputs).cpu()
    numCorrect = torch.sum(torch.argmax(probs, dim=1) == valLabels).detach().numpy() * 1.0
    valAcc = numCorrect / valData.shape[0] * 100
    _run.log_scalar("valAcc", valAcc.item())


    "Save everything for later analysis"
    model_data = {'parameters': mlp.cpu().state_dict(),
                  'training': trainLoss,
                  'valAcc': valAcc,
                  'test': testResult}
    if modelType == 'cnn':
        dims = dims[1:]  # first number is number of convolutional layers
    path = pathSave + modelType + '_' + activation + '_hidden=('
    for idx in range(len(dims) - 1):
        path = path + str(dims[idx][1]) + ','

    path = path + str(dims[-1][1]) + ')_lr=' + str(lr) + '_alpha=' + str(alpha) + '_batch_size=' \
           + str(batch_size) + '_seed=' + str(_seed) + '_epochs=' + str(numEpochs)
    torch.save(model_data, path)
    _run.add_artifact(path, "model_data")  # saves the data dump as model_data
    # shutil.rmtree(pathSave)
    # Returning the validation loss to do model comparision and selection
    return valAcc


def train(name, args):
    sacredObj = ex.run(config_updates={**args},
                       options={'--name': name})
    return sacredObj.result

def parser():
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', default=200, type=int)
    argparser.add_argument('--lr', default=1e-3, type=float)
    argparser.add_argument('--dims', type=str, default=argparse.SUPPRESS)
    argparser.add_argument('--numEpochs', default=250, type=int)
    argparser.add_argument('--cuda', default=True, type=bool)
    argparser.add_argument('--alpha', default=0.01, type=float)
    argparser.add_argument('--eps', default=0.3, type=float)
    argparser.add_argument('--gradSteps', default=40, type=int)
    argparser.add_argument('--noRestarts', default=1, type=int)
    argparser.add_argument('--pathLoad', default='../simple_regularization/data/mnist.npy', type=str)
    argparser.add_argument('--pathSave', default='trained_models/data_analysis/mlp/', type=str)
    argparser.add_argument('--epochSave', default=10, type=int)
    argparser.add_argument('--activation', default='relu')
    argparser.add_argument('--modelType',
                           choices=['mlp', 'cnn', 'autoencoder'],
                           default='mlp',
                           help='Type of neural network.')
    argparser.add_argument('--runName', default=rw.random_word() + str(random.randint(0, 100)),
                           type=str)
    argparser.add_argument('--computeEigVectorsOnline', default=False, type=bool)
    argparser.add_argument('--regularizerFcn', default='default') # added it just to keep track of experiments
    args = argparser.parse_args()
    args = vars(args)

    name = args['runName']
    args.pop('runName', None)
    args['pathSave'] = os.path.join(args['pathSave'], 'tmp' + str(uuid.uuid4())[:8])
    return name, args


if __name__ == '__main__':
    name, args = parser()
    train(name, args)
