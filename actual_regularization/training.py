from models import MLP, CNN
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from utils import create_batches, eigen_val_regulate, computeEigVectors, compute_loss, split_mnist

import copy
## Piotr's edits
import os
import shutil
import uuid
from sacred import Experiment
from sacred.observers import MongoObserver
from random_words import RandomWords
import random
import logging
logging.basicConfig(level=logging.DEBUG)

rw = RandomWords()
ex = Experiment(name=rw.random_word() + str(random.randint(0, 100)))
# ex.observers.append(MongoObserver.create(db_name='isonetry-hyperparams2'))  #TODO: fix this so it can connect to
# Ackermann
ex.observers.append(MongoObserver.create(
        url='mongodb://powerLawNN:***@***.***.com/admin?authMechanism=SCRAM-SHA-1',
        db_name='powerLawExpts'))

##
@ex.config  # This is defines the default model parameters
# Instead of putting the default parameters in the function trainMLP we put them here
# NB _seed and _run are protected names in sacred since they are automatically generated
def cfg():
    batch_size = 200
    lr = 1e-3
    dims = [(28 * 28, 1000), (1000, 10)]
    numEpochs = 1000  # TODO: set it back to 100s
    cuda = True if torch.cuda.is_available() else False
    alpha = 0.0
    pathLoad = '/home/***/projects/power_law_nn/data/mnist.npy'
    pathSave = os.path.join('trained_models/data_analysis/mlp/', 'tmp' + str(uuid.uuid4())[:8])  # saves model to unique
    # temp directory
    # we'll be removing the directory after putting the data in the database
    epochSave = 10
    activation = 'relu'
    modelType = 'mlp'
    computeEigVectorsOnline = False
    regularizerFcn = 'default'


@ex.main
def model(batch_size, lr, dims, numEpochs, cuda, alpha, pathLoad, pathSave, epochSave, activation, modelType,
             computeEigVectorsOnline, regularizerFcn, _seed, _run):
    """
    Function for creating and training MLPs on MNIST.
    :param batch_size: specifies batch size
    :param rlr: learning rate of stochastic optimizer
    :param dims: A list of N tuples that specifies the input and output sizes for the FC layers. where the last layer is the output layer
    :param numEpochs: number of epochs to train the network for
    :param cuda: boolean variable that will specify whether to use the GPU or nt
    :param alpha: weight for regularizer on spectra. If 0, the regularizer will not be used
    :param pathLoad: path to where MNIST lives
    :param pathSave: path specifying where to save the models
    :param epochSave: integer specifying how often to save loss
    :param activation: string that specified whether to use relu or not
    :param _seed: seed for RNG
    :param _run: Sacred object that logs the relevant data and stores them to a database

    :param computeEigVectorsOnline: online or offline eig estimator
    :param regularizerFcn: function name that computes the discrepancy between the idealized and empirical eigs
    """
    device = 'cuda' if cuda == True else 'cpu'
    os.makedirs(pathSave, exist_ok=True)
    npr.seed(_seed)
    torch.manual_seed(_seed + 1)
    alpha = alpha * torch.ones(1, device=device)

    "Load in MNIST"
    fracVal = 0.1
    train, val, test = split_mnist(pathLoad, fracVal)
    trainData, trainLabels = train[0], train[1]
    valData, valLabels = val[0], val[1]
    testData, testLabels = test[0], test[1]
    numSamples = trainData.shape[0]

    if modelType == 'mlp':
        model = MLP(dims, activation=activation)  # create a mlp object
    elif modelType == 'cnn':
        model = CNN(dims, activation=activation)  # create a CNN object
    else:
        print('WOAHHHHH RELAX')

    model = model.to(device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    "Objects used to store performance metrics while network is training"
    trainSpectra = []  # store the (estimated) spectra of the network at the end of each epoch
    trainLoss = []  # store the training loss (reported at the end of each epoch on the last batch)
    trainRegularizer = []  # store the value of the regularizer during training
    valLoss = []  # validation loss
    valRegularizer = []  # validation regularizer

    "Sample indices for eigenvectors all at once"
    eigBatchIdx = npr.randint(numSamples, size=(numEpochs + 1, batch_size))

    "Get initial estimate of eigenvectors and check loss"
    with torch.no_grad():
        model.eigVec, loss, spectraTemp, regul = computeEigVectors(model, trainData[eigBatchIdx[0, :], :],
                                                                   trainLabels[eigBatchIdx[0, :]], lossFunction,
                                                                   alpha=alpha, cuda=cuda)
        trainSpectra.append(spectraTemp)  # store computed eigenspectra
        trainLoss.append(loss.cpu().item())  # store training loss
        _run.log_scalar("trainLoss", loss.item())
        _run.log_scalar("trainRegularizer", float(alpha * regul) )
        trainRegularizer.append(alpha * regul)  # store value of regularizer

        "Check on validation set"
        loss, regul = compute_loss(model, valData, valLabels, lossFunction, alpha, cuda=cuda)
        valLoss.append(loss.item())
        _run.log_scalar("valLoss", loss.item())
        valRegularizer.append(regul)
        prevVal = loss.item() + alpha * regul.item()  # use for early stopping
        prevModel = copy.deepcopy(model)

    patience = 0
    howMuchPatience = 4
    "Train that bad boy!"
    for epoch in tqdm(range(numEpochs), desc="Epochs", ascii=True, position=0, leave=False):
        batches = create_batches(batch_size=batch_size, numSamples=numSamples)  # create indices for batches
        for batch in tqdm(batches, desc='Train Batches', ascii=True, position=1, leave=False):
            optimizer.zero_grad()
            "Compute a forward pass through the network"
            loss, regul = compute_loss(model, trainData[batch, :], trainLabels[batch], lossFunction, alpha, cuda=cuda)
            lossR = loss + alpha * regul  # compute augmented loss function
            lossR.backward()  # backprop!
            optimizer.step()  # take a gradient step

        "Recompute estimated eigenvectors"
        with torch.no_grad():
            model.eigVec, loss, spectraTemp, regul = computeEigVectors(model, trainData[eigBatchIdx[epoch + 1, :], :],
                                                                       trainLabels[eigBatchIdx[epoch + 1, :]],
                                                                       lossFunction, alpha=alpha, cuda=cuda)
            trainSpectra.append(spectraTemp)  # store computed eigenspectra
            trainLoss.append(loss.cpu().item())  # store training loss
            _run.log_scalar("trainLoss", loss.item())
            trainRegularizer.append(alpha * regul)  # store value of regularizer
            if (epoch + 1) % epochSave == 0:
                "Check early stopping condition"
                loss, regul = compute_loss(model, valData, valLabels, lossFunction, alpha, cuda=cuda)
                currVal = loss.item() + alpha * regul.item()
                percentImprove = (currVal - prevVal) / prevVal
                if percentImprove > 0:
                    if patience > howMuchPatience:
                        model = prevModel
                        break
                    else:
                        patience += 1

                else:
                    patience = 0
                prevVal = currVal
                prevModel = copy.deepcopy(model)  # save for early stopping
                valLoss.append(loss.item())
                _run.log_scalar("valLoss", loss.item())
                valRegularizer.append(regul.item())
                _run.log_scalar("valRegularizer", regul.item())


    "Check accuracy on test set"
    outputs = model(testData.to(device))
    softMax = nn.Softmax(dim=1)
    probs = softMax(outputs.cpu())
    numCorrect = torch.sum(torch.argmax(probs, dim=1) == testLabels).detach().numpy() * 1.0
    testResult = numCorrect / testData.shape[0] * 100

    "Collect accuracy on validation set"
    outputs = model(valData.to(device))
    softMax = nn.Softmax(dim=1)
    probs = softMax(outputs).cpu()
    numCorrect = torch.sum(torch.argmax(probs, dim=1) == valLabels).detach().numpy() * 1.0
    valAcc = numCorrect / valData.shape[0] * 100
    _run.log_scalar("valAcc", valAcc.item())


    "Save everything for later analysis"
    model_data = {'parameters': model.cpu().state_dict(),
                  'training': (trainLoss, trainRegularizer, trainSpectra),
                  'val': (valLoss, valRegularizer, valAcc),
                  'test': testResult}

    if modelType == 'cnn':
        dims = dims[1:]  # first number is number of convolutional layers
    path = pathSave + modelType + '_' + activation + '_hidden=('
    for idx in range(len(dims) - 1):
        path = path + str(dims[idx][1]) + ','

    path = path + str(dims[-1][1]) + ')_lr=' + str(lr) + '_alpha=' + str(alpha) + '_batch_size=' \
           + str(batch_size) + '_seed=' + str(_seed) + '_epochs=' + str(numEpochs)
    torch.save(model_data, path)
    _run.add_artifact(path, "model_data.pt", content_type="application/octet-stream")  # saves the data dump as model_data
    # os.system('ls -l --block-size=M {}'.format(path))
    # shutil.rmtree(pathSave)
    # Returning the validation loss to do model comparision and selection
    return valAcc


def train(name, args):
    sacredObj = ex.run(config_updates={**args},
                       options={'--name': None})
    return sacredObj.result


def parser():
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', default=200, type=int)
    argparser.add_argument('--lr', default=1e-3, type=float)
    argparser.add_argument('--dims', default=[(28 * 28, 1000), (1000, 10)], nargs='+')
    argparser.add_argument('--numEpochs', default=100, type=int)
    argparser.add_argument('--cuda', default=False, type=bool)
    argparser.add_argument('--alpha', default=0.0, type=float)
    argparser.add_argument('--pathLoad', default=argparse.SUPPRESS, type=str)
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
