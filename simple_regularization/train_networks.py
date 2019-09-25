from simple_architectures import MLP, twoLayerMLP
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
from torch import optim
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from utils import create_batches, get_jacobian, generate_adv_images
import fire
import copy


def training(batch_size=128, lr=1e-3, d_hidden=300, maxEpochs=100, seed=0, cuda=False,
             advTrain=False, eps=10, alpha=0.5):
    """
    Function that creates a MLP and trains it on mnist
    :param batch_size: batch size for training
    :param lr: learning rate for optimizer
    :param d_hidden: number of units in hidden layer
    :param maxEpochs: maximum number of epochs to use. A validation set is used for early stopping
    :param seed: seed for RNG
    :param cuda: boolean variable that will specify whether to run things on the GPU or not
    :param advTrain: boolean variable denoting whether we should do adversarial training or not
    :param eps: parameter used to create adversarial images
    :param alpha: used to weight the adverserial vs regular loss
    :return: Script will save trained weights, training loss and test loss
    """
    npr.seed(seed)
    torch.manual_seed(seed + 1)
    # In[]
    "load in data"
    mnist_data = np.load('data/mnist.npy')[()]
    trainData, trainLabels = mnist_data['train']  # training set
    trainData = trainData.double()  # convert from integer to floating point

    testData, testLabels = mnist_data['test']  # test set
    testData = testData.double()  # convert from integer to floating point

    valData, valLabels = mnist_data['val']  # validation set
    valData = valData.double()  # convert from integer to floating point

    d_in = 28 * 28  # input dimensions are flattened 28 by 28 images
    d_out = 10  # 10 classes 0 - 9
    numSamples = trainLabels.shape[0]  # number of samples
    # In[]
    mlp = MLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out)  # create a mlp object
    if cuda:
        mlp = mlp.cuda()

    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=lr)

    # In[]
    "Train that bad boy"
    with torch.no_grad():
        if cuda:
            outputs = mlp(valData.cuda())
            loss = lossFunction(outputs, valLabels.cuda())  # check on validation set
        else:
            outputs = mlp(valData)
            loss = lossFunction(outputs, valLabels)  # check on validation set
        currVal = loss.item()

    prevModel = copy.deepcopy(mlp)
    for epoch in tqdm(range(maxEpochs)):
        prevVal = currVal
        batches = create_batches(batch_size=batch_size, numSamples=numSamples)  # create indices for batches
        for batch in batches:
            optimizer.zero_grad()
            if cuda:
                outputs = mlp(trainData[batch, :].cuda())  # feed data forward
                if advTrain:
                    # Generate adversarial image
                    advImage = generate_adv_images(mlp, trainData[batch, :], trainLabels[batch],
                                                   lossFunction, eps=eps, cuda=cuda)
                    outputAdv = mlp(advImage.cuda())  # get output from adversarial image passed in
                    loss = alpha * lossFunction(outputs, trainLabels.cuda()[batch]) \
                           + (1 - alpha) * lossFunction(outputAdv, trainLabels[batch].cuda())
                else:
                    loss = lossFunction(outputs, trainLabels.cuda()[batch])  # compute loss
                loss.backward()  # backprop!
                optimizer.step()  # take a gradient step
            else:
                outputs = mlp(trainData[batch, :])  # feed data forward
                if advTrain:
                    # Generate adversarial image
                    advImage = generate_adv_images(mlp, trainData[batch, :], trainLabels[batch],
                                                   lossFunction, eps=eps, cuda=cuda)
                    outputAdv = mlp(advImage)  # get output from adversarial image passed in
                    loss = alpha * lossFunction(outputs, trainLabels[batch]) \
                           + (1 - alpha) * lossFunction(outputAdv, trainLabels[batch])
                else:
                    loss = lossFunction(outputs, trainLabels[batch])  # compute loss
                loss.backward()  # backprop!
                optimizer.step()  # take a gradient step

        # Check on validation set
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                if cuda:
                    outputs = mlp(valData.cuda())
                    loss = lossFunction(outputs, valLabels.cuda())  # check on validation set
                else:
                    outputs = mlp(valData)
                    loss = lossFunction(outputs, valLabels)  # check on validation set
                currVal = loss.item()
                percentImprove = (currVal - prevVal) / prevVal
                if percentImprove >= 0 and abs(100 * percentImprove) < 1:
                    mlp = prevModel  # use previous model parameters
                    break  # perform early stopping
                else:
                    prevModel = copy.deepcopy(mlp)  # save for early stopping

    # In[]
    "Check on test set"
    if cuda:
        outputs = mlp(testData.cuda())
        testLoss = lossFunction(outputs, testLabels.cuda())
    else:
        outputs = mlp(testData)
        testLoss = lossFunction(outputs, testLabels)

    # In[]
    model_data = {'parameters': mlp.cpu().state_dict(),
                  'test_loss': testLoss.item()}

    # In[]
    "Save model for later"
    if advTrain:
        path = 'models/advmnist/' + '_hidden=' + str(d_hidden) + '_lr=' + str(lr) + '_seed=' + str(seed)
    else:
        path = 'models/mnist/' + '_hidden=' + str(d_hidden) + '_lr=' + str(lr) + '_seed=' + str(seed)
    torch.save(model_data, path)


def training2layers(batch_size=128, lr=1e-3, d_hidden=300, maxEpochs=100, seed=0, cuda=False,
                    advTrain=False, eps=10, alpha=0.5):
    """
    Function that creates a MLP and trains it on mnist
    :param batch_size: batch size for training
    :param lr: learning rate for optimizer
    :param d_hidden: number of units in hidden layer
    :param maxEpochs: maximum number of epochs to use. A validation set is used for early stopping
    :param seed: seed for RNG
    :param cuda: boolean variable that will specify whether to run things on the GPU or not
    :param advTrain: boolean variable denoting whether we should do adversarial training or not
    :param eps: parameter used to create adversarial images
    :param alpha: used to weight the adverserial vs regular loss
    :return: Script will save trained weights, training loss and test loss
    """
    npr.seed(seed)
    torch.manual_seed(seed + 1)
    # In[]
    "load in data"
    mnist_data = np.load('data/mnist.npy')[()]
    trainData, trainLabels = mnist_data['train']  # training set
    trainData = trainData.double()  # convert from integer to floating point

    testData, testLabels = mnist_data['test']  # test set
    testData = testData.double()  # convert from integer to floating point

    valData, valLabels = mnist_data['val']  # validation set
    valData = valData.double()  # convert from integer to floating pointt

    d_in = 28 * 28  # input dimensions are flattened 28 by 28 images
    d_out = 10  # 10 classes 0 - 9
    numSamples = trainLabels.shape[0]  # number of samples
    # In[]
    mlp = twoLayerMLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out)  # create a mlp object
    if cuda:
        mlp = mlp.cuda()

    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-5)

    # In[]
    "Train that bad boy"
    with torch.no_grad():
        if cuda:
            outputs = mlp(valData.cuda())
            loss = lossFunction(outputs, valLabels.cuda())  # check on validation set
        else:
            outputs = mlp(valData)
            loss = lossFunction(outputs, valLabels)  # check on validation set
        currVal = loss.item()

    prevModel = copy.deepcopy(mlp)
    for epoch in tqdm(range(maxEpochs)):
        prevVal = currVal
        batches = create_batches(batch_size=batch_size, numSamples=numSamples)  # create indices for batches
        for batch in batches:
            optimizer.zero_grad()
            if cuda:
                outputs = mlp(trainData[batch, :].cuda())  # feed data forward
                if advTrain:
                    # Generate adversarial image
                    advImage = generate_adv_images(mlp, trainData[batch, :], trainLabels[batch],
                                                   lossFunction, eps=eps, cuda=cuda)
                    outputAdv = mlp(advImage.cuda())  # get output from adversarial image passed in
                    loss = alpha * lossFunction(outputs, trainLabels.cuda()[batch]) \
                           + (1 - alpha) * lossFunction(outputAdv, trainLabels[batch].cuda())
                else:
                    loss = lossFunction(outputs, trainLabels.cuda()[batch])  # compute loss
                loss.backward()  # backprop!
                optimizer.step()  # take a gradient step
            else:
                outputs = mlp(trainData[batch, :])  # feed data forward
                if advTrain:
                    # Generate adversarial image
                    advImage = generate_adv_images(mlp, trainData[batch, :], trainLabels[batch],
                                                   lossFunction, eps=eps, cuda=cuda)
                    outputAdv = mlp(advImage)  # get output from adversarial image passed in
                    loss = alpha * lossFunction(outputs, trainLabels[batch]) \
                           + (1 - alpha) * lossFunction(outputAdv, trainLabels[batch])
                else:
                    loss = lossFunction(outputs, trainLabels[batch])  # compute loss
                loss.backward()  # backprop!
                optimizer.step()  # take a gradient step

        # Check on validation set
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                if cuda:
                    outputs = mlp(valData.cuda())
                    loss = lossFunction(outputs, valLabels.cuda())  # check on validation set
                else:
                    outputs = mlp(valData)
                    loss = lossFunction(outputs, valLabels)  # check on validation set
                currVal = loss.item()
                percentImprove = (currVal - prevVal) / prevVal
                if percentImprove >= 0 and abs(100 * percentImprove) < 1:
                    mlp = prevModel  # use previous model parameters
                    break  # perform early stopping
                else:
                    prevModel = copy.deepcopy(mlp)  # save for early stopping

    # In[]
    "Check on test set"
    if cuda:
        outputs = mlp(testData.cuda())
        testLoss = lossFunction(outputs, testLabels.cuda())
    else:
        outputs = mlp(testData)
        testLoss = lossFunction(outputs, testLabels)

    # In[]
    model_data = {'parameters': mlp.cpu().state_dict(),
                  'test_loss': testLoss.item()}

    # In[]
    "Save model for later"
    if advTrain:
        path = 'models/advmnist/' + 'two_hidden=' + str(d_hidden) + '_lr=' + str(lr) + '_seed=' + str(seed)
    else:
        path = 'models/mnist/' + 'two_hidden=' + str(d_hidden) + '_lr=' + str(lr) + '_seed=' + str(seed)

    torch.save(model_data, path)


# In[]
if __name__ == '__main__':
    fire.Fire(training())
    print("something")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("something")
