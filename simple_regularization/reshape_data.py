import torch
import torchvision
import numpy as np
import os
import fire
homedir = os.path.expanduser("~")
data_dir = os.path.join(homedir, 'project/power_law/data')

def download_and_convert(pth=data_dir):
    mnist_data = torchvision.datasets.MNIST('path/to/mnist_root/', download=True)  # load in MNIST for training
    test_mnist_data = torchvision.datasets.MNIST('path/to/mnist_root/', train=False,
                                                 download=True)  # load in MNIST for testing
    "Test data"
    testData = test_mnist_data.test_data.view(test_mnist_data.test_data.shape[0], 28 * 28).float() / 256
    testLabels = test_mnist_data.test_labels
    "80/20 split for training and validation data"
    numTrainSamples = int(1 * mnist_data.train_data.shape[0])  # get number of samples
    numValSamples = int(0 * mnist_data.train_data.shape[0])
    trainData = mnist_data.train_data[:numTrainSamples, :, :].view(numTrainSamples, 28 * 28).float() / 256  # reshape into 2D tensor
    trainLabels = mnist_data.train_labels[:numTrainSamples]

    valData = mnist_data.train_data[numTrainSamples:, :, :].view(numValSamples, 28 * 28).float() / 256
    valLabels = mnist_data.train_labels[numTrainSamples:]

    "Save for easy access later"
    mnist = {'train': (trainData, trainLabels),
             'test': (testData, testLabels)}

    np.save('data/mnist', mnist, allow_pickle=True)
    # print('Done')

if __name__ == '__main__':
    fire.Fire(download_and_convert)