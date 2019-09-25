import torch
import torch.nn as nn
# PETER WAS HERE#

class MLP(nn.Module):
    # Multilayer perceptron with a variable amount of hidden layers
    def __init__(self, dims, activation='relu'):
        """
        Constructor for MLP with a variable number of hidden layers
        :param dims: A list of N tuples where the first N -1 determine the N - 1 hidden layers and the last tuple
        determines the output layer
        :param activation: a string that determines which activation layer to use. The default is relu
        """
        super().__init__()
        self.numHiddenLayers = len(dims[:-1])  # number of hidden layers in the network
        modules = []
        for idx in range(len(dims) - 1):
            modules.append(nn.Linear(dims[idx][0], dims[idx][1]))
            if activation == 'relu':
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Tanh())
        modules.append(nn.Linear(dims[-1][0], dims[-1][1]))
        self.sequential = nn.Sequential(*modules)
        self.eigVec = [None] * len(dims[:-1])

    def forward(self, x):
        return self.sequential(x)

    def bothOutputs(self, x):
        hidden = [None] * self.numHiddenLayers
        for idx in range(self.numHiddenLayers):
            if idx == 0:
                hidden[idx] = self.sequential[2 * idx: 2 * idx + 2](x)
            else:
                hidden[idx] = self.sequential[2 * idx: 2 * idx + 2](hidden[idx - 1])

        return hidden, self.sequential[-1](hidden[-1])


class CNN(nn.Module):
    # CNN architecture with 2 convolutional layers and a variable number of fully connected layers
    def __init__(self, dims, activation='relu'):
        """
        Constructor for CNN with 2 convolutional layers and 1 fully connected layer. Assuming 2 by 2 max pooling layer
        :param dims: A list of N tuples where the first element states how many convolutional layers to use
        are defined as (# input channels, kernel size, # output channels)
        :param activation: a string that determines which activation layer to use. The default is relu
        """
        super().__init__()
        self.numConvLayers = dims[0]
        dims = dims[1:]
        self.numHiddenLayers = len(dims) - 1  # number of hidden layers in the network

        # Construct convolutional layers
        convModules = []
        for idx in range(self.numConvLayers):
            convModules.append(nn.Conv2d(dims[idx][0], dims[idx][-1], kernel_size=(5, 5)))
            if activation == 'relu':
                convModules.append(nn.ReLU())
            else:
                convModules.append(nn.Tanh())
            convModules.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.convSequential = nn.Sequential(*convModules)  # convolution layers

        # Construct fully connected layers
        linModules = []
        for idx in range(self.numConvLayers, len(dims) - 1):
            linModules.append(nn.Linear(dims[idx][0], dims[idx][1]))
            if activation == 'relu':
                linModules.append(nn.ReLU())
            else:
                linModules.append(nn.Tanh())
        linModules.append(nn.Linear(dims[-1][0], dims[-1][1]))
        self.linSequential = nn.Sequential(*linModules)
        self.eigVec = [None] * (len(dims) - 1)  # eigenvectors for all hidden layers

    def forward(self, x):
        xT = x.view(x.shape[0], 1, 28, 28)
        hT = self.convSequential(xT)
        return self.linSequential(hT.view(-1, hT.shape[1] * hT.shape[2] * hT.shape[3]))

    def bothOutputs(self, x):
        xT = x.view(x.shape[0], 1, 28, 28)
        hidden = [None] * self.numHiddenLayers
        convHidden = [None] * self.numConvLayers
        for idx in range(self.numConvLayers):
            if idx == 0:
                convHidden[0] = self.convSequential[3 * idx: 3 * idx + 3](xT)
            else:
                convHidden[idx] = self.convSequential[3 * idx: 3 * idx + 3](convHidden[idx - 1])
            hidden[idx] = convHidden[idx].view(-1, convHidden[idx].shape[1] * convHidden[idx].shape[2]
                                               * convHidden[idx].shape[3])

        for idx in range(self.numHiddenLayers - self.numConvLayers):
            if idx == 0:
                hidden[self.numConvLayers] = self.linSequential[2 * idx: 2 * idx + 2](hidden[self.numConvLayers - 1])
            else:
                hidden[self.numConvLayers + idx] = self.linSequential[2 * idx: 2 * idx + 2](hidden[self.numConvLayers
                                                                                                   + idx - 1])

        return hidden, self.linSequential[-1](hidden[-1])
