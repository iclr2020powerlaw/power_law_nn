import torch
import torch.nn as nn


class MLP(nn.Module):
    # Multilayer perceptron with 1 hidden layer
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.d_in = d_in  # dimension of input
        self.d_hidden = d_hidden  # size of hidden layer
        self.d_out = d_out  # dimension of output layer
        # self.add_module("input_to_hidden", nn.Linear(d_in, d_hidden))
        # self.add_module("hidden_to_output", nn.Linear(d_hidden, d_out))
        self.input_to_hidden = nn.Linear(d_in, d_hidden)  # fully connected from input to hidden layer
        self.hidden_to_output = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        # hiddenX = torch.relu(self.input_to_hidden(x))
        hiddenX = torch.tanh(self.input_to_hidden(x))
        return self.hidden_to_output(hiddenX)

    def bothOutputs(self, x):
        hiddenX = torch.tanh(self.input_to_hidden(x))
        return hiddenX, self.hidden_to_output(hiddenX)


class twoLayerMLP(nn.Module):
    # Multilayer perceptron with 1 hidden layer
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.d_in = d_in  # dimension of input
        self.d_hidden = d_hidden  # size of hidden layer
        self.d_out = d_out  # dimension of output layer
        # self.add_module("input_to_hidden", nn.Linear(d_in, d_hidden))
        # self.add_module("hidden_to_output", nn.Linear(d_hidden, d_out))
        self.input_to_hidden = nn.Linear(d_in, d_hidden)  # fully connected from input to hidden layer
        self.hidden_to_hidden = nn.Linear(d_hidden, d_hidden)
        self.hidden_to_output = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        # hiddenX = torch.relu(self.input_to_hidden(x))
        hiddenX = torch.tanh(self.input_to_hidden(x))
        hiddenY = torch.tanh(self.hidden_to_hidden(hiddenX))
        return self.hidden_to_output(hiddenY)

    def bothOutputs(self, x):
        hiddenX = torch.tanh(self.input_to_hidden(x))
        hiddenY = torch.tanh(self.hidden_to_hidden(hiddenX))
        return hiddenX, hiddenY, self.hidden_to_output(hiddenX)

