import numpy as np
import numpy.random as npr
import torch


def create_batches(batch_size, numSamples):
    """
    Will create an iterable that will partition data into random batches for SGD
    :param batch_size: number of elements in a batch
    :param numSamples: number of total samples
    :return: a list of indices
    """
    indices = npr.choice(numSamples, numSamples).astype('int')
    numBatches = np.ceil(numSamples / batch_size).astype('int')
    idx = [indices[i * batch_size: (i + 1) * batch_size] for i in range(numBatches)]
    return idx


def get_jacobian(net, x, target, loss, cuda=False):
    """
    Function for computing the input-output jacobian of a neural network
    :param net: neural network
    :param x: input image that has beeen compressed into an array
    :param target: target label
    :param loss: corresponding loss function
    :param cuda: boolean variable deciding whether to do stuff on the gpu
    :return: jacobian loaded back on the cpu
    """
    device = 'cuda' if cuda == True else 'cpu'
    x.requires_grad_(True)
    y = net(x.to(device)).to(device)
    ell = loss(y, target.to(device))
    ell.backward()
    return x.cpu().grad.data.squeeze()


def generate_adv_images(model, images, targets, lossFunc, eps=50, cuda=False, rand=True):
    """
    Given images, targets and a model will create adversarial images
    :param model: neural network class
    :param images: input images that are to be corrupted
    :param targets: target labels for the images
    :param lossFunc: corresponding loss function
    :param eps: parameter for bad images
    :param cuda: boolean variable for using GPU
    :return:
    """
    jacob = get_jacobian(model, images, targets, lossFunc, cuda=cuda)
    if rand:
        advImages = images + eps * torch.rand(images.shape[0], 1) * torch.sign(jacob)
    else:
        advImages = images + eps * torch.sign(jacob)
    return advImages

