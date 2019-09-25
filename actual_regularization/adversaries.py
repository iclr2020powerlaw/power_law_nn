import torch
import utils
import copy
import numpy as np


class Adversary:
    "Class that will be in charge of generating batches of adversarial images"
    def __init__(self, eps=0.3, alpha=0.01, gradSteps=100, noRestarts=0, cuda=False):
        """
        Constructor for a first order adversary
        :param eps: radius of l infinity ball
        :param alpha: learning rate
        :param gradSteps: number of gradient steps
        :param noRestarts: number of restarts
        """
        self.eps = eps  # radius of l infinity ball
        self.alpha = alpha.cuda() if cuda == True else alpha.cpu()  # learning rate
        self.gradSteps = gradSteps  # number of gradient steps to take
        self.noRestarts = noRestarts  # number of restarts
        self.cuda = cuda

    def generateAdvImages(self, x_nat, y, network, loss):
        """
        Given a batch of images and labels, it creates an adversarial batch
        :param x_nat: original image
        :param y: label
        :param network: neural network
        :param loss: function for loss function to be used
        :return: x, a tensor containing the adversarial images
        """
        if self.eps == 0:
            x = x_nat
            ell = loss(x, y)
        elif self.gradSteps == 0 or self.alpha == 0:  # FGSM by Goodfellow et al 2018
            jacobian, ell = utils.get_jacobian(network, x_nat, y, loss, cuda=self.cuda)  # get jacobian
            x = x_nat + self.eps * torch.sign(jacobian)
        else:
            losses = torch.zeros(self.noRestarts)
            xs = []
            for r in range(self.noRestarts):
                perturb = 2 * self.eps * torch.rand(x_nat.shape, device=x_nat.device) - self.eps
                xT, ellT = self.pgd(x_nat, x_nat + perturb, y, network, loss)  # do pgd
                xs.append(xT)
                losses[r] = ellT
            idx = torch.argmax(losses)
            x = xs[idx]  # choose the one with the largest loss function
            ell = losses[idx]
        return x, ell

    def pgd(self, x_nat, x, y, network, loss):
        """
        Perform projected gradient descent from Madry et al 2018
        :param x_nat: starting image
        :param x: starting point for optimization
        :param y: true label of image
        :param network: network
        :param loss: loss function
        :return: x, the maximum found
        """
        for i in range(self.gradSteps):
            # if self.cuda:
            #     x = x.cuda()
            jacobian, ell = utils.get_jacobian(network, copy.deepcopy(x), y, loss, cuda=self.cuda)  # get jacobian
            x += self.alpha * torch.sign(jacobian)  # take gradient step
            # if self.cuda:
            #     x = x.cpu()
            #     x_nat = x_nat.cpu()
            # xT = x.detach().numpy()
            xT = x.detach()
            xT = utils.clip(xT, x_nat.detach() - self.eps,
                            x_nat.detach() + self.eps)
            xT = torch.clamp(xT, 0, 1)
            x = xT
            # xT = np.clip(xT, x_nat.detach().numpy() - self.eps, x_nat.detach().numpy() + self.eps)
            # x = torch.from_numpy(xT)

        # if self.cuda:
        #     x = x.cuda()
        ell = loss(x, y)
        return x, ell.item()
