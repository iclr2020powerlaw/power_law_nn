import numpy as np
import numpy.random as npr
from numpy import newaxis as na
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../actual_regularization/')
import utils
npr.seed(0)

# In[]
"Construct input vector"
numPts = 5000  # number of input points
theta = np.linspace(0, 4 * np.pi, numPts)
x = np.vstack((np.cos(theta), np.sin(theta)))
numNeurons = 2000  # number of neurons in each hidden layer
q = 4  # radius thingy?
# In[]
u = np.zeros((2, numNeurons))
u[0, 0] = 1
u[1, 1] = 1
h1 = np.zeros((numPts, numNeurons))
for n in range(numPts):
    h1[n, :] = np.sqrt(numNeurons * q) * (x[0, n] * u[0, :] + x[1, n] * u[1, :])
h1 = h1.T
# In[]
sigmab = 0.3
sigmaws = [1.3, 2.5, 4]
numLayers = 20
W = npr.randn(numNeurons, numNeurons, numLayers) / np.sqrt(numNeurons)
b = sigmab * npr.randn(numNeurons, numLayers)
neuralActivity = []
for sigma in sigmaws:
    hiddens = np.zeros((numNeurons, numPts, numLayers + 1))
    hiddens[:, :, 0] = h1
    for layer in range(numLayers):
        hiddens[:, :, layer + 1] = np.tanh(sigma * W[:, :, layer] @ hiddens[:, :, layer] + b[:, layer][:, na])
    neuralActivity.append(copy.deepcopy(hiddens))

# In[]
"Compute spectra and top 3 eigenvectors"
neuralSpectra = []
neuralEigVectors = []
for neural in neuralActivity:
    spectra = np.zeros((numLayers + 1, numNeurons))
    vectors = np.zeros((numNeurons, 3, numLayers + 1))
    for layer in range(numLayers + 1):
        hidden = neural[:, :, layer] - np.mean(neural[:, :, layer], 1)[:, na]
        cov = hidden @ hidden.T / numPts
        w, v = np.linalg.eigh(cov)
        w = w[::-1]
        v = v[:, ::-1]
        spectra[layer, :] = w
        vectors[:, :, layer] = v[:, :3]
    neuralSpectra.append(copy.deepcopy(spectra))
    neuralEigVectors.append(copy.deepcopy(vectors))

# In[]
"Look at layers 5, 10, 15"
layers = [5, 10, 15]
start = 100
fin = 500
j = 0
fig = plt.figure(figsize=(12, 8))
axes = []
axes.append(fig.add_subplot(331, projection='3d'))
axes.append(fig.add_subplot(332, projection='3d'))
axes.append(fig.add_subplot(333, projection='3d'))
for idx in range(3):
    xT = neuralEigVectors[j][:, :, layers[idx]].T @ neuralActivity[j][:, :, layers[idx]]
    axes[idx].plot(xT[0, :], xT[1, :], xT[2, :])
    axes[idx].set_xticklabels([])
    axes[idx].set_yticklabels([])
    axes[idx].set_zticklabels([])
    if idx == 2:
        axes[idx].set_zlabel(r'$\sigma_w= $' + str(sigmaws[j]), labelpad=-10)
    "Compute alpha"
    m, c = utils.estimate_slope(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    axes[idx].set_title('Layer ' + str(layers[idx]) + '\n' + str(m) + ' ' + str(idx))


# In[]
start = 500
fin = 1200
j = 1
axes = []
axes.append(fig.add_subplot(334, projection='3d'))
axes.append(fig.add_subplot(335, projection='3d'))
axes.append(fig.add_subplot(336, projection='3d'))
for idx in range(3):
    xT = neuralEigVectors[j][:, :, layers[idx]].T @ neuralActivity[j][:, :, layers[idx]]
    axes[idx].plot(xT[0, :], xT[1, :], xT[2, :])
    axes[idx].set_xticklabels([])
    axes[idx].set_yticklabels([])
    axes[idx].set_zticklabels([])
    if idx == 2:
        axes[idx].set_zlabel(r'$\sigma_w= $' + str(sigmaws[j]), labelpad=-10)
    "Compute alpha"
    m, c = utils.estimate_slope(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    axes[idx].set_title(str(m))

# In[]
j = 2
axes = []
axes.append(fig.add_subplot(337, projection='3d'))
axes.append(fig.add_subplot(338, projection='3d'))
axes.append(fig.add_subplot(339, projection='3d'))
for idx in range(3):
    xT = neuralEigVectors[j][:, :, layers[idx]].T @ neuralActivity[j][:, :, layers[idx]]
    axes[idx].plot(xT[0, :], xT[1, :], xT[2, :])
    axes[idx].set_xticklabels([])
    axes[idx].set_yticklabels([])
    axes[idx].set_zticklabels([])
    if idx == 2:
        axes[idx].set_zlabel(r'$\sigma_w= $' + str(sigmaws[j]), labelpad=-10)
    "Compute alpha"
    m, c = utils.estimate_slope(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    axes[idx].set_title(str(m))
fig.show()







# In[]
"Look at layers 5, 10, 15"
layers = [5, 10, 15]
start = 100
fin = 1000
j = 0
fig = plt.figure(figsize=(12, 8))
axes = []
axes.append(fig.add_subplot(331))
axes.append(fig.add_subplot(332))
axes.append(fig.add_subplot(333))
for idx in range(3):
    axes[idx].loglog(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    m, c = utils.estimate_slope(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    logy = c + m * np.log(np.arange(start + 1, fin + 1))
    axes[idx].loglog(np.arange(start + 1, fin + 1), np.exp(logy))
    axes[idx].set_xticklabels([])
    if idx == 0:
        axes[idx].set_ylabel(r'$\sigma_w= $' + str(sigmaws[j]))


# In[]
start = 500
fin = 1200
j = 1
axes = []
axes.append(fig.add_subplot(334))
axes.append(fig.add_subplot(335))
axes.append(fig.add_subplot(336))
for idx in range(3):
    axes[idx].loglog(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    m, c = utils.estimate_slope(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    logy = c + m * np.log(np.arange(start + 1, fin + 1))
    axes[idx].loglog(np.arange(start + 1, fin + 1), np.exp(logy))
    axes[idx].set_xticklabels([])
    if idx == 0:
        axes[idx].set_ylabel(r'$\sigma_w= $' + str(sigmaws[j]))

# In[]
j = 2
axes = []
axes.append(fig.add_subplot(337))
axes.append(fig.add_subplot(338))
axes.append(fig.add_subplot(339))
for idx in range(3):
    axes[idx].loglog(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    m, c = utils.estimate_slope(np.arange(start + 1, fin + 1), neuralSpectra[j][layers[idx], start:fin])
    logy = c + m * np.log(np.arange(start + 1, fin + 1))
    axes[idx].loglog(np.arange(start + 1, fin + 1), np.exp(logy))
    if idx == 0:
        axes[idx].set_ylabel(r'$\sigma_w= $' + str(sigmaws[j]))
fig.show()
