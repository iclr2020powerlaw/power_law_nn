import torch.nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from simple_architectures import twoLayerMLP
import copy
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times'
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# In[]
"Load in all models"
allModels = []
hidden = [100, 200]
noReals = 25
seeds = [int(i) for i in range(noReals)]  # seeds for RNGs

# In[]
baseFile = 'models/mnist/two_hidden='
lr = 1e-3
for hu in hidden:
    fileNameT = baseFile + str(hu) + '_lr=' + str(lr) + '_seed='
    tempModels = []
    for seed in seeds:
        fileName = fileNameT + str(seed)
        params = torch.load(fileName)['parameters']
        model = twoLayerMLP(28 * 28, hu, 10)
        model.load_state_dict(params)  # load in parameters
        tempModels.append(copy.deepcopy(model))
    allModels.append(copy.deepcopy(tempModels))

# In[]
"Check test accuracy to make sure they were trained correctly"
mnist_data = np.load('data/mnist.npy')[()]
testData, testLabels = mnist_data['test']  # test set
testData = testData.float()
testResults = np.zeros((len(hidden), len(seeds)))
softMax = torch.nn.Softmax(dim=1)
totNum = testLabels.shape[0] * 1.0
for idx in range(len(hidden)):
    for j in tqdm(range(len(seeds))):
        outputs = allModels[idx][j](testData)  # do forward pass
        probs = softMax(outputs)
        numCorrect = torch.sum(torch.argmax(probs, dim=1) == testLabels).detach().numpy() * 1.0
        testResults[idx, j] = numCorrect / totNum

# In[]
"Pool all data together"
trainData, trainLabels = mnist_data['train']  # training set
trainData = trainData.float()  # convert from integer to floating point

valData, valLabels = mnist_data['val']  # validation set
valData = valData.float()  # convert from integer to floating point

pooledData = torch.cat((torch.cat((trainData, valData), 0), testData), 0)

# In[]
"First layer ad second layers"
hiddenEigVals = [np.zeros((noReals, hidden[i])) for i in range(len(hidden))]
hiddenEigVals2 = [np.zeros((noReals, hidden[i])) for i in range(len(hidden))]

for idx in range(len(hidden)):
    for j in tqdm(range(len(seeds))):
        temp, temp2, _ = allModels[idx][j].bothOutputs(pooledData)
        temp = temp - torch.mean(temp, dim=0)
        cov = (temp.T @ temp) / temp.shape[0]
        eigVals, _ = np.linalg.eig(cov.detach().numpy())
        eigVals = np.sort(eigVals)[::-1]
        hiddenEigVals[idx][j, :] = eigVals

        temp2 = temp2 - torch.mean(temp2, dim=0)
        cov = (temp2.T @ temp2) / temp2.shape[0]
        eigVals, _ = np.linalg.eig(cov.detach().numpy())
        eigVals = np.sort(eigVals)[::-1]
        hiddenEigVals2[idx][j, :] = eigVals

# In[]

fig = plt.figure()
ax = fig.add_subplot(121)
startEig = 10
endEigs = [65, 90]
for idx in range(len(hidden)):
    for j in range(len(seeds)):
        ax.loglog(np.arange(startEig + 1, endEigs[idx] + 1), hiddenEigVals[idx][j, startEig:endEigs[idx]])
ax.set_title('First Hidden Layer')

ax = fig.add_subplot(122)
startEig2 = 10
endEigs2 = [90, 180]
for idx in range(len(hidden)):
    for j in range(len(seeds)):
        ax.loglog(np.arange(startEig2 + 1, endEigs2[idx] + 1), hiddenEigVals2[idx][j, startEig2:endEigs2[idx]])
ax.set_title('Second Hidden Layer')
fig.show()

# In[]
"Compute rate of decay"
alphas = np.zeros((len(hidden), noReals))
alphas2 = np.zeros((len(hidden), noReals))
for idx in range(len(hidden)):
    for j in range(len(seeds)):
        eigT = hiddenEigVals[idx][j, startEig:endEigs[idx]]
        y = np.log(eigT)
        x = np.log(np.arange(startEig + 1, endEigs[idx] + 1))
        x = np.vstack((x, np.ones(x.size)))
        m, c = np.linalg.lstsq(x.T, y)[0]
        alphas[idx, j] = m  # get slope

        eigT = hiddenEigVals2[idx][j, startEig2:endEigs2[idx]]
        y = np.log(eigT)
        x = np.log(np.arange(startEig2 + 1, endEigs2[idx] + 1))
        x = np.vstack((x, np.ones(x.size)))
        m, c = np.linalg.lstsq(x.T, y)[0]
        alphas2[idx, j] = m  # get slope

# In[]
fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(alphas.flatten())
ax.set_title('First layer')

ax = fig.add_subplot(122)
ax.hist(alphas2.flatten())
ax.set_title('Second layer')
fig.show()

# In[]
alphaLower = -1
bLower = min([np.min(np.log(hiddenEigVals[idx][:, 0])) for idx in range(len(hidden))]) - 0.1
xS = np.log(np.arange(1, endEigs[-1] + 1))
yLower = alphaLower * xS + bLower

bLower2 = min([np.min(np.log(hiddenEigVals2[idx][:, 0])) for idx in range(len(hidden))]) - 0.1
yLower2 = alphaLower * xS + bLower2

# In[]
fig = plt.figure()
ax = fig.add_subplot(121)
colors = ['steelblue', 'forestgreen', 'darkred']
for idx in range(len(hidden)):
    for j in range(len(seeds)):
        if j == 0:
            ax.loglog(np.arange(1, endEigs[idx] + 1), hiddenEigVals[idx][j, 0:endEigs[idx]],
                      alpha=0.3, color=colors[idx], label=str(hidden[idx]) + ' HU')
        else:
            ax.loglog(np.arange(1, endEigs[idx] + 1), hiddenEigVals[idx][j, 0:endEigs[idx]],
                      alpha=0.3, color=colors[idx])
ax.loglog(np.exp(xS), np.exp(yLower), color='black')
ax.set_xlabel('PC dimension')
ax.set_ylabel('Variance')
ax.set_title('Spectra of 1st Hidden Layer')
ax.legend()


ax = fig.add_subplot(122)
for idx in range(len(hidden)):
    for j in range(len(seeds)):
        if j == 0:
            ax.loglog(np.arange(1, endEigs[idx] + 1), hiddenEigVals2[idx][j, 0:endEigs[idx]],
                      alpha=0.3, color=colors[idx], label=str(hidden[idx]) + ' HU')
        else:
            ax.loglog(np.arange(1, endEigs[idx] + 1), hiddenEigVals2[idx][j, 0:endEigs[idx]],
                      alpha=0.3, color=colors[idx])

ax.loglog(np.exp(xS), np.exp(yLower2), color='black')
ax.set_xlabel('PC dimension')
# ax.set_ylabel('Variance')
ax.set_title('Spectra of 2nd Hidden Layer')
ax.legend()
fig.tight_layout()
fig.show()
fig.savefig('figures/2hidden.pdf')