import torch.nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from simple_architectures import MLP
import copy
import seaborn as sns
color_names = ["windows blue", "leaf green", "red", "orange", 'salmon', 'purple']
colors = sns.xkcd_palette(color_names)
torch.set_default_dtype(torch.float64)
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
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# In[]
"Load in all models"
allModels = []
# hidden = [100, 300, 500, 1000]
hidden = [100, 300, 500, 1000, 3000, 5000]
noReals = 25
seeds = [int(i) for i in range(noReals)]  # seeds for RNGs
reload = True

# In[]
if reload:
    hiddenEigVals = np.load('models/advEig.npy', allow_pickle=True)[()]['eig']
else:
    baseFile = 'models/advmnist/_hidden='
    lr = 1e-3
    for hu in hidden:
        fileNameT = baseFile + str(hu) + '_lr=' + str(lr) + '_seed='
        tempModels = []
        for seed in seeds:
            fileName = fileNameT + str(seed)
            params = torch.load(fileName)['parameters']
            model = MLP(28 * 28, hu, 10)
            model.load_state_dict(params)  # load in parameters
            tempModels.append(copy.deepcopy(model))
        allModels.append(copy.deepcopy(tempModels))

    # In[]
    "Check test accuracy to make sure they were trained correctly"
    mnist_data = np.load('data/mnist.npy')[()]
    testData, testLabels = mnist_data['test']  # test set
    testData = testData.double()
    testResults = np.zeros((len(hidden), len(seeds)))
    softMax = torch.nn.Softmax(dim=1)
    totNum = testLabels.shape[0] * 1.0
    # for idx in range(len(hidden)):
    #     for j in tqdm(range(len(seeds))):
    #         outputs = allModels[idx][j](testData)  # do forward pass
    #         probs = softMax(outputs)
    #         numCorrect = torch.sum(torch.argmax(probs, dim=1) == testLabels).detach().numpy() * 1.0
    #         testResults[idx, j] = numCorrect / totNum

    # In[]
    "Pool all data together"
    trainData, trainLabels = mnist_data['train']  # training set
    trainData = trainData.double()  # convert from integer to floating point

    valData, valLabels = mnist_data['val']  # validation set
    valData = valData.double()  # convert from integer to floating point

    pooledData = torch.cat((torch.cat((trainData, valData), 0), testData), 0)

    # In[]
    hiddenEigVals = [np.zeros((noReals, hidden[i])) for i in range(len(hidden))]
    for idx in range(len(hidden)):
        for j in tqdm(range(len(seeds))):
            temp, _ = allModels[idx][j].bothOutputs(pooledData)
            temp = temp - torch.mean(temp, dim=0)
            cov = (temp.T @ temp) / temp.shape[0]
            eigVals, _ = np.linalg.eigh(cov.detach().numpy())
            eigVals = np.sort(eigVals)[::-1]
            hiddenEigVals[idx][j, :] = eigVals
        # In[]
        "Save eigenvalues T.T"
        eigDict = {'eig': hiddenEigVals}
        np.save('models/advEig', eigDict, allow_pickle=True)


# In[]
fig = plt.figure()
ax = fig.add_subplot(111)
startEig = 10
endEigs = [60, 100, 150, 250, 500, 650]
for idx in range(len(hidden)):
    for j in range(len(seeds)):
        ax.loglog(np.arange(startEig + 1, endEigs[idx] + 1), hiddenEigVals[idx][j, startEig:endEigs[idx]])
fig.show()

# In[]
avgEig = []
for idx in range(len(hidden)):
    avgEig.append(np.mean(hiddenEigVals[idx], 0))
    print(avgEig[idx].shape)

# In[]
"Compute rate of decay (hacky)b"
startEig = 10
# endEig = 70
alphas = np.zeros((len(hidden), noReals))
intercepts = np.zeros((len(hidden), noReals))
for idx in range(len(hidden)):
    for j in range(len(seeds)):
        eigT = hiddenEigVals[idx][j, startEig:endEigs[idx]]
        y = np.log(eigT)
        x = np.log(np.arange(startEig + 1, endEigs[idx] + 1))
        x = np.vstack((x, np.ones(x.size)))
        m, c = np.linalg.lstsq(x.T, y)[0]
        alphas[idx, j] = m  # get slope
        intercepts[idx, j] = c  # get intercept

# In[]
"Compute rate of decay for average eigenvalues"
alphaAvg = np.zeros(len(hidden))
cAvg = np.zeros(len(hidden))
for idx in range(len(hidden)):
    eigT = avgEig[idx][startEig:endEigs[idx]]
    y = np.log(eigT)
    x = np.log(np.arange(startEig + 1, endEigs[idx] + 1))
    x = np.vstack((x, np.ones(x.size)))
    m, c = np.linalg.lstsq(x.T, y)[0]
    alphaAvg[idx] = m  # get slope
    cAvg[idx] = c  # get intercept

# In[]
"Plot 100 HU"

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(231)
ax.cla()
for idx in range(len(seeds)):
    ax.loglog(np.arange(1, endEigs[0] + 1), hiddenEigVals[0][idx, :endEigs[0]], alpha=0.2, color=colors[0])
xs = np.log(np.arange(1, endEigs[0] + 1))
ys = alphaAvg[0] * xs + cAvg[0]
ax.loglog(np.exp(xs), np.exp(ys), color='black', label=r'$\alpha=0.89$')
ax.set_ylabel('Variance')
ax.set_title('100 HU')
ax.legend()

# In[]
"Plot 300 HU"
ax = fig.add_subplot(232)
ax.cla()
j = 1
for idx in range(len(seeds)):
    ax.loglog(np.arange(1, endEigs[j] + 1), hiddenEigVals[j][idx, :endEigs[j]], alpha=0.2, color=colors[j])
xs = np.log(np.arange(1, endEigs[j] + 1))
ys = alphaAvg[j] * xs + cAvg[j]
ax.loglog(np.exp(xs), np.exp(ys), color='black', label=r'$\alpha=0.87$')
ax.set_title('300 HU')
ax.legend()

# In[]
"Plot 500 HU"
ax = fig.add_subplot(233)
ax.cla()
j = 2
for idx in range(len(seeds)):
    ax.loglog(np.arange(1, endEigs[j] + 1), hiddenEigVals[j][idx, :endEigs[j]], alpha=0.2, color=colors[j])
xs = np.log(np.arange(1, endEigs[j] + 1))
ys = alphaAvg[j] * xs + cAvg[j]
ax.loglog(np.exp(xs), np.exp(ys), color='black', label=r'$\alpha=0.87$')
ax.set_title(str(hidden[j]) + ' HU')
ax.legend()

# In[]
"Plot 1000 HU"
ax = fig.add_subplot(234)
ax.cla()
j = 3
for idx in range(len(seeds)):
    ax.loglog(np.arange(1, endEigs[j] + 1), hiddenEigVals[j][idx, :endEigs[j]], alpha=0.2, color=colors[j])
xs = np.log(np.arange(1, endEigs[j] + 1))
ys = alphaAvg[j] * xs + cAvg[j]
ax.loglog(np.exp(xs), np.exp(ys), color='black', label=r'$\alpha=0.88$')
ax.set_title(str(hidden[j]) + ' HU')
ax.set_ylabel('Variance')
ax.set_xlabel('PC dimension')
ax.legend()

# In[]
"Plot 3000 HU"
ax = fig.add_subplot(235)
ax.cla()
j = 4
for idx in range(len(seeds)):
    ax.loglog(np.arange(1, endEigs[j] + 1), hiddenEigVals[j][idx, :endEigs[j]], alpha=0.2, color=colors[j])
xs = np.log(np.arange(1, endEigs[j] + 1))
ys = alphaAvg[j] * xs + cAvg[j]
ax.loglog(np.exp(xs), np.exp(ys), color='black', label=r'$\alpha=0.89$')
ax.set_title(str(hidden[j]) + ' HU')
ax.set_xlabel('PC dimension')
ax.legend()

# In[]
"Plot 5000 HU"
ax = fig.add_subplot(236)
ax.cla()
j = 5
for idx in range(len(seeds)):
    ax.loglog(np.arange(1, endEigs[j] + 1), hiddenEigVals[j][idx, :endEigs[j]], alpha=0.2, color=colors[j])
xs = np.log(np.arange(1, endEigs[j] + 1))
ys = alphaAvg[j] * xs + cAvg[j]
ax.loglog(np.exp(xs), np.exp(ys), color='black', label=r'$\alpha=0.88$')
ax.set_title(str(hidden[j]) + ' HU')
ax.set_xlabel('PC dimension')
ax.legend()
fig.tight_layout()
fig.show()
fig.savefig('figures/adv_mlp.pdf', transparency=True)

# In[]
"Plot histogram of 100 HU"
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(231)
j = 0
ax.hist(alphas[j], color=colors[j])
ax.set_title(str(hidden[j]) + ' HU')

# In[]
"Plot histogram of 300 HU"
ax = fig.add_subplot(232)
j = 1
ax.hist(alphas[j], color=colors[j])
ax.set_title(str(hidden[j]) + ' HU')

# In[]
"Plot histogram of 500 HU"
ax = fig.add_subplot(233)
j = 2
ax.hist(alphas[j], color=colors[j])
ax.set_title(str(hidden[j]) + ' HU')
# In[]
"Plot histogram of 1000 HU"
ax = fig.add_subplot(234)
j = 3
ax.hist(alphas[j], color=colors[j])
ax.set_title(str(hidden[j]) + ' HU')
ax.set_xlabel(r'$\alpha$')
# In[]
"Plot histogram of 3000 HU"
ax = fig.add_subplot(235)
j = 4
ax.hist(alphas[j], color=colors[j])
ax.set_title(str(hidden[j]) + ' HU')
ax.set_xlabel(r'$\alpha$')
# In[]
"Plot histogram of 5000 HU"
ax = fig.add_subplot(236)
j = 5
ax.hist(alphas[j], color=colors[j])
ax.set_title(str(hidden[j]) + ' HU')
ax.set_xlabel(r'$\alpha$')
fig.tight_layout()
fig.show()
fig.savefig('figures/adv_mlp_hist.pdf')