import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# In[]
# load in imagenet
dataTransform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
imagenet = torchvision.datasets.ImageNet('path/to/imagenet_root/',
                                         transform=dataTransform)
batchSize = 4200
loadImageNet = DataLoader(imagenet, batch_size=batchSize, shuffle=True, num_workers=8)  # data loader


# In[]
def compute_cov(model, x):
    "Function for computing covariance estimates from batches of data"
    # a = model.features(x)[:3]
    # b = model.features(a)[3:6]
    # c = model.features(b)[6:8]
    # d = model.features(c)[8:10]
    # e = model.features(d)[10:]
    #
    # z = model.avgpool(e)
    # z = torch.flatten(z, 1)
    # f = model.features(z)[:3]
    # g = model.features(f)[3:-1]

    y = model.features(x)
    z = model.avgpool(y)
    z = torch.flatten(z, 1)
    f = model.classifier[:3](z)
    ell = model.classifier[3:-1](f)
    ell -= torch.mean(ell, dim=0)
    z -= torch.mean(z, dim=0)
    f -= torch.mean(f, dim=0)
    return ell, z, f


# In[]
alexnet = models.alexnet(pretrained=True)  # Load in pre-trained AlexNet
alexnet.eval()  # switch to eval mode
alexnet.cuda()  # load network onto GPU

# In[]
maxNumBatches = 30  # number of batches of data needed to estimate covariance values
miniBatches = 100  # split up each batch into smaller batches to make things more manageable
sizeBatch = int(batchSize / miniBatches)
print(sizeBatch)
covEstEll = np.zeros((4096, 4096))  # pre-initialize
covEstZ = np.zeros((batchSize, batchSize))  # pre-initialize
covEstF = np.zeros((4096, 4096))  # pre-initialize
numBatches = 0

with torch.no_grad():
    for step, (images, target) in enumerate(loadImageNet):
        tempEll = torch.zeros((batchSize, 4096))
        tempZ = torch.zeros((batchSize, 256 * 6 * 6))
        tempF = torch.zeros((batchSize, 4096))
        for j in tqdm(range(miniBatches)):
            startIdx = j * sizeBatch
            endIdx = (j + 1) * sizeBatch
            tempEll[startIdx:endIdx, :], tempZ[startIdx:endIdx, :], tempF[startIdx:endIdx, :] = compute_cov(alexnet,
                                                                                images[startIdx:endIdx, :, :, :].cuda())
        tempEll = tempEll.detach().numpy()
        tempZ = tempZ.detach().numpy()
        tempF = tempF.detach().numpy()
        for j in range(covEstEll[:, 0].shape[0]):
            covEstEll[:, j] += tempEll.T @ tempEll[:, j] / maxNumBatches
            covEstF[:, j] += tempF.T @ tempF[:, j] / maxNumBatches

        for j in range(covEstF[:, 0].shape[0]):
            covEstZ[:, j] += tempZ @ tempZ.T[:, j] / maxNumBatches
        numBatches += 1
        covariances = {'conv': covEstZ,
                       'H1': covEstF,
                       'H2': covEstEll}
        np.save('alexNetCov', covariances, allow_pickle=True)
        print(step)
        if numBatches + 3 >= maxNumBatches:
            break


# In[]
"Get eigenvalues of estimated covariance matrix from AlexNet"
eigValsZ, _ = np.linalg.eigh(covEstZ)
eigValsZ = np.sort(eigValsZ)[::-1]

eigValsF, _ = np.linalg.eigh(covEstF)
eigValsF = np.sort(eigValsF)[::-1]

eigValsEll, _ = np.linalg.eigh(covEstEll)
eigValsEll = np.sort(eigValsEll)[::-1]


eig = {'conv': eigValsZ,
       'H1': eigValsF,
       'H2': eigValsEll}
np.save('alexNetEigVals_last', eig, allow_pickle=True)