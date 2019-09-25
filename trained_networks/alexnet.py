"This script will be used for analyzing the properties of CNN AlexNet."
import torch
import torchvision
import torchvision.models as models

# In[]
nThreads = 4  # number of threads used for loading in data
alexnet = models.alexnet(pretrained=True)  # Load in pre-trained AlexNet
imagenet_data = torchvision.datasets.imagenet.ImageNet('path/to/imagenet_root/', download=True)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=nThreads)
