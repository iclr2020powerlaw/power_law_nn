from advTraining import ex as advEx
from training import ex as regEx
import fire
import os
import numpy as np
import math

def server(noReals=3, cuda=1, numEpochs=371):
    architectures_mlp = [[(28 * 28, 200), (200, 200), (200, 10)]]
    pathSave_mlp = os.path.join('/home/***/projects/power_law_nn'
                            ,'trained_models/data_analysis/mlp/')
    architectures_cnn = [[1, (1, 28), (4032, 128), (128, 10)],
                     [2, (1, 32), (32, 64), (1024, 1024), (1024, 10)]]
    pathSave_cnn = os.path.join('/home/***/projects/power_law_nn',
                            'trained_models/data_analysis/cnn/')
    alphas = [0.01, 0, 0.1, 1, 10]
    batch_size_mlp = 300
    batch_size_cnn = [6048,6912]
    if numEpochs is None:
        numEpochs = 556
    for _ in range(noReals):
        for alpha in alphas[::-1]:
            for i in range(len(architectures_cnn)):
                config = {'batch_size': batch_size_cnn[i],
                          'alpha': alpha,
                          'dims' : architectures_cnn[i],
                          'numEpochs': numEpochs,
                          'modelType': 'cnn',
                          'pathLoad': '/home/***/projects/power_law_nn/data/mnist.npy',
                          'pathSave': pathSave_cnn}
                regEx.run(config_updates={**config})
                print('Double tasty!')
            config = {'batch_size': batch_size_mlp,
                      'alpha': alpha,
                      'dims': architectures_mlp[0],
                      'numEpochs': numEpochs,
                      'modelType': 'mlp',
                      'pathLoad': '/home/***/projects/power_law_nn/data/mnist.npy',
                      'pathSave': pathSave_mlp}
            regEx.run(config_updates={**config})
            print('Tasty!')
if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--noReals', help='Server where the code ran', type=int,
                           default=3)
    argparser.add_argument('--cuda', help='Server where the code ran', type=int,
                           default=1)
    argparser.add_argument('--numEpochs', help='Server where the code ran', type=int,
                           default=None)
    args, unknownargs = argparser.parse_known_args()
    args = vars(args)
    server(**args)