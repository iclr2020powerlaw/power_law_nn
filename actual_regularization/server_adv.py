from advTraining import ex as advEx
from training import ex as regEx
import fire
import os
import numpy as np

def server(noReals=3, cuda=1, numEpochs=371):
    # architectures_mlp = [[(28 * 28, 200), (200, 200), (200, 10)],
    #                  [(28 * 28, 10_000), (10_000, 10)]]
    # pathSave_mlp = os.path.join('/home/***/projects/power_law_nn'
    #                         ,'trained_models/data_analysis/mlp/')
    architectures_cnn = [[1, (1, 28), (4032, 128), (128, 10)],
                     [2, (1, 32), (32, 64), (1024, 1024), (1024, 10)]]
    pathSave_cnn = os.path.join('/home/***/projects/power_law_nn',
                            'trained_models/data_analysis/cnn/')

    for _ in range(noReals):
        for i in range(len(architectures_cnn)):
            if i == cuda:
                config = {'dims' : architectures_cnn[i],
                          'numEpochs': numEpochs,
                          'modelType': 'cnn',
                          'pathLoad': '/home/***/projects/power_law_nn/data/mnist.npy',
                          'pathSave': pathSave_cnn}
                advEx.run(config_updates={**config})
                # config = {'dims': architectures_mlp[i],
                #           'numEpochs': numEpochs,
                #           'modelType': 'mlp',
                #           'pathLoad': '/home/***/projects/power_law_nn/data/mnist.npy',
                #           'pathSave': pathSave_mlp}
                # advEx.run(config_updates={**config})

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--noReals', help='Server where the code ran', type=int,
                           default=3)
    argparser.add_argument('--cuda', help='Server where the code ran', type=int,
                           default=1)
    argparser.add_argument('--numEpochs', help='Server where the code ran', type=int,
                           default=371)
    args, unknownargs = argparser.parse_known_args()
    args = vars(args)
    server(**args)