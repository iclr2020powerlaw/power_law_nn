from advTraining import ex
import random
import argparse
from random_words import RandomWords
import uuid
import os

rw = RandomWords()
argparser = argparse.ArgumentParser()
argparser.add_argument('--batch_size', default=200, type=int)
argparser.add_argument('--lr', default=1e-3, type=float)
argparser.add_argument('--dims', default=[(28 * 28, 1000), (1000, 10)], nargs='+')
argparser.add_argument('--numEpochs', 100, type=int)
argparser.add_argument('--cuda', False, type=bool)
argparser.add_argument('--alpha', 0.0, type=float)
argparser.add_argument('--pathSave', '../simple_regularization/data/mnist.npy', type=str)
argparser.add_argument('--pathSave',default='trained_models/data_analysis/mlp/', type=str)
argparser.add_argument('--epochSave', default=10, type=int)
argparser.add_argument('--activation', default='relu')
argparser.add_argument('--modelType',
                           choices=['mlp', 'cnn', 'autoencoder'],
                           default='mlp',
                           help='Type of neural network.')
argparser.add_argument('--runName', default=rw.random_word() + str(random.randint(0, 100)),
                       type=str)

args = argparser.parse_args()
args = vars(args)

name = args['runName']
args.pop('runName', None)
args['pathSave'] = default=os.path.join(args['pathSave'], 'tmp'+str(uuid.uuid4())[:8])


def main():
    sacredObj = ex.run(config_updates={**args},
                   options={'--name': name})
    return sacredObj.result


if __name__ == '__main__':
    main()
