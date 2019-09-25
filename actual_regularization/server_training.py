from training import train
import os
from subprocess import check_output
from joblib import Parallel, delayed
import argparse
import queue
import itertools
import queue
import torch
import fire

nJobs = 1
nGPUs = torch.cuda.device_count()
assert nJobs <= nGPUs
q = queue.Queue(maxsize=nJobs)

for i in range(nJobs):
    q.put(i)


def runner(args):
    """

    :param args: dict with parameters accepted by
    :return:
    """
    gpu = q.get()
    os.system("CUDA_VISIBLE_DEVICES={}".format(gpu))
    name = None # when passing name = None the name will be generate automatically
    training.train(name, args)
    #release
    q.put(gpu)


def server(nJobs=4, batch_size=200, noReals=3,  numEpochs=10000):
    architectures = [[(28 * 28, 100), (100, 10)],
                     [(28 * 28, 500), (500, 10)],
                     [(28 * 28, 1000), (1000, 10)],
                     [(28 * 28, 100), (100, 100), (100, 10)],
                     [(28 * 28, 200), (200, 10)]]
    reps = range(noReals)
    alphas = [1e-3, 1e-2, 1e-1, 1, 0]
    lr = 1e-3
    allTheNets = itertools.product(alphas, architectures, reps)

    Parallel(n_jobs=len(nJobs))(delayed(runner)(args={"dims": arch,
                                                      "alpha": alpha,}) for alpha, arch, _ in allTheNets)


if __name__ == '__main__':
    fire.Fire(server)