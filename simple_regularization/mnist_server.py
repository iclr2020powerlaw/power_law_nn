import train_networks
import numpy.random as npr
from joblib import Parallel, delayed
import fire


def server(nJobs=4, noReals=50, cuda=False):
    seeds = [int(i) for i in range(noReals)]  # seeds for RNGs
    hidden_units = [100, 300, 500, 1000]

    for d_hidden in hidden_units:
        Parallel(n_jobs=nJobs)(delayed(train_networks.training)(batch_size=256, lr=1e-3, d_hidden=d_hidden, seed=seed,
                                                                   cuda=cuda) for seed in seeds)

    # for d_hidden in hidden_units:
    #     for seed in seeds:
    #         train_networks.training(batch_size=256, lr=1e-3, d_hidden=d_hidden, seed=seed, cuda=cuda)


if __name__ == '__main__':
    fire.Fire(server())
    print("something")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("something")