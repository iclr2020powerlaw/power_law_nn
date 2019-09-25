import ray
import ray.tune as tune
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.skopt import SkOptSearch
from skopt.space import Real
from skopt import Optimizer
from multiprocessing import cpu_count
from torch.cuda import device_count
import argparse
import uuid
import os
from sacred.observers import MongoObserver


nCPU = cpu_count()
nGPU = device_count()
load = 1  # how large a fraction of the GPU memory does the model take up 0 <= load <=  1

argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--modelType',
                       choices=['mlp', 'cnn'],
                       default=argparse.SUPPRESS,
                       help='Type of neural network.')
argparser.add_argument('--activation',
                       choices=['relu', 'tanh'],
                       help='Nonlinearity.',default=argparse.SUPPRESS)
argparser.add_argument('--advTraining', action='store_true')
argparser.add_argument('--pathSave', type=str, required=True)
argparser.add_argument('--pathLoad', type=str, required=True)
argparser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS)
argparser.add_argument('--alpha', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--gradSteps', type=int, default=argparse.SUPPRESS)
argparser.add_argument('--noRestarts', type=int, default=argparse.SUPPRESS)
argparser.add_argument('--numEpochs', default=100, type=int)
argparser.add_argument('--eps', type=float, default=argparse.SUPPRESS)
argparser.add_argument('--dims', type=str, default=argparse.SUPPRESS)
argparser.add_argument('--smoke-test', action="store_true", default=False)
argparser.add_argument('--host', choices=['***', '***', '***'], help='Server where the code ran',
                       default='***')
args, unknownargs = argparser.parse_known_args()

args = vars(args)
try:
    args['dims'] = eval(args['dims'])
except:
    pass

TT = args['advTraining']
smoke_test = args['smoke_test']
host = args['host']
[args.pop(k) for k in ['advTraining','smoke_test','host']]
args['pathSave'] = os.path.join(args['pathSave'], 'tmp' + str(uuid.uuid4())[:8])

def train(config, reporter):
    import time, random
    time.sleep(random.uniform(0.0, 10.0))
    if TT:
        from advTraining import ex
    else:
        from training import ex
    if smoke_test:
        config = {'numEpochs': 1,**config}

    mongoDBobserver = MongoObserver.create(
        url='mongodb://powerLawNN:***@***.***.com/admin?authMechanism=SCRAM-SHA-1',
        db_name='powerLawHypers')
    ex.observers.append(mongoDBobserver)
    ex.run(config_updates={**args, **config})
    result = ex.current_run.result
    reporter(result=result, done=True)


if __name__ == '__main__':
    import tweepy
    consumer_key = "***"
    consumer_secret = "***"
    access_token = "***"
    access_token_secret = "***"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    ncalls = 20
    if TT:
        parameter_names = ['lr']
        space = [Real(10 ** -7, 10 ** -3, "log-uniform", name='lr')]
    else:
        parameter_names = ['lr']
        space = [Real(10 ** -7, 10 ** -3, "log-uniform", name='lr')]

    ray.init(num_cpus=nCPU, num_gpus=nGPU)

    optimizer = Optimizer(
        dimensions=space,
        random_state=1,
        base_estimator='gp'
    )
    algo = SkOptSearch(
        optimizer,
        parameter_names=parameter_names,
        max_concurrent=4,
        metric="result",
        mode="max")

    scheduler = FIFOScheduler()
    tune.register_trainable("train_func", train)
    import time, random
    time.sleep(random.uniform(0.0, 10.0))
    tune.run_experiments({
        'my_experiment': {
            'run': 'train_func',
            'resources_per_trial': {"cpu": int(nCPU * load // nGPU), "gpu": load},
            'num_samples': ncalls,
        }
    }, search_alg=algo, scheduler=scheduler)
    api.update_status(status="Optimized hypers for a power_law_nn {} with {} layer dims on {}.".format(
        args['modelType'], str(args['dims']),host))
