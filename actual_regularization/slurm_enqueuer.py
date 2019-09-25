import os
from itertools import product, cycle, count
import random
from random_words import RandomWords
from reshape_data import download_and_convert
from socket import gethostname
import numpy as np
import collections
import json

depends = collections.defaultdict(dict)

env_name = "power_law"

rw = RandomWords()
homedir = os.path.expanduser("~")
job_directory = os.path.join(homedir, 'batch/')
os.makedirs(job_directory, exist_ok=True)
log_dir = os.path.join(homedir, 'logs/')
os.makedirs(log_dir, exist_ok=True)

if gethostname() == '***':
    hostname = '***'
elif gethostname() == '***':
    hostname = '***'
else:
    hostname = '***'

project_dir = os.path.join(homedir, 'projects/power_law_nn')
data_dir = os.path.join(project_dir, 'data')
result_dir = os.path.join(project_dir, 'result')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

architectures = [[(28 * 28, 200), (200, 200), (200, 10)],
                 [1, (1, 12), (1728, 1024), (1024, 10)]]
typesOfTraining = [False, True]
# activations = ['relu', 'tanh']
activations = ['relu']
# alphas = [0.1, 1, 5, 10]
alphas = 10**np.linspace(-3, 2, 10)
download_and_convert(data_dir)
data_dir = os.path.join(data_dir, 'mnist.npy')

queue_name = cycle(['gpu', 'gpu-large'])
queue_time = cycle(['8', '8'])
njobs = count()

for arch, act, alpha in product(architectures, activations, alphas):
    randname = rw.random_word() + str(random.randint(0, 100))
    job_file = os.path.join(job_directory, "{}.job".format(randname))
    if isinstance(arch[0], int):
        model = 'cnn'
        batch_size = int(arch[2][0] * 1.5)
    else:
        model = 'mlp'
        batch_size = int(arch[1][0] * 1.5)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name={}.job\n".format(randname))
        fh.writelines("#SBATCH --output={}/{}.out\n".format(log_dir, randname))
        fh.writelines("#SBATCH --error={}/{}.err\n".format(log_dir, randname))
        fh.writelines("#SBATCH --ntasks-per-node=8\n")
        # fh.writelines("#SBATCH --cpus-per-task=24\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --time={}:00:00\n".format(next(queue_time)))
        fh.writelines("#SBATCH --partition={}\n".format(next(queue_name)))
        fh.writelines("module load shared\n")
        fh.writelines("module load anaconda/3\n")
        fh.writelines("module load cuda100/toolkit/10.0\n")
        fh.writelines("module load cudnn/7.0.5\n")
        fh.writelines("source /***/software/Anaconda3/bin/activate {}\n".format(env_name))
        fh.writelines("cd \n")
        fh.writelines("cd {}/actual_regularization \n".format(project_dir))
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=$***.***@***.edu\n")
        fh.writelines("python ray_runner.py --dims \'{}\' --host {} --pathLoad {} --pathSave {} "
                      "--activation {} --alpha {} --batch_size {} --modelType {}\n".format(arch,
                                                                                           hostname,
                                                                                           data_dir, result_dir, act, alpha,
                                                                                           batch_size, model))
    os.system("sbatch %s" % job_file)
    njob = next(iter(njobs))
    depends[str(njob)]['modelType'] = model
    depends[str(njob)]['dims'] = arch
    depends[str(njob)]['batch_size'] = batch_size
    depends[str(njob)]['activation'] = act
    depends[str(njob)]['alpha'] = alpha
    depends[str(njob)]['jobName'] = randname
    break
for arch,act in product(architectures, activations):
    eps = 0.3
    randname = rw.random_word() + str(random.randint(0, 100))
    job_file = os.path.join(job_directory, "{}.job".format(randname))
    if isinstance(arch[0], int):
        model = 'cnn'
        batch_size = int(arch[2][0] * 1.5)
    else:
        model = 'mlp'
        batch_size = int(arch[1][0] * 1.5)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name={}.job\n".format(randname))
        fh.writelines("#SBATCH --output={}/{}.out\n".format(log_dir, randname))
        fh.writelines("#SBATCH --error={}/{}.err\n".format(log_dir, randname))
        fh.writelines("#SBATCH --ntasks-per-node=8\n")
        # fh.writelines("#SBATCH --cpus-per-task=24\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --time={}:00:00\n".format(next(queue_time)))
        fh.writelines("#SBATCH --partition={}\n".format(next(queue_name)))
        fh.writelines("module load shared\n")
        fh.writelines("module load anaconda/3\n")
        fh.writelines("module load cuda100/toolkit/10.0\n")
        fh.writelines("module load cudnn/7.0.5\n")
        fh.writelines("source /***/software/Anaconda3/bin/activate {}\n".format(env_name))
        fh.writelines("cd \n")
        fh.writelines("cd {}/actual_regularization \n".format(project_dir))
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=$***.***@***.edu\n")

        fh.writelines("python ray_runner.py --dims \'{}\' --host {} --pathLoad {} --pathSave {} --activation {} "
                      "--advTraining  "
                      "--eps {} --alpha {} --batch_size {} --modelType {}\n".format(
            arch, hostname, data_dir, result_dir, act, eps, 0.01, batch_size, model))
    # os.system("sbatch %s" % job_file)
    njob = next(iter(njobs))
    depends[str(njob)]['modelType'] = model
    depends[str(njob)]['dims'] = arch
    depends[str(njob)]['batch_size'] = batch_size
    depends[str(njob)]['activation'] = act
    depends[str(njob)]['eps'] = 0.01
    depends[str(njob)]['alpha'] = 0.3
    depends[str(njob)]['jobName'] = randname

with open("run_dependencies.json", "w") as f:
    json.dump(depends,f)
