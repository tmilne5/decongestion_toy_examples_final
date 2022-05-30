import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

"""Function to create plot of squared Frechet Distance, trace term from sqFD, and average
eigenvalue of covariance matrix vs iteration
    Inputs
    - temp_dir; directory where log files are located
    - dimension; dimension of the problem
    Outputs
    - saved fig in args.path/sqFD_plots/, according to dimension"""

parser = argparse.ArgumentParser('Code for plotting congestion effects with shifted Gaussians')
parser.add_argument('--path', type=str, required=True, help='directory where data is located')
parser.add_argument('--dimension', type=str, required=True, help='dimension of the problem')
args = parser.parse_args()


labellist = ['MORS', 'standard', 'bradford10']
paths = {}
FD = {}
trace_terms = {}
min_eig = {}
max_eig = {}

colors = {'MORS': 'b', 'standard': 'r', 'bradford10': 'g'}

for entry in labellist:
    paths[entry] = os.path.join(args.path, '{}.pkl'.format(entry))
    log = pd.read_pickle(paths[entry])
    FD[entry] = np.stack(tuple(log['FD'].values()))
    trace_terms[entry] = [np.stack(tuple(log['tr_term'].values())), np.stack(tuple(log['min_eig'].values())), np.stack(tuple(log['max_eig'].values()))]

for entry in labellist:
    plt.plot(FD[entry], alpha=0.5, color=colors[entry], label=entry)

plt.title('Frechet Distance vs Generator Iteration: dim {}'.format(args.dimension))
plt.xlabel('Generator Iteration')
plt.ylabel('Frechet Distance')
plt.legend()

plt.savefig(os.path.join(args.path, 'FD.pdf'))
plt.close()

for entry in labellist:
    plt.plot(trace_terms[entry][0], alpha=0.5, color=colors[entry], label=entry)

plt.title('Trace term of FD vs Generator Iteration: dim {}'.format(args.dimension))
plt.xlabel('Generator Iteration')
plt.ylabel('Trace term of FD')
plt.legend()

plt.savefig(os.path.join(args.path, 'trace_term.pdf'))
plt.close()

for entry in labellist:
    plt.plot(trace_terms[entry][1], alpha=0.5, color=colors[entry], label=entry)

plt.title('Min Eigenvalue vs Generator Iteration: dim {}'.format(args.dimension))
plt.xlabel('Generator Iteration')
plt.ylabel('Min Eigenvalue')
plt.legend()

plt.savefig(os.path.join(args.path, 'min_eig.pdf'))
plt.close()

for entry in labellist:
    plt.plot(trace_terms[entry][2], alpha=0.5, color=colors[entry], label=entry)

plt.title('Max Eigenvalue vs Generator Iteration: dim {}'.format(args.dimension))
plt.xlabel('Generator Iteration')
plt.ylabel('Max Eigenvalue')
plt.legend()

plt.savefig(os.path.join(args.path, 'max_eig.pdf'))
plt.close()