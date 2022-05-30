"""This is code for training a critic with or without minibatch optimal ray selection.
Once the critic is trained, contour plots are made which record some gradient flow lines for n points. 
"""

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'networks'))
sys.path.append(os.path.join(os.getcwd(), 'helpers'))
sys.path.append(os.path.join(os.getcwd(), 'plotting'))

import argparse
import time
import log
import json
import random

import numpy as np

import torch

from torch import optim

from calc_gradient_penalty import calc_gradient_penalty
from compute_emd import compute_emd
from scatter_plot import scatter_plot
from contour_plot import contour_plot
from sigma_plot import sigma_plot
import window_finder

import dataloader
import networks
from get_data import get_data

# get command line args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser('Testbed for applying backward Euler to learned critics')
parser.add_argument('--save_dir', type=str, required=True, help='directoy for saving')

# inputs for datasets
parser.add_argument('--source', type=str, required=True, default='circle',
                    choices=['circle', 'rotated_circle', 'line', 'gaussian', 'uniform', 'sin', 'spiral',
                             'hollow_rectangle'],
                    help='Which source distribution?')
parser.add_argument('--source_params', nargs='+', help='A list specifying the source dist. Enter 0 to get syntax',
                    required=True, type=float)
parser.add_argument('--target', type=str, required=True, default='circle',
                    choices=['circle', 'rotated_circle', 'line', 'gaussian', 'uniform', 'sin', 'spiral',
                             'hollow_rectangle'],
                    help='Which target distribution?')
parser.add_argument('--target_params', nargs='+', help='A list specifying the target dist. Enter 0 to get syntax',
                    required=True, type=float)

# critic parameters
parser.add_argument('--dim', type=int, default=64, help='int determining width of critic')
parser.add_argument('--lamb', type=float, default=100., help='parameter multiplying gradient penalty')
parser.add_argument('--clr', type=float, default=1e-4, help='learning rate for critic updates')
parser.add_argument('--critters', type=int, default=5000, help='number of iters to train critic')
parser.add_argument('--ot', action='store_true', help='use minibatch optimal ray selection')
parser.add_argument('--p', type=int, default=1, help='power for ot cost matrix; only used if ot is True')
parser.add_argument('--relu', action='store_true', help='use relus in discriminator')
parser.add_argument('--bs', type=int, default=128, help='batch size')

#plotting args
parser.add_argument('--num_points', type=int, default=20,
                    help='For generating flow lines - Number of points to flow')
parser.add_argument('--num_step', type=int, default=100,
                    help='For generating flow lines - Number of steps for gradient descent iterations')
parser.add_argument('--step_size', type=float, default=0.1,
                    help='For generating flow lines - Step size for gradient descent iterations')
parser.add_argument('--nice_contours', action='store_true',
                    help='use automatic spacing for contours. If False, use spacing of 1')
parser.add_argument('--xwin', nargs='+',
                    help='A list for plotting window. Syntax: [xlow, xhigh]. If not specified'
                         'will compute automatically from data', default=[0, 0], type=float)
parser.add_argument('--ywin', nargs='+',
                    help='A list for plotting window. Syntax: [ylow, yhigh]. If not specified'
                         'will compute automatically from data', default=[0, 0], type=float)



# random seed
parser.add_argument('--seed', type=int, default=-1, help='Set random seed for reproducibility')


args = parser.parse_args()

# code to get deterministic behaviour
if args.seed != -1:  # if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # If true, optimizes conv for hardware, but gives non-determ. behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# save args in config file
config_file_name = os.path.join(args.save_dir, 'train_config.txt')
with open(config_file_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# begin definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
source_loader = getattr(dataloader, args.source)(args.source_params, args.bs)
source_gen = iter(source_loader)

target_loader = getattr(dataloader, args.target)(args.target_params, args.bs)
target_gen = iter(target_loader)

critic = getattr(networks, 'Discriminator')(args.dim, relu=args.relu)
generator = getattr(networks, 'Identity')()
print(critic)

print('Arguments:')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

use_cuda = torch.cuda.is_available()

if use_cuda:
    critic = critic.cuda()

beta_1 = 0.0
beta_2 = 0.9
print('Using betas of {} and {} for Adam'.format(beta_1, beta_2))

optimizerD = optim.Adam(critic.parameters(), lr=args.clr, betas=(beta_1, beta_2))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

start_time = time.time()

max_len = 1000000  # max number of points for histogram
interpolates_history = []

flowed_points = get_data(source_gen)[0:args.num_points,:]

# Train critic
for iteration in range(args.critters):
    ############################
    # (1) Update D network
    ###########################

    real = get_data(target_gen)

    for param in critic.parameters():  # more efficient than critic.zero_grad()
        param.grad = None

    D_real = critic(real)
    D_real = D_real.mean()
    D_real.backward()

    # generate fake data
    fake = get_data(source_gen)

    if args.ot:
        G0, mb_wcost = compute_emd(np.array(real.cpu()), np.array(fake.cpu()), args.p)

        otmap = torch.LongTensor(G0.dot(np.arange(G0.shape[0])))
        fake = fake[otmap, :].clone().detach()  # unscrambles fake data according to OT plan.

    D_fake = -critic(fake)
    D_fake = D_fake.mean()
    D_fake.backward()

    # compute gradient penalty
    gradient_penalty, interpolates = calc_gradient_penalty(critic, real, fake, args, return_samples=True)
    gradient_penalty.backward()

    if len(interpolates_history) < max_len:
        interpolates_history.append(interpolates.cpu())

    D_cost = D_fake + D_real + gradient_penalty  # D_fake has negative baked in
    nopen = D_real + D_fake
    optimizerD.step()

    log.plot('dcost', D_cost.cpu().data.numpy())
    log.plot('time', time.time() - start_time)
    log.plot('no_gpen', nopen.cpu().data.numpy())
    if args.ot:
        log.plot('mb_Wcost', mb_wcost)
    log.tick()
    if iteration % 1000 == 999 or iteration == args.critters - 1:
        log.flush(args.save_dir)


interpolates_history = torch.cat(interpolates_history, dim=0)
if args.xwin == [0, 0] or args.ywin == [0, 0]:
    _ = window_finder.window_finder(interpolates_history, args)  # finds window for plotting from data

# plot initial distributions, contour, and sigma histogram
scatter_plot(generator, '0', args)
contour_plot(critic, args,  data=flowed_points)
sigma_plot(interpolates_history, iteration, args)

path = os.path.join(args.save_dir, 'log.pkl')
os.replace(path, os.path.join(args.save_dir, 'log_ot{}_p{}.pkl'.format(args.ot, args.p)))

