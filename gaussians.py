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

import dataloader
import networks
from get_data import get_data
from FD_calc import FD_calc
from scatter_plotV2 import scatter_plot


"""Code for training a linear generator to translate a Gaussian in arbitrary dimensions."""

# get command line args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser('Testbed for 2D WGAN with Linear generator')
parser.add_argument('--temp_dir', type=str, required=True, help='directory for saving')
parser.add_argument('--source_params', type=str, default='0_2_1_1',
                    help='A string specifying the source dist, of the form shift_dimension_stdx_stdy. stdy will be '
                         'ignored unless dim = 2')
parser.add_argument('--target_params', type=str, default='6_2_1_1',
                    help='A string specifying the source dist, of the form shift_dimension_stdx_stdy. stdy will be '
                         'ignored unless dim = 2')
parser.add_argument('--dim', type=int, default=64, help='int determining width of critic')
parser.add_argument('--seed', type=int, default=-1, help='Set random seed for reproducibility')
parser.add_argument('--lamb', type=float, default=100., help='parameter multiplying gradient penalty')
parser.add_argument('--glr', type=float, default=1e-4, help='learning rate for generator updates')
parser.add_argument('--clr', type=float, default=1e-4, help='learning rate for critic updates')
parser.add_argument('--critters', type=int, default=5, help='number of critic iters per gen update')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--iters', type=int, default=100000,
                    help='number of generator updates')
parser.add_argument('--plus', action='store_true', help='take one sided penalty')
parser.add_argument('--ot', action='store_true', help='use minibatch optimal ray sampling')
parser.add_argument('--relu', action='store_true', help='use relu for non-linearity. If not, use tanh')
parser.add_argument('--gen_adam', action='store_true', help='use adam for generator training. If False, use SGD')
parser.add_argument('--gt_period', type=int, default=100,
                    help='how many consecutive generator iters before critic trained.')
parser.add_argument('--ct_period', type=int, default=1000, help='how long to train critic for while generator rests')
parser.add_argument('--bradford_c', type=float, default=0.,
                    help='Do ray sampling with Bradford distribution with given parameter. Does uniform if param = 0')

args = parser.parse_args()

temp_dir = args.temp_dir  # directory for temp saving


# code to get deterministic behaviour
if args.seed != -1:  # if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # If true, optimizes conv for hardware, but gives non-determ. behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# save args in config file
config_file_name = os.path.join(temp_dir, 'train_config.txt')
with open(config_file_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# begin definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
args.source = 'gaussian'
args.target = 'gaussian'
args.gen_model = 'Affine'

source_loader = getattr(dataloader, args.source)(args.source_params, args.bs)
source_gen = iter(source_loader)

target_loader = getattr(dataloader, args.target)(args.target_params, args.bs)
target_gen = iter(target_loader)

generator = getattr(networks, args.gen_model)(source_loader.latent_dim)

critic = getattr(networks, 'Discriminator')(args.dim, relu=args.relu, source_dim=target_loader.latent_dim)
print(generator)
print(critic)

print('Arguments:')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

use_cuda = torch.cuda.is_available()

if use_cuda:
    critic = critic.cuda()
    generator = generator.cuda()

beta_1 = 0.0
beta_2 = 0.9
print('Using betas of {} and {} for Adam'.format(beta_1, beta_2))

optimizerD = optim.Adam(critic.parameters(), lr=args.clr, betas=(beta_1, beta_2))
if args.gen_adam:
    optimizerG = optim.Adam(generator.parameters(), lr=args.glr, betas=(beta_1, beta_2))
else:
    optimizerG = optim.SGD(generator.parameters(), lr=args.glr)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

start_time = time.time()
g_idx = 0

if target_loader.latent_dim == 2:
    scatter_plot(generator, '00', args)

for iteration in range(args.iters):
    ############################
    # (1) Update D network
    ###########################

    for p in critic.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in generator update

    if iteration % args.gt_period == 0:
        CRITIC_ITERS = args.ct_period
    else:
        CRITIC_ITERS = args.critters

    for i in range(CRITIC_ITERS):

        real = get_data(target_gen)

        for param in critic.parameters():  # more efficient than critic.zero_grad()
            param.grad = None

        D_real = critic(real)
        D_real = D_real.mean()
        D_real.backward()

        # generate fake data
        source = get_data(source_gen)
        fake = generator(source).detach().clone()  # detach from comp graph, o.w. grads w.r.t Gen params are computed

        if args.ot:
            G0, mb_wcost = compute_emd(np.array(real.view(args.bs, -1).cpu()),
                                       np.array(fake.view(args.bs, -1).cpu()))

            otmap = torch.LongTensor(G0.dot(np.arange(G0.shape[0])))
            fake = fake[otmap, :].clone().detach()  # unscrambles fake data according to OT plan.

        D_fake = -critic(fake)
        D_fake = D_fake.mean()
        D_fake.backward()

        # compute gradient penalty
        gradient_penalty, interpolates = calc_gradient_penalty(critic, real,
                                                               fake, args.lamb,
                                                               return_samples=(args.gt_period > 0),
                                                               bradford_c=args.bradford_c)
        gradient_penalty.backward()

        D_cost = D_fake + D_real + gradient_penalty  # D_fake has negative baked in
        nopen = D_real + D_fake
        optimizerD.step()

        log.plot('dcost', D_cost.cpu().data.numpy())
        log.plot('time', time.time() - start_time)
        log.plot('no_pen', nopen.cpu().data.numpy())
        log.tick()
        if i % 100 == 99:
            log.flush(temp_dir)

    ############################
    # (2) Update G network
    ###########################
    for p in critic.parameters():
        p.requires_grad = False  # to avoid computation

    for param in generator.parameters():
        param.grad = None  # more efficient than generator.zero_grad()

    source = get_data(source_gen)
    fake = generator(source)
    G = critic(fake)

    G = G.mean()
    G.backward()
    G_cost = G
    optimizerG.step()

    # Write logs and save samples
    log.plot('gcost', G_cost.cpu().data.numpy())

    FD, trace_term, avg_eigs, min_eig, max_eig = FD_calc(generator, args)
    log.plot('FD', FD.cpu().data.numpy())
    log.plot('tr_term', trace_term.cpu().data.numpy())
    log.plot('avg_eigs', avg_eigs.cpu().data.numpy())
    log.plot('min_eig', min_eig.cpu().data.numpy())
    log.plot('max_eig', max_eig.cpu().data.numpy())

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        log.flush(temp_dir)

    log.tick()

    if iteration % (args.iters // 10) == (args.iters // 10 - 1) or iteration == 0:
        if target_loader.latent_dim == 2:
            scatter_plot(generator, g_idx, args)
        g_idx += 1


path = os.path.join(args.temp_dir, 'log.pkl')
os.rename(path, os.path.join(args.temp_dir, 'log_ot{}_bradford{}.pkl'.format(args.ot, args.bradford_c)))
