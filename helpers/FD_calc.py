import dataloader
import torch
import sys

def FD_calc(generator, args):
    """Code for computing the square Frechet distance between
    two Gaussians. The second is assumed to be isotropic with componentwise standard deviation
    of target_std
    Inputs
    - generator; a linear generator that acts on the source Gaussian
    - args; includes the parameters of the source and target Gaussians, including dimension,
    standard deviation and bias.
    """

    source_params = dataloader.param_splitter(args.source_params)
    target_params = dataloader.param_splitter(args.target_params)
    source_std = source_params[2]
    target_std = target_params[2]
    dimension = int(target_params[1])

    matrix = generator.linear.weight.data.detach().clone()
    covariance_mat = source_std ** 2 * torch.matmul(matrix, torch.transpose(matrix, 0, 1)).detach().clone()

    eigs = torch.linalg.eigvalsh(covariance_mat)
    eigs = (eigs + torch.abs(eigs))/2  # zeros out negative eigenvalues, which may appear due to numerical precision

    eigs_prod = target_std ** 2 * eigs

    gen_bias = generator.linear.bias.data
    target_bias = target_params[0] * dimension **(-1/2) * torch.ones(dimension)
    bias_diff = torch.sum((gen_bias.cpu() - target_bias) ** 2)
    trace_term = torch.sum(eigs) + target_std ** 2 * dimension - 2 * torch.sum(eigs_prod ** (1 / 2))
        

    FD = (bias_diff + trace_term)**(1/2)  # computes Frechet distance between target and generated distribution
    if torch.isnan(FD):
        print('bias diff {}'.format(bias_diff))
        print('torch.sum(eigs) is {}'.format(torch.sum(eigs)))
        print('eigs_prod ** (1/2) is {}'.format(eigs_prod ** (1/2)))
        print('eigs_prod is {}'.format(eigs_prod))
        print('trace_term is {}'.format(trace_term))
        sys.exit()

    return FD, trace_term**(1/2), torch.mean(eigs**(1/2)), torch.min(eigs**(1/2)), torch.max(eigs**(1/2))
