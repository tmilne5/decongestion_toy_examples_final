import numpy as np
import matplotlib.pyplot as plt
import os


def sigma_plot(samples, iteration, args):
    """
    Generates an empirical histogram of sigma using the generator
    Inputs
    - samples; list of samples from sigma
    - iteration; iteration at which they are captured
    - args; arguments to critic_trainer.py

    Outputs
    - an empirical histogram of sigma saved in args.save_dir/sigmas
    """

    range = np.array([args.xwin, args.ywin])
    samples = samples.data.numpy()
    plt.hist2d(samples[:, 0], samples[:, 1], bins=100, density=True, range=range)
    plt.colorbar()
    plt.title('Empirical Distribution for $\sigma$')

    os.makedirs(os.path.join(args.save_dir, 'sigmas'), exist_ok=True)
    plt.savefig(
        os.path.join(args.save_dir, 'sigmas', 'iter{}_MORS{}_p{}.pdf'.format(iteration, args.ot, args.p)))
    plt.close()
