import numpy as np
import matplotlib.pyplot as plt
import os


def sigma_plot(samples, iteration, args):
    """
    Generates an empirical histogram of sigma using the generator
    Inputs
    - samples; list of samples from sigma
    - iteration; generator iteration at which they are captured
    - args; arguments to wgan_gp_ot_2d.py

    Outputs
    - sigma.pdf; an empirical histogram of sigma saved in args.temp_dir
    """

    range = np.array([args.xwin, args.ywin])
    samples = samples.data.numpy()
    plt.hist2d(samples[:, 0], samples[:, 1], bins=100, density=True, range=range)
    #plt.colorbar()
    plt.title('Empirical sigma distribution')

    os.makedirs(os.path.join(args.temp_dir, 'sigmas'), exist_ok=True)
    plt.savefig(
        os.path.join(args.temp_dir, 'sigmas', 'iter{}_ot{}_bradford{}.pdf'.format(iteration, args.ot, args.bradford_c)))
    plt.close()
