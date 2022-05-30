import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from steptaker import steptaker


def contour_plot(critic, args, data=None):
    """
    Generates a contour plot of the critic
    Inputs:
    - critic; the trained neural network you want to make a contour plot for
    - args; full list of args to critic_trainer.py
    - data; if not None, do gradient descent for the data on the critic, and superimpose it on the contour
    """
    num_points = 100  # number of points to sample in each axis

    points = np.zeros((num_points, num_points, 2), dtype='float32')  # grid of points to sample critic at
    # a method for generating a grid of points
    points[:, :, 0] = np.linspace(args.xwin[0], args.xwin[1], num_points)[:, None]  # repeats x values over y index
    points[:, :, 1] = np.linspace(args.ywin[0], args.ywin[1], num_points)[None, :]  # repeats y values over x index
    points = points.reshape((-1, 2))  # removes excess dimensions
    points = torch.from_numpy(points)

    critic_vals = critic(points)  # applies discriminator to points.
    critic_vals = critic_vals.data.cpu().numpy()
    if args.nice_contours:
        levels = 75  # automatically spaces 75 contour lines for visual appeal
    else:
        levels = np.arange(np.amin(critic_vals), np.amax(critic_vals))  # there will be spacing of 1 between contours

    x = np.linspace(args.xwin[0], args.xwin[1], num_points)
    y = np.linspace(args.ywin[0], args.ywin[1], num_points)
    plt.contour(x, y, critic_vals.reshape((len(x), len(y))).transpose(),
                levels=levels)  # creates contour map of critic
    plt.colorbar()
    plt.title('Critic Contour Plot')
    os.makedirs(os.path.join(args.save_dir, 'contours'), exist_ok=True)

    # if there are points to flow
    if data is not None:
        num_points = data.size(0)
        flow = []  # to be filled with gradient descent of points
        for _ in range(args.num_step):
            flow.append(data)
            # applies one step of forward Euler discretization for gradient descent
            data = steptaker(data, critic, args.step_size, num_step=1)  # data is num_points x 2

        flow = torch.stack(flow, dim=0)  # flow is num_step x num_points x 2
        for curve in range(num_points):
            plt.plot(flow[:, curve, 0], flow[:, curve, 1], color='#ff964f')  # pastel orange from xkcd
            plt.scatter(flow[0, curve, 0], flow[0, curve, 1], color='#ff964f', s=40, zorder=100)  # initial point

        plt.xlim(args.xwin[0], args.xwin[1])
        plt.ylim(args.ywin[0], args.ywin[1])

    plt.savefig(os.path.join(args.save_dir, 'contours',
                             'critters{}_dim{}_lamb{}_MORS{}_p{}.pdf'.format(args.critters, args.dim, args.lamb,
                                                                             args.ot, args.p)), dpi=300)
    plt.close()
