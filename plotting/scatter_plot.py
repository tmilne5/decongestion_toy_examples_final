import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..', 'helpers'))
from get_data import get_data
import dataloader


def scatter_plot(generator, frame, args, source_data = None, target_data = None):
    """plots samples of real and fake distributions for 2D example
    Inputs
    - generator; generator for fake data from R^2 to R^2
    - frame; index for saving picture
    - args; various arguments.
    - source/target_data: use these if you want to plot specific data

    Outputs
    - None, saves scatter in args.temp_dir/scatter/gen{}.jpg
    - Also, if frame = 00, updates args to specify view window"""

    num_samp = 1024

    if source_data == None:
        source_loader = getattr(dataloader, args.source)(args.source_params, num_samp)
        source_gen = iter(source_loader)
        fake = generator(get_data(source_gen))
    else:
        fake = source_data

    if target_data == None:
        target_loader = getattr(dataloader, args.target)(args.target_params, num_samp)
        target_gen = iter(target_loader)
        real = get_data(target_gen)
    else:
        real = target_data

    real = real.data.cpu().numpy()
    fake = fake.data.cpu().numpy()

    plt.scatter(fake[:, 0], fake[:, 1], alpha=0.1, label=r'$\mu$')
    plt.scatter(real[:, 0], real[:, 1], alpha=0.1, label=r'$\nu$')
    plt.xlim(args.xwin[0], args.xwin[1])
    plt.ylim(args.ywin[0], args.ywin[1])
    plt.title('Scatter Plot')
    plt.legend()
    os.makedirs(os.path.join(args.save_dir, 'scatter'), exist_ok=True)
    plt.savefig(os.path.join(args.save_dir, 'scatter', 'scatter.pdf'))
    plt.close()
