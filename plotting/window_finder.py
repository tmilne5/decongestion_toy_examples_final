import torch


def window_finder(data, args):
    """
    inputs: data, which is full set of interpolates for gradient penalty sampling
    return: nothing, but modifies args with window information
    """
    mins, _ = torch.min(data, dim=0)  # x min, then y min
    mins = mins.data.cpu().numpy()
    maxs, _ = torch.max(data, dim=0)
    maxs = maxs.data.cpu().numpy()
    spreads = maxs - mins

    args.xwin = [mins[0] - 0.25 * spreads[0], maxs[0] + 0.25 * spreads[0]]
    args.ywin = [mins[1] - 0.25 * spreads[1], maxs[1] + 0.25 * spreads[1]]

    return None
