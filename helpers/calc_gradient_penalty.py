import torch
from torch import autograd
from scipy.stats import bradford
from to_polar import to_polar

# ~~~~~~Gradient Penalty
def calc_gradient_penalty(model, real_data, fake_data, args, return_samples=False):
    use_cuda = torch.cuda.is_available()
    bs = real_data.shape[0]

    if args.bradford_c > 0:
        alpha = bradford.rvs(args.bradford_c, size=(bs, 1))
        alpha = torch.Tensor(alpha)
    elif args.source_only:
        alpha = torch.zeros(bs, 1)
    elif args.target_only:
        alpha = torch.ones(bs,1)
    else:
        alpha = torch.rand(bs, 1)
    alpha = alpha.expand_as(real_data)

    alpha = alpha.cuda() if use_cuda else alpha

    if args.curved_lines:
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = to_polar(interpolates)
    elif args.to_polar:
        interpolates = alpha * to_polar(real_data) + (1-alpha) * to_polar(fake_data)
    else:
        interpolates = alpha * real_data + (1 - alpha) * fake_data

    if use_cuda:
        interpolates = interpolates.cuda()

    interpolates.requires_grad = True
    disc_interpolates = model(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    if args.plus:
        gradient_penalty = (torch.clamp(gradients.norm(2, dim=1) - 1, min=0) ** 2).mean() * args.lamb
    else:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lamb

    if return_samples:
        return gradient_penalty, interpolates.detach().clone()
    else:
        return gradient_penalty, torch.zeros([1, 2])
