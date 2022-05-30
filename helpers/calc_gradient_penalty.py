import torch
from torch import autograd

# ~~~~~~Gradient Penalty
def calc_gradient_penalty(model, real_data, fake_data, args, return_samples=False):
    use_cuda = torch.cuda.is_available()
    bs = real_data.shape[0]

    alpha = torch.rand(bs, 1)
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + (1 - alpha) * fake_data

    if use_cuda:
        interpolates = interpolates.cuda()

    interpolates.requires_grad = True
    disc_interpolates = model(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = (torch.clamp(gradients.norm(2, dim=1) - 1, min=0) ** 2).mean() * args.lamb

    if return_samples:
        return gradient_penalty, interpolates.detach().clone()
    else:
        return gradient_penalty, torch.zeros([1, 2])
