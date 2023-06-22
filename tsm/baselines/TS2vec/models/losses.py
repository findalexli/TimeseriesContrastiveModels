import torch
from torch import nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, beta=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if beta != 0:
            loss += beta * wavelet_scatter_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # (B x 2T x C) * (B x 2T x 2T)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def wavelet_scatter_contrastive_loss(w1, w2):
    loss = torch.tensor(0., device=w1.device)
    d = 0
    while w1.size(1) > 4 and w2.size(1) > 4: # TODO: INvestigate why they are not equal
        print(w1.size(), w2.size())
        loss += instance_contrastive_loss(w1, w1)
        d += 1
        w1 = F.max_pool1d(w1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        w2 = F.max_pool1d(w2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    return loss / d

def _single_channel_scatter(matrix, scattering):
    # Apply scattering transform to each channel in z1 and z2
    return torch.mean(torch.log(scattering(matrix)[:, 1:, :] + 1e-6), dim= -1)
