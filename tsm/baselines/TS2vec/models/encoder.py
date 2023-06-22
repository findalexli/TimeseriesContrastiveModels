import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from kymatio.torch import Scattering1D
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class ScatterNet1D(nn.Module):
    def __init__(self, Q=8):
        super().__init__()
        self.Q = Q
        
    def forward(self, x):
        B, T, C = x.size(0), x.size(1), x.size(2)
        J = int(torch.floor(torch.log2(torch.tensor(T / 2))))
        scattering_f = Scattering1D(J=J, shape=T, Q=self.Q).to(x.device)
        sample_output = torch.mean(torch.log(scattering_f(x[1:2, :, 0].contiguous())[:, 1:, :] + 1e-6), dim= -1)
        scattering_output = torch.zeros((B, sample_output.size(1), C))
        for i in range(C):
            scattering_output[:, :, i] = torch.mean(torch.log(scattering_f(x[:, :, i].contiguous())[:, 1:, :] + 1e-6), dim= -1)
        return scattering_output
        


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        #assert(hidden_dims == 1)
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.wavelet_extractor = ScatterNet1D()
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        # self.wavelet_extractor.to(x.device)
        # wavelet_code = self.wavelet_extractor(x)
        wavelet_code = None
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
        
