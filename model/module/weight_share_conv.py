import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]
    return sample_weight

def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]
    # sample_bias = bias
    return sample_bias


# Weight Shape (C_out, C_in // groups, H_k, W_k)
def sample_conv_weight(weight, conv_choice):
    in_channels = weight.shape[1]
    if conv_choice == 0:
        sample_weight = weight[:, :, 2:3, 2:3]
        # sample_weight = weight[:, :, 0:1, 0:1]
        padding = 0
        dilation = 1
        groups = 1
    ### 3x3 conv
    elif conv_choice == 1:
        sample_weight = weight[:, :, 1:4, 1:4]
        # sample_weight = weight[:, :, 0:3, 0:3]
        padding = 1
        dilation = 1
        groups = 1
    ### 5x5 conv
    elif conv_choice == 2:
        sample_weight = weight[:, :, 0:5, 0:5]
        padding = 2
        dilation = 1
        groups = 1
    ### 3x3 dilation conv
    elif conv_choice == 3:
        sample_weight = weight[:, :, 1:4, 1:4]
        # sample_weight = weight[:, :, 0:3, 0:3]
        padding = 2
        dilation = 2
        groups = 1
    elif conv_choice == 4:
        # depthwise conv: shape should be [in_channels, 1, k, k]
        k = 3  # 3x3
        out_ch, in_ch, h, w = weight.shape
        use_ch = min(in_channels, out_ch, in_ch)
        # Take the diagonal line
        diag_weight = weight[:use_ch, :use_ch, 1:4, 1:4]
        diag_weight = diag_weight[range(use_ch), range(use_ch), :, :].unsqueeze(1)  # [use_ch, 1, 3, 3]
        # If in_channels > use_ch, pad zero
        if in_channels > use_ch:
            pad_shape = (in_channels - use_ch, 1, k, k)
            pad_weight = weight.new_zeros(pad_shape)
            sample_weight = torch.cat([diag_weight, pad_weight], dim=0)
        else:
            sample_weight = diag_weight

        sample_weight = sample_weight.detach().clone().requires_grad_()
        padding = 1
        dilation = 1
        groups = in_channels
    
    return sample_weight, padding, dilation, groups


class SharedWeightConv2d(nn.Conv2d):
    def __init__(self, super_in_dim, super_out_dim,kernel_size, stride, padding=0, groups=None, bias=True):
        super().__init__(super_in_dim, super_out_dim, kernel_size, stride,padding=padding,
                         groups=groups if groups is not None else 1, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.super_groups = groups
        self.stride = stride
        self.padding = padding

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.sample_groups = None

        self.samples = {}
        self.profiling = False

        # Use pw conv to change the number of channels (after dw)
        self.pw_conv = nn.Conv2d(super_in_dim, super_out_dim, 1, 1, 1, bias=False)

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters(conv_choice=self.conv_choice)
        return self.samples
    
    def set_sample_config(self, sample_in_dim, sample_out_dim, 
                          sample_groups=None, sample_choice=None):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        if sample_groups is not None:
            self.sample_groups = sample_groups
        else:
            self.sample_groups = 1

        self.conv_choice = sample_choice
        self.samples, self.padding, self.dilation, self.sample_groups = self._sample_parameters(conv_choice=self.conv_choice)

    
    def _sample_parameters(self, conv_choice):
        # print(self.sample_in_dim, self.sample_out_dim)
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
      
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)

        self.samples['weight'], padding, dilation, groups = sample_conv_weight(self.samples['weight'], conv_choice)
        # print(self.samples['weight'].shape)

        return self.samples, padding, dilation, groups


    def forward(self, x):
        self.sample_parameters()
        # print(self.samples['weight'].shape)
        # print(self.sample_groups)
        # Calculate the weight shape to determine if it is dw conv. If so, add a pw convo to change the number of channels. 
        if self.samples['weight'].shape[1] == 1:
            out = F.conv2d(x, self.samples['weight'], None, padding= self.padding,stride= self.stride ,groups=self.sample_groups, dilation=self.dilation)
            pw_weight = self.pw_conv.weight[:self.sample_out_dim, :self.sample_in_dim, :, :]
            out = F.conv2d(out, pw_weight, None, stride=1, padding=0, groups=1)
        else:
            out = F.conv2d(x, self.samples['weight'], self.samples['bias'],padding= self.padding,stride= self.stride ,groups=self.sample_groups, dilation=self.dilation)
        return out
        # return F.conv2d(x, self.samples['weight'], self.samples['bias'],padding= self.padding,stride= self.stride ,groups=self.sample_groups, dilation=self.dilation)
    
    def recalculate_params(self):
        if 'weight' in self.samples.keys():
            del self.samples['weight']
            if self.samples['bias'] is not None:
                del self.samples['bias']
    
    def calc_sampled_param_num(self):
        if 'weight' in self.samples.keys():
            weight_numel = self.samples['weight'].numel()
            if self.samples['bias'] is not None:
                bias_numel = self.samples['bias'].numel()
            else:
                bias_numel = 0
            return weight_numel + bias_numel
        else:
            return 0

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops