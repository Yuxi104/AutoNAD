import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv2dSuper(nn.Conv2d):
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

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim,sample_groups=None):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        if sample_groups is not None:
            self.sample_groups = sample_groups
        else:
            self.sample_groups = 1
        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        # self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        # print(self.samples)
        return F.conv2d(x, self.samples['weight'], self.samples['bias'],padding= self.padding,stride= self.stride ,groups=self.sample_groups)
    
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
        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops

def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]
    # sample_weight=weight
    return sample_weight

def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]
    # sample_bias = bias
    return sample_bias

class DPConv2dSuper(nn.Conv2d):
    def __init__(self, super_in_dim, super_out_dim,kernel_size,bias,groups,padding):
        super().__init__(super_in_dim, super_out_dim,
                         kernel_size,bias=bias,groups=groups,
                         padding=padding)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim,sample_num_heads, divisor):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_num_heads = sample_num_heads
        self.divisor = divisor
        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.conv2d(x, self.samples['weight'], self.samples['bias'],
                        # kernel_size=self.kernel_size,
                        padding=self.padding,
                        groups=self.divisor)
                        # groups=64)
                        # groups=self.sample_out_dim//self.sample_num_heads)

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
        
    def recalculate_params(self):
        if 'weight' in self.samples.keys():
            del self.samples['weight']
            if self.samples['bias'] is not None:
                del self.samples['bias']

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops


class DWConvSuper(nn.Module):
    def __init__(self, super_embed_dim):
        super(DWConvSuper, self).__init__()
        self.super_embed_dim = super_embed_dim
        self.super_out_dim = super_embed_dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False
        self.sampled_weight = None
        self.sampled_bias = None
        self.dwconv = nn.Conv2d(super_embed_dim, super_embed_dim, 3, 1, 1, bias=True, groups=super_embed_dim)

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.dwconv.weight[:sample_embed_dim,:sample_embed_dim, ...]
        self.sampled_bias = self.dwconv.bias[:self.sample_embed_dim, ...]
        # self.sampled_weight=self.dwconv.weight
        # self.sampled_bias = self.dwconv.bias

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W)
        # x = self.dwconv(x,self.sampled_weight, self.sampled_bias)
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=1, padding=1,
                         groups=self.sample_embed_dim)
        # x = self.dwconv(x)
        # x = x.flatten(2).transpose(1, 2)

        return x

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

    def recalculate_params(self):
        if 'weight' in self.samples.keys():
            del self.samples['weight']
            if self.samples['bias'] is not None:
                del self.samples['bias']

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


## 膨胀卷积
class DLConvSuper(nn.Conv2d):
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

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim,sample_groups=None):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        if sample_groups is not None:
            self.sample_groups = sample_groups
        else:
            self.sample_groups = 1
        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        # self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        # print(self.samples)
        return F.conv2d(x, self.samples['weight'], self.samples['bias'],padding= self.padding,stride= self.stride ,groups=self.sample_groups, dilation=2)

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
        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops