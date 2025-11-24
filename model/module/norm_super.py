import torch
import torch.nn as nn
import torch.nn.functional as F

   
class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False
        self.eps=1e-6

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        # self.samples['weight'] = self.weight
        # self.samples['bias'] = self.bias

        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self._sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):

        if 'weight' in self.samples.keys():
            return self.samples['weight'].numel() + self.samples['bias'].numel()
        else:
            return 0

    def recalculate_params(self):
        if 'weight' in self.samples.keys():
            del self.samples['weight']
            if self.samples['bias'] is not None:
                del self.samples['bias']

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim


class GroupNormSuper(torch.nn.GroupNorm):
    def __init__(self, num_groups, num_channels):
        super().__init__(num_groups, num_channels)

        # the largest embed dim
        self.super_embed_dim = num_channels

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False
        self.eps=1e-5

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        # self.samples['weight'] = self.weight
        # self.samples['bias'] = self.bias

        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        # print("sample_embed_dim",sample_embed_dim)
        if sample_embed_dim // 64 == 0 and sample_embed_dim >= 64:
            self.num_groups = 64
        elif sample_embed_dim // 32 == 0:
            self.num_groups = 32
        elif sample_embed_dim // 16 == 0:
            self.num_groups = 16
        else:
            self.num_groups = 8
        self._sample_parameters()
        # print("self.num_groups",self.num_groups)

    def forward(self, x):
        self._sample_parameters()
        # return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)
        return F.group_norm(x, num_groups=self.num_groups, weight=self.samples['weight'], bias=self.samples['bias'], eps=1e-05)

    def calc_sampled_param_num(self):

        if 'weight' in self.samples.keys():
            return self.samples['weight'].numel() + self.samples['bias'].numel()
        else:
            return 0
        
    def recalculate_params(self):
        if 'weight' in self.samples.keys():
            del self.samples['weight']
            if self.samples['bias'] is not None:
                del self.samples['bias']

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim


# class BatchNormSuper(torch.nn.BatchNorm2d):
#     def __init__(self, super_embed_dim, momentum, eps):
#         super().__init__(num_features=super_embed_dim)

#         # the largest embed dim
#         self.super_embed_dim = super_embed_dim

#         # the current sampled embed dim
#         self.sample_embed_dim = None

#         self.samples = {}
#         self.profiling = False

#         self.momentum = momentum
#         self.eps = eps

#     def profile(self, mode=True):
#         self.profiling = mode

#     def sample_parameters(self, resample=False):
#         if self.profiling or resample:
#             return self._sample_parameters()
#         return self.samples

#     def _sample_parameters(self):
#         self.samples['weight'] = self.weight[:self.sample_embed_dim]
#         self.samples['bias'] = self.bias[:self.sample_embed_dim]
#         # self.samples['weight'] = self.weight
#         # self.samples['bias'] = self.bias

#         return self.samples

#     def set_sample_config(self, sample_embed_dim):
#         self.sample_embed_dim = sample_embed_dim
#         self._sample_parameters()

#     def forward(self, x):
#         self._sample_parameters()
#         # 计算输入张量的均值和方差
#         mean = x.mean(dim=(0, 2, 3), keepdim=True)
#         # print("mean.shape",mean.shape)
#         var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
#         # print("var.shape",var.shape)
        
#         # 更新运行时统计信息 (Batch Normalization在训练和推理模式下的行为不同)
#         if self.training:
#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()

#         # return self.custom_bn(x=x, scale=self.samples['weight'], shift=self.samples['bias'])
#         return F.batch_norm(input, self.running_mean, self.running_var, 
#                             weight=self.samples['weight'], bias=self.samples['bias'], 
#                             training=False, momentum=self.momentum, eps=self.eps)
#         # return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)
#         # return F.group_norm(x, num_groups=32, weight=self.samples['weight'], bias=self.samples['bias'], eps=1e-05)
    

#     def calc_sampled_param_num(self):

#         if 'weight' in self.samples.keys():
#             return self.samples['weight'].numel() + self.samples['bias'].numel()
#         else:
#             return 0

#     def get_complexity(self, sequence_length):
#         return sequence_length * self.sample_embed_dim
    
# class CustomBatchNorm2d(torch.nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-3, momentum=0.03):
#         super().__init__(num_features)


#         # the largest embed dim
#         self.super_embed_dim = num_features

#         # the current sampled embed dim
#         self.sample_embed_dim = None

#         self.samples = {}
#         self.profiling = False
    
#         # self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
        
#         # 不可训练的运行时统计信息
#         self.register_buffer('running_mean',torch.zeros(self.sample_embed_dim))
#         self.register_buffer('running_var',torch.zeros(self.sample_embed_dim))

#     def profile(self, mode=True):
#         self.profiling = mode

#     def sample_parameters(self, resample=False):
#         if self.profiling or resample:
#             return self._sample_parameters()
#         return self.samples

#     def _sample_parameters(self):
#         self.samples['weight'] = self.weight[:self.sample_embed_dim]
#         self.samples['bias'] = self.bias[:self.sample_embed_dim]
#         # self.sample_embed_dim = 
#         # self.samples['weight'] = self.weight
#         # self.samples['bias'] = self.bias

#         return self.samples

#     def set_sample_config(self, sample_embed_dim):
#         self.sample_embed_dim = sample_embed_dim
#         # self.register_buffer('running_mean',torch.zeros(sample_embed_dim))
#         # self.register_buffer('running_var',torch.zeros(sample_embed_dim))

#         self._sample_parameters()

#     def forward(self, x):
#         # 可训练参数
#         self.scale = self.samples['weight']
#         self.shift = self.samples['bias']
#         # 计算输入张量的均值和方差
#         mean = x.mean(dim=(0, 2, 3), keepdim=True)
#         # print("mean.shape",mean.shape)
#         var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
#         # print("var.shape",var.shape)
        
#         # 更新运行时统计信息 (Batch Normalization在训练和推理模式下的行为不同)
#         if self.training:
#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        
#         # 归一化输入张量
#         x_normalized = (x - mean) / torch.sqrt(var + self.eps)

#         # 应用 scale 和 shift 参数
#         scaled_x = self.scale.view(1, -1, 1, 1) * x_normalized + self.shift.view(1, -1, 1, 1)
        
#         return scaled_x
    
#     def calc_sampled_param_num(self):

#         if 'weight' in self.samples.keys():
#             return self.samples['weight'].numel() + self.samples['bias'].numel()
#         else:
#             return 0

#     def get_complexity(self, sequence_length):
#         return sequence_length * self.sample_embed_dim
