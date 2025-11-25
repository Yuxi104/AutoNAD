import math
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module.qkv_super import qkv_super
from model.module.Linear_super import LinearSuper
from model.module.norm_super import LayerNormSuper
from model.module.mlp_super import spatial_shift1, spatial_shift2
from model.module.conv_super import Conv2dSuper, DWConvSuper, DPConv2dSuper

from model.decoder import Decoder_Super
from model.utils import DropPath, nlc2nchw
from model.utils import trunc_normal_, to_2tuple


class MLP_Super(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = LinearSuper(super_in_dim=in_features, super_out_dim=hidden_features)
        self.act = act_layer()
        self.fc2 = LinearSuper(super_in_dim=hidden_features, super_out_dim=out_features)
        self.drop = nn.Dropout(drop)

    
    def set_sample_config(self, sample_in_dim,sample_hidden_dim, sample_out_dim):
        self.sample_embed_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_hidden_dim = sample_hidden_dim

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim,
                                    sample_out_dim=self.sample_hidden_dim)
        self.fc2.set_sample_config(sample_in_dim=self.sample_hidden_dim,
                                    sample_out_dim=self.sample_out_dim)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OverlapPatchEmbed_Super(nn.Module):
    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768
                 ):
        super(OverlapPatchEmbed_Super, self).__init__()

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        # Use `self.proj` for downsampling
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.super_embed_dim = embed_dim

        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_norm = {}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def set_sample_config(self, sample_embed_dim, sample_in_dim=None):
        self.sample_embed_dim = sample_embed_dim
        if sample_in_dim is not None:
            self.sampled_weight = self.proj.weight[:sample_embed_dim,:sample_in_dim,...]
        else:
            self.sampled_weight = self.proj.weight[:sample_embed_dim,...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        self.sampled_norm['weight'] = self.norm.weight[:self.sample_embed_dim]
        self.sampled_norm['bias'] = self.norm.bias[:self.sample_embed_dim]


    def _sample_parameters(self):
        self.sampled_norm['weight'] = self.weight[:self.sample_embed_dim,]
        self.sampled_norm['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.proj.stride,padding=self.proj.padding, dilation=self.proj.dilation)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = F.layer_norm(x,(self.sample_embed_dim,),weight= self.sampled_norm['weight'], bias= self.sampled_norm['bias'])
        return x, H, W

    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel()+self.sampled_norm['weight'].numel() + self.sampled_norm['bias'].numel()
    
    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        total_flops += sequence_length * self.sample_embed_dim
        return total_flops
    

class Attention_Super(nn.Module):
    """An implementation of Spatial Reduction Attention of PVT.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 super_embed_dim,
                 num_heads,
                 kernel_size=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 qkv_bias=True,
                 qk_scale=None,
                 sr_ratio=1,
                 linear=False,
                 scale=False,
                 change_qkv=True):
        super().__init__()
        assert super_embed_dim % num_heads == 0, f"dim {super_embed_dim} should be divided by num_heads {num_heads}."

        self.dim = super_embed_dim
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.super_embed_dim = super_embed_dim
        self.linear = linear
        self.super_kernel_size = kernel_size

        self.fc_scale = scale
        self.change_qkv = change_qkv

        if change_qkv:
            self.q = qkv_super(super_embed_dim, super_embed_dim, bias=qkv_bias)
            self.kv = qkv_super(super_embed_dim, 2 * super_embed_dim, bias=qkv_bias)
        else:
            self.q = LinearSuper(super_embed_dim, super_embed_dim, bias=qkv_bias)
            self.kv = LinearSuper(super_embed_dim, 2 * super_embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearSuper(super_embed_dim, super_embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_sampled_weight=None
        self.sr_sampled_bias=None
        self.sr_ratio = sr_ratio
        self.act = nn.GELU()

        # if not linear:
        if sr_ratio > 1:
            self.sr = nn.Conv2d(super_embed_dim, super_embed_dim, kernel_size=sr_ratio,stride=sr_ratio)
            self.norm = LayerNormSuper(super_embed_dim)
        # else:
        #     self.pool = nn.AdaptiveAvgPool2d(7)
        #     self.sr = nn.Conv2d(super_embed_dim, super_embed_dim, kernel_size=1,stride=1)
        #     self.norm = LayerNormSuper(super_embed_dim)
            # self.act = nn.GELU()


            # The ret[0] of build_norm_layer is norm name.
            # self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        # self.fc = nn.Conv2d(3*self.num_heads, 9, kernel_size=1, bias=True)
        
        # For CNN
        self.cov_3_norm = LayerNormSuper(super_embed_dim)
        # self.cov_5_norm = LayerNormSuper(super_embed_dim)
        self.fc = Conv2dSuper(3*self.num_heads,self.super_kernel_size*self.super_kernel_size, kernel_size=1,stride=1)
        self.dep_conv_3 = DPConv2dSuper(9*super_embed_dim//self.num_heads, super_embed_dim, kernel_size=3, bias=True, groups=super_embed_dim//self.num_heads, padding=1)
        # self.dep_conv_5 = DPConv2dSuper(25*super_embed_dim//self.num_heads, super_embed_dim, kernel_size=5, bias=True, groups=super_embed_dim//self.num_heads, padding=2)
        
        # For MLP
        self.mlp = LinearSuper(super_embed_dim, super_embed_dim)
        self.reweight = MLP_Super(super_embed_dim, super_embed_dim//4, super_embed_dim*3)

       
    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None,sample_kernel_size=None, divisor=64):

        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        self.sample_kernel_size = sample_kernel_size
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (sample_in_embed_dim // self.sample_num_heads) ** -0.5

        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim // self.sample_num_heads) ** -0.5

        self.kv.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=2*self.sample_qk_embed_dim)
        self.q.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=self.sample_qk_embed_dim)
        self.proj.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)
        if sample_kernel_size==3:
            self.fc.set_sample_config(sample_in_dim=3 * self.sample_num_heads, sample_out_dim=9)
            self.dep_conv_3.set_sample_config(sample_in_dim=9 * sample_in_embed_dim // self.sample_num_heads,
                                            sample_out_dim=self.sample_qk_embed_dim, sample_num_heads=sample_num_heads,
                                            divisor=divisor)
            self.cov_3_norm.set_sample_config(self.sample_qk_embed_dim)

        # if sample_kernel_size==5:
        #     self.fc.set_sample_config(sample_in_dim=3 * self.sample_num_heads, sample_out_dim=25)
        #     self.dep_conv_5.set_sample_config(sample_in_dim=25 * sample_in_embed_dim // self.sample_num_heads,
        #                                     sample_out_dim=self.sample_qk_embed_dim, sample_num_heads=sample_num_heads,
        #                                     divisor=divisor)
        #     self.cov_5_norm.set_sample_config(self.sample_qk_embed_dim)

        # `0` represents MLP
        if sample_kernel_size==0:
            self.mlp.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, 
                                        sample_out_dim=self.sample_qk_embed_dim)
            
            self.reweight.set_sample_config(sample_in_dim=self.sample_qk_embed_dim,
                                            sample_hidden_dim=self.sample_qk_embed_dim//4,
                                            sample_out_dim=self.sample_qk_embed_dim*3)
            
        if sample_kernel_size==1:
            if not self.linear:
                if self.sr_ratio > 1:
                    self.sr_sampled_weight = self.sr.weight[:sample_in_embed_dim,:sample_in_embed_dim, ...]
                    self.sr_sampled_bias = self.sr.bias[:sample_in_embed_dim,...]

                    # self.sr_sampled_weight = self.sr.weight
                    # self.sr_sampled_bias = self.sr.bias

                    self.norm.set_sample_config(sample_in_embed_dim)

            else:
                self.sr_sampled_weight = self.sr.weight[:sample_in_embed_dim, ...]
                self.sr_sampled_bias = self.sr.bias[:sample_in_embed_dim, ...]
                self.norm.set_sample_config(sample_in_embed_dim)
        # self.sr.set_sample_config(sample_in_embed_dim,sample_in_embed_dim)
        # if self.relative_position:
        #     self.rel_pos_embed_k.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)
        #     self.rel_pos_embed_v.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)
    def calc_sampled_param_num(self):
        return 0
    

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.q.get_complexity(sequence_length)
        total_flops += self.kv.get_complexity(sequence_length)
        # attn
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        # x
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        total_flops += self.proj.get_complexity(sequence_length)
        # if self.relative_position:
        #     total_flops += self.max_relative_position * sequence_length * sequence_length + sequence_length * sequence_length / 2.0
        #     total_flops += self.max_relative_position * sequence_length * sequence_length + sequence_length * self.sample_qk_embed_dim / 2.0
        return total_flops
    

    def forward(self, x, H, W):
        B, N, C = x.shape
        # print("x:", x.shape)
        q = self.q(x)
        q = q.reshape(B, N, self.sample_num_heads, -1)
        _,_,_,d=q.shape
        q = q.permute(0, 2, 1, 3)
        # q=q.reshape(B, N, self.sample_num_heads, C // self.sample_num_heads).permute(0, 2, 1, 3)

        # kv_ = self.kv(x).reshape(B, N, 2*self.sample_num_heads, -1).permute(0, 2, 1, 3)
        kv_ = self.kv(x)
        kv_ = kv_.reshape(B, N, 2*self.sample_num_heads, -1).permute(0, 2, 1, 3)
        
        # MLP
        if self.sample_kernel_size==0:
            qkv = torch.cat((q, kv_), dim=1)
            f_all = qkv.reshape(B, H * W, 3 * self.sample_num_heads, -1).permute(0, 2, 1, 3)  # B, 3*nhead, H*W, C//nhead
            f_all = f_all.reshape(B, H , W, -1)
            b, h, w, c = f_all.shape
            
            x1 = spatial_shift1(f_all[:,:,:,:c//3])
            x2 = spatial_shift2(f_all[:,:,:,c//3: c//3*2])
            x3 = f_all[:,:,:,c//3 * 2:] 

            a = (x1 + x2 + x3).permute(0, 3, 1, 2).flatten(2).mean(2)
    
            a = self.reweight(a)
            
            a = a.reshape(b, c//3, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
            a = x1 * a[0] + x2 * a[1] + x3 * a[2]
             
            out_mlp = self.mlp(a)
            out_mlp = out_mlp.reshape(B, -1, C)
            

        if self.sample_kernel_size==3:
            qkv = torch.cat((q, kv_), dim=1)
           
            f_all = qkv.reshape(B, H * W, 3 * self.sample_num_heads, -1).permute(0, 2, 1, 3)  # B, 3*nhead, H*W, C//nhead
            f_conv = self.fc(f_all).permute(0, 3, 1, 2) # B, 3*nhead, H*W, C//nhead => B, K^2, H*W, C//nhead,    
                                                        # B, K^2, H*W, C//nhead
            f_conv = f_conv.reshape(B, -1, H, W)  # B, 9*C//nhead, H, W
            
            
            out_conv = self.dep_conv_3(f_conv).permute(0, 2, 3, 1).reshape(B, N, -1) # B, H, W, C
            out_conv = self.cov_3_norm(out_conv) # B, N, C
            out_conv = self.act(out_conv)


        # if self.sample_kernel_size==5:
        #     qkv = torch.cat((q, kv_), dim=1)
        #     f_all = qkv.reshape(B, H * W, 3 * self.sample_num_heads, -1).permute(0, 2, 1, 3)  # B, 3*nhead, H*W, C//nhead
        #     f_conv = self.fc(f_all).permute(0, 3, 1, 2)
        #     f_conv = f_conv.reshape(B, -1, H, W)  # B, 9*C//nhead, H, W
        #     out_conv = self.dep_conv_5(f_conv).permute(0, 2, 3, 1).reshape(B, N, -1) # B, H, W, C
        #     out_conv = self.cov_5_norm(out_conv)
        #     out_conv = self.act(out_conv)
            

        if self.sample_kernel_size==1:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = F.conv2d(x_, self.sr_sampled_weight, self.sr_sampled_bias, stride=self.sr_ratio).reshape(B, C,
                                                                                                                -1).permute(
                    0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.sample_num_heads, d).permute(2, 0, 3, 1, 4)
                # kv = self.kv(x_).reshape(B, -1, 2, self.sample_num_heads, C // self.sample_num_heads).permute(2, 0, 3, 1, 4)

            else:
                # kv = self.kv(x).reshape(B, -1, 2, self.sample_num_heads, C // self.sample_num_heads).permute(2, 0, 3, 1, 4)

                kv = self.kv(x).reshape(B, -1, 2, self.sample_num_heads, d).permute(2, 0, 3, 1, 4)

            # else:
            #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            #     x_ = self.sr(self.pool(x_), self.sr_sampled_weight, self.sr_sampled_bias, stride=self.sr_ratio).reshape(
            #         B, C, -1).permute(0, 2, 1)
            #     x_ = self.norm(x_)
            #     x_ = self.act(x_)
            #     kv = self.kv(x_).reshape(B, -1, 2, self.sample_num_heads, d).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            # qkv=torch.cat((q,k,v),dim=2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # x = self.rate1 * x + self.rate2 * out_conv
        if self.sample_kernel_size==0:
            x = out_mlp
        if self.sample_kernel_size==1:
            x = x
        if self.sample_kernel_size == 3:
            x = out_conv
        # if self.sample_kernel_size == 5:
        #     x = out_conv

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
        

class FFN_super(nn.Module):
    """An implementation of MixFFN of PVT.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Depth-wise Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
            Default: None.
        use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
            Defaults: False.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 drop_path=0.,
                 dropout_layer=None,
                 use_conv=False,
                 scale=False):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.use_conv = use_conv
        self.scale = scale

        # activate = build_activation_layer(act_cfg)
        in_channels = embed_dims
        # self.ffn_layer_norm = LayerNormSuper(self.embed_dims)
        # fc1 = Conv2d(
        #     in_channels=in_channels,
        #     out_channels=feedforward_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=True)
        self.fc1 = LinearSuper(
            super_in_dim=in_channels,
            super_out_dim=feedforward_channels)
        # if self.use_conv:
            # 3x3 depth wise conv to provide positional encode information
        self.dwconv = DWConvSuper(
                super_embed_dim=feedforward_channels)
        # fc2 = Conv2d(
        #     in_channels=feedforward_channels,
        #     out_channels=in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=True)
        self.fc2 = LinearSuper(
            super_in_dim=feedforward_channels,
            super_out_dim=in_channels)
        # self.drop_path = nn.Dropout(ffn_drop)
        self.drop = nn.Dropout(ffn_drop)

        self.act = nn.GELU()
        # layers = [fc1, activate, drop, fc2, drop]
        # if use_conv:
        #     layers.insert(1, dw_conv)
        # self.layers = Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sample_ffn_dropout = None
        # self.dropout_layer = build_dropout(
        #     dropout_layer) if dropout_layer else torch.nn.Identity()

    def set_sample_config(self, sample_embed_dim=None,
                          sample_mlp_ratio=None,sample_dropout=None):

        self.sample_embed_dim = sample_embed_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_out_dim = sample_embed_dim
        # self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        # self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim * sample_mlp_ratio)
        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim,
                                   sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.dwconv.set_sample_config(self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer,
                                   sample_out_dim=self.sample_embed_dim)
        # self.drop_path = sample_dropout
        self.sample_ffn_dropout = sample_dropout

    def calc_sampled_param_num(self):
        return 0
    
    def forward(self, x, H, W):
        out =self.fc1(x)
        out = nlc2nchw(out, H, W)
        out = self.dwconv(out)
        out = out.flatten(2).transpose(1, 2)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return out
    

class EncoderLayer_Super(nn.Module):
    """Implements one encoder layer in PVT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default: 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 kernel_size=None,
                 mlp_ratio=4.,
                 k_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 qkv_bias=True,
                 qk_scale=None,
                 pre_norm=True,
                 sr_ratio=1,
                 scale=False,
                 linear=False,
                 use_conv_ffn=True):
        super().__init__()
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.use_conv_ffn=use_conv_ffn

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.scale = scale
        self.is_identity_layer = None

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        
        self.attn = Attention_Super(
            super_embed_dim=self.super_embed_dim, kernel_size=kernel_size,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, qkv_bias=qkv_bias, qk_scale=qk_scale,
            sr_ratio=sr_ratio, linear=linear)

        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)


        self.ffn=FFN_super(embed_dims=self.super_embed_dim,
                              feedforward_channels=self.super_ffn_embed_dim_this_layer,
                              use_conv=self.use_conv_ffn)

        # self.ffn=Mlp(in_features=self.super_embed_dim,
        #              hidden_features=self.super_ffn_embed_dim_this_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop = nn.Dropout(drop)
        # self.linear = linear
        # if self.linear:
        #     self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None,sample_kernel_size=None,
                          sample_ffn_dropout=None, sample_attn_dropout=None, sample_out_dim=None, divisor=64):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        # print(sample_mlp_ratio)
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim * sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads
        self.sample_kernel_size = sample_kernel_size


        self.sample_attn_dropout = sample_attn_dropout
        self.sample_ffn_dropout = sample_ffn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_embed_dim,#self.,sample_embed_dim
                                    # sample_num_heads=self.sample_embed_dim//64,
                                    sample_num_heads=self.sample_embed_dim//divisor,
                                    sample_in_embed_dim=self.sample_embed_dim,
                                    sample_kernel_size=self.sample_kernel_size,
                                    divisor=divisor)
        self.ffn.set_sample_config(sample_embed_dim=self.sample_embed_dim,
                                   sample_mlp_ratio=self.sample_mlp_ratio,
                                   sample_dropout = self.sample_ffn_dropout
                                   )
        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(self, x, H, W,):
        if self.is_identity_layer:
            return x
        x = x + self.drop_path(self.attn(self.attn_layer_norm(x), H, W))
        x = x + self.drop_path(self.ffn(self.ffn_layer_norm(x), H, W))
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
        

class AutoNAD(nn.Module):
    def __init__(self,
                in_channels=3,
                # embed_dims=[64, 128, 320, 512],
                embed_dims=[1, 1, 1, 1],
                num_stages=4,
                depths=[2, 2, 2, 2],
                kernel_size=None,
                choice = None, 
                num_heads=[1, 2, 5, 8],
                mlp_ratio=[4, 4, 4, 4],
                sr_ratios=[8, 4, 2, 1],
                qkv_bias=True,
                qk_scale=None,
                out_indices=(0, 1, 2, 3),
                drop_rate=0.,
                attn_drop_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_path_rate=0.1,
                pre_norm=True,
                use_conv=True,
                scale=False,
                gp=False,
                linear=False,
                pretrained=None,
                num_classes=1000):
        super().__init__()
      
        self.super_embed_dims = embed_dims
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depths
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.pre_norm = pre_norm
        self.scale = scale
        self.num_stages = num_stages
        self.sr_ratios = sr_ratios
        self.out_indices = out_indices
        self.pretrained = pretrained
        self.use_conv = use_conv
        self.num_classes = num_classes
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax()
        self.gp = gp

        self.sample_fpn_dim = 256
        self.sample_choice = choice

        self.sample_embed_dim = embed_dims

        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None
        
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages):
            patch_embed_super = OverlapPatchEmbed_Super(patch_size=7 if i == 0 else 3,
                                                   stride=4 if i == 0 else 2,
                                                   in_chans=in_channels if i == 0 else self.super_embed_dims[i - 1],
                                                   embed_dim=self.super_embed_dims[i])

            blocks = nn.ModuleList([EncoderLayer_Super(
                dim=embed_dims[i], num_heads=num_heads[i], kernel_size=kernel_size[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear,use_conv_ffn=self.use_conv) for j in range(depths[i])])
            norm = LayerNormSuper(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed_super{i + 1}", patch_embed_super)
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)
      
        self.seg_head = Decoder_Super(embed_dims=self.sample_embed_dim,num_class=self.num_classes, 
                                      fpn_dim=256, choice=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        
        self.out_conv = Conv2dSuper(super_in_dim=self.num_classes,
                                super_out_dim=self.num_classes,
                                kernel_size=3, stride=1, 
                                padding=1, bias=False)

        self.apply(self._init_weights)
       

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def freeze_patch_emb(self):
        self.patch_embed_super1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['depth']
        self.sample_num_heads = config['num_heads']
        self.sample_kernel_size = config['kernel_size']
        self.sample_choice = config['conv_choice']
        self.sample_pool_scale = config['pool_scale']

        if self.sample_embed_dim[0] == 32:
            divisor = 32
        else:
            divisor = 64

        # self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dims[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i in range(self.num_stages):
            patch_embed_super = getattr(self, f"patch_embed_super{i + 1}")
            if i ==0:
                patch_embed_super.set_sample_config(self.sample_embed_dim[i])
            else:
                patch_embed_super.set_sample_config(self.sample_embed_dim[i],sample_in_dim=self.sample_embed_dim[i-1])
            setattr(self, f"patch_embed_super{i + 1}", patch_embed_super)
            blocks = getattr(self, f"block{i + 1}")
            # not exceed sample layer number
            for j, block in enumerate(blocks):
                if j < self.sample_layer_num[i]:
                    sample_dropout = 0.
                    sample_attn_dropout = 0.
                    block.set_sample_config(is_identity_layer=False,
                                            sample_embed_dim=self.sample_embed_dim[i],
                                            sample_mlp_ratio=self.sample_mlp_ratio[i][j],
                                            sample_num_heads=self.sample_num_heads[i],
                                            sample_kernel_size=self.sample_kernel_size[i][j],
                                            sample_ffn_dropout=sample_dropout,
                                            sample_out_dim=self.sample_output_dim[i],
                                            sample_attn_dropout=sample_attn_dropout,
                                            divisor=divisor)
                else:
                    block.set_sample_config(is_identity_layer=True)

            setattr(self, f"block{i + 1}", blocks)
            norm = getattr(self, f"norm{i + 1}")
            norm.set_sample_config(self.sample_embed_dim[i])
            setattr(self, f"norm{i + 1}", norm)

        self.seg_head.set_sample_config(sample_in_dim = self.sample_embed_dim, 
                                        num_classes = self.num_classes,
                                        sample_choice = self.sample_choice,
                                        sample_pool_scale=self.sample_pool_scale)

        self.out_conv.set_sample_config(sample_in_dim=self.num_classes,
                                   sample_out_dim=self.num_classes)

    # check the depth
    def should_skip_layer(self, name, config):
        if not name.startswith('block'):
            return False
        
        # get block number and layer number
        parts = name.split('.')
        # block_index starts from 1, layer index starts from 0
        try:
            block_index = int(parts[0].replace('block', ''))  
            layer_index = int(parts[1])
        except (ValueError, IndexError):
            return False

        # Check if the depth exceeds the current depth
        if layer_index >= config['depth'][block_index - 1]:
            return True

        return False
    
  
    def should_skip_operator(self, name, config):
        if not name.startswith('block'):
            return False
        
        # get block number and layer number
        parts = name.split('.')

        try:
            block_index = int(parts[0].replace('block', ''))  
            layer_index = int(parts[1])
        except (ValueError, IndexError):
            return False
        
        try:
            current_operator = config['kernel_size'][block_index - 1][layer_index]
        except (IndexError, KeyError):
            return False  
        
        skip_conditions = {
        0: {'fc', 'dep_conv_3', 'cov_3_norm', 'dep_conv_5', 'cov_5_norm', 'sr', 'norm'},
        1: {'fc', 'dep_conv_3', 'cov_3_norm', 'dep_conv_5', 'cov_5_norm', 'mlp', 'reweight','fc1', 'fc2'},
        3: {'sr', 'norm', 'dep_conv_5', 'cov_5_norm', 'mlp', 'reweight', 'fc1', 'fc2'},
        5: {'sr', 'norm', 'dep_conv_3', 'cov_3_norm', 'mlp', 'reweight', 'fc1', 'fc2'}
        }
        
        
        specific_skip_conditions = {
            'fc1': {'reweight'},  
            'fc2': {'reweight'}   
        }
        
        target_module = parts[-1]  
        parent_module = parts[-2] if len(parts) > 2 else ''   

        if current_operator in skip_conditions:
            if target_module in skip_conditions[current_operator]:
                if target_module in specific_skip_conditions:
                    if parent_module in specific_skip_conditions[target_module]:
                        return True
                else:
                    return True
            
        return False
        
        
    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        print(config)
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if self.should_skip_layer(name, config):
                    continue
                if self.should_skip_operator(name, config):
                    continue

                numels.append(module.calc_sampled_param_num())

        return sum(numels)

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops += blk.get_complexity(sequence_length + 1)
        total_flops += self.head.get_complexity(sequence_length + 1)
        return total_flops


    def forward_features(self, x):
        B = x.shape[0]
        out_features = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed_super{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            # print(x.shape)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            out_features.append(x)
           
        return out_features


    def forward(self, x):
        H, W = x.size(2), x.size(3)

        x = self.forward_features(x)
        # print("forward_features")
        # print(x[0].shape, x[1].shape, x[2].shape, x[3].shape)
        x = self.seg_head(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
    
        x = self.out_conv(x)
        return {"out": x} 
