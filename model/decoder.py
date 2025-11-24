import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module.conv_super import Conv2dSuper
from model.module.norm_super import GroupNormSuper
from model.module.weight_share_conv import SharedWeightConv2d


class PyramidPooling_Super(nn.Module):

    def __init__(self, in_channels=[64, 128, 320, 512], out_channels=None, pool_scales=[1, 2, 3, 6], bias=False):
        super().__init__()

        inter_channels = in_channels[-1] //4

        self.pool_scale = pool_scales
        
        self.conv1 = SharedWeightConv2d(super_in_dim=in_channels[-1],
                                        super_out_dim=inter_channels,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        
        self.conv2 = SharedWeightConv2d(super_in_dim=in_channels[-1],
                                        super_out_dim=inter_channels,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        
        self.conv3 = SharedWeightConv2d(super_in_dim=in_channels[-1],
                                        super_out_dim=inter_channels,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        
        self.conv4 = SharedWeightConv2d(super_in_dim=in_channels[-1],
                                        super_out_dim=inter_channels,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)

        self.gn_1 = GroupNormSuper(num_groups=64, num_channels=inter_channels)
        self.gn_2 = GroupNormSuper(num_groups=64, num_channels=inter_channels)
        self.gn_3 = GroupNormSuper(num_groups=64, num_channels=inter_channels)
        self.gn_4 = GroupNormSuper(num_groups=64, num_channels=inter_channels)
        self.gn_out = GroupNormSuper(num_groups=64, num_channels=out_channels)

        self.act = nn.ReLU(inplace=True)
        # inter_channels * 4 + in_channels = 2 * in_channels
        self.out = Conv2dSuper(super_in_dim=in_channels[-1]*2,
                                super_out_dim=out_channels,
                                kernel_size=1, stride=1, 
                                padding=0, bias=bias)
    

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)
    
    def set_sample_config(self, sample_in_dim, sample_out_dim, sample_choice, sample_pool_scale):
        
        self.sample_in_dim = sample_in_dim[-1]
        self.sample_inter_dim = sample_in_dim[-1] // 4
        self.sample_out_dim = sample_out_dim
        self.sample_choice = sample_choice
       
        self.pool_scale = sample_pool_scale

        self.conv1.set_sample_config(sample_in_dim=self.sample_in_dim, sample_out_dim=self.sample_inter_dim,
                                    sample_choice=self.sample_choice[0])
        self.conv2.set_sample_config(sample_in_dim=self.sample_in_dim, sample_out_dim=self.sample_inter_dim,
                                    sample_choice=self.sample_choice[1])
        self.conv3.set_sample_config(sample_in_dim=self.sample_in_dim, sample_out_dim=self.sample_inter_dim,
                                    sample_choice=self.sample_choice[2])
        self.conv4.set_sample_config(sample_in_dim=self.sample_in_dim, sample_out_dim=self.sample_inter_dim,
                                    sample_choice=self.sample_choice[3])
        self.out.set_sample_config(sample_in_dim=self.sample_in_dim*2, sample_out_dim=self.sample_out_dim,)
        
        self.gn_1.set_sample_config(self.sample_inter_dim)
        self.gn_2.set_sample_config(self.sample_inter_dim)
        self.gn_3.set_sample_config(self.sample_inter_dim)
        self.gn_4.set_sample_config(self.sample_inter_dim)
        self.gn_out.set_sample_config(self.sample_out_dim)

    def calc_sampled_param_num(self):
        return 0
    
    def forward(self, x):
        size = x.shape[2:]
        
        f1 = self.pool(x, self.pool_scale[0])
        f1 = self.act(self.gn_1(self.conv1(f1)))
        f1 = self.upsample(f1, size)

        f2 = self.pool(x, self.pool_scale[1])
        f2 = self.act(self.gn_2(self.conv2(f2)))
        f2 = self.upsample(f2, size)

        f3 = self.pool(x, self.pool_scale[2])
        f3 = self.act(self.gn_3(self.conv3(f3)))
        f3 = self.upsample(f3, size)

        f4 = self.pool(x, self.pool_scale[3])
        f4 = self.act(self.gn_4(self.conv4(f4)))
        f4 = self.upsample(f4, size)

        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        out = self.act(self.gn_out(self.out(f)))
        # print(out.shape)
        return out
    

class FeaturePyramidNet_Super(nn.Module):

    def __init__(self, embed_dims=[64, 128, 320, 512], fpn_dim=None, bias=False):
        super().__init__()

        self.fpn_dim = fpn_dim
        self.embed_dims = embed_dims
        
        self.fpn_in_1 = SharedWeightConv2d(super_in_dim=self.embed_dims[0],
                                        super_out_dim=self.fpn_dim,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        self.fpn_in_2 = SharedWeightConv2d(super_in_dim=self.embed_dims[1],
                                        super_out_dim=self.fpn_dim,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        self.fpn_in_3 = SharedWeightConv2d(super_in_dim=self.embed_dims[2],
                                        super_out_dim=self.fpn_dim,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        
        self.fpn_out_1 = SharedWeightConv2d(super_in_dim=self.fpn_dim,
                                        super_out_dim=self.fpn_dim,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        self.fpn_out_2 = SharedWeightConv2d(super_in_dim=self.fpn_dim,
                                        super_out_dim=self.fpn_dim,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)
        self.fpn_out_3 = SharedWeightConv2d(super_in_dim=self.fpn_dim,
                                        super_out_dim=self.fpn_dim,
                                        kernel_size=5, stride=1, 
                                        padding=0, bias=bias)

        self.gn_in_1 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)
        self.gn_in_2 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)
        self.gn_in_3 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)

        self.gn_out_1 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)
        self.gn_out_2 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)
        self.gn_out_3 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)

        self.act = nn.ReLU(inplace=True) 
    
    def set_sample_config(self, sample_in_dim, sample_choice):
        self.sample_in_dim = sample_in_dim
        self.sample_choice = sample_choice
        # FPN_dim = 256

        self.fpn_in_1.set_sample_config(sample_in_dim=self.sample_in_dim[0], sample_out_dim=self.fpn_dim,
                                    sample_choice=self.sample_choice[6])
        self.fpn_in_2.set_sample_config(sample_in_dim=self.sample_in_dim[1], sample_out_dim=self.fpn_dim,
                                    sample_choice=self.sample_choice[5])
        self.fpn_in_3.set_sample_config(sample_in_dim=self.sample_in_dim[2], sample_out_dim=self.fpn_dim,
                                    sample_choice=self.sample_choice[4])

        self.fpn_out_1.set_sample_config(sample_in_dim=self.fpn_dim, sample_out_dim=self.fpn_dim,
                                    sample_choice=self.sample_choice[7])
        self.fpn_out_2.set_sample_config(sample_in_dim=self.fpn_dim, sample_out_dim=self.fpn_dim,
                                    sample_choice=self.sample_choice[8])
        self.fpn_out_3.set_sample_config(sample_in_dim=self.fpn_dim, sample_out_dim=self.fpn_dim,
                                    sample_choice=self.sample_choice[9])
        
        self.gn_in_1.set_sample_config(self.fpn_dim)
        self.gn_in_2.set_sample_config(self.fpn_dim)
        self.gn_in_3.set_sample_config(self.fpn_dim)

        self.gn_out_1.set_sample_config(self.fpn_dim)
        self.gn_out_2.set_sample_config(self.fpn_dim)
        self.gn_out_3.set_sample_config(self.fpn_dim)

    def calc_sampled_param_num(self):
        return 0
    
    def forward(self, pyramid_features):
        ## Collecting output
        fpn_out = []

        ### 1/32
        f = pyramid_features[3]
        fpn_out.append(f)

        ### 1/16
        x = self.act(self.gn_in_3(self.fpn_in_3(pyramid_features[2])))
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        out = self.act(self.gn_out_3(self.fpn_out_3(f)))
        fpn_out.append(out)

        ### 1/8
        x = self.act(self.gn_in_2(self.fpn_in_2(pyramid_features[1])))
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        out = self.act(self.gn_out_2(self.fpn_out_2(f)))
        fpn_out.append(out)
       
        ### 1/4
        x = self.act(self.gn_in_1(self.fpn_in_1(pyramid_features[0])))
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        out = self.act(self.gn_out_1(self.fpn_out_1(f)))
        fpn_out.append(out)
        
        return fpn_out
    

class Decoder_Super(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512], num_class=5, fpn_dim=256, choice=None):
        super().__init__()

        # self.embed_dims = in_channel
        self.embed_dims = embed_dims
        self.num_classes = num_class
        self.fpn_dim = fpn_dim

        self.ppm = PyramidPooling_Super(in_channels=self.embed_dims, out_channels=self.fpn_dim)
        self.fpn = FeaturePyramidNet_Super(embed_dims=self.embed_dims, fpn_dim=self.fpn_dim)

       
        self.fuse = Conv2dSuper(super_in_dim=self.fpn_dim*4,
                                super_out_dim=self.fpn_dim,
                                kernel_size=1, stride=1, 
                                padding=0, bias=True)
        self.gn1 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)

       
        self.seg_1 = Conv2dSuper(super_in_dim=self.fpn_dim,
                                super_out_dim=self.fpn_dim,
                                kernel_size=1, stride=1, 
                                padding=0, bias=True)
        self.gn2 = GroupNormSuper(num_groups=64, num_channels=self.fpn_dim)
        
        self.seg_2 = Conv2dSuper(super_in_dim=self.fpn_dim,
                                super_out_dim=self.num_classes,
                                kernel_size=1, stride=1, 
                                padding=0, bias=True)
        
        self.act = nn.ReLU(inplace=True)
        
        
    
    def calc_sampled_param_num(self):
        return 0
    
    def set_sample_config(self, sample_in_dim, num_classes, sample_choice, sample_pool_scale):

        self.sample_in_dim = sample_in_dim
        self.num_classes = num_classes
        self.sample_choice = sample_choice
        # self.sample_choice = [5,5,5,5,5,5,5,5,5,5]
        self.sample_pool_scale = sample_pool_scale
        
        self.ppm.set_sample_config(sample_in_dim=self.sample_in_dim, sample_out_dim=self.fpn_dim,
                                    sample_choice=self.sample_choice, sample_pool_scale=self.sample_pool_scale)
        self.fpn.set_sample_config(sample_in_dim=self.sample_in_dim,
                                    sample_choice=self.sample_choice)
        self.fuse.set_sample_config(sample_in_dim=self.fpn_dim*4, sample_out_dim=self.fpn_dim)
        self.seg_1.set_sample_config(sample_in_dim=self.fpn_dim, sample_out_dim=self.fpn_dim)
        self.seg_2.set_sample_config(sample_in_dim=self.fpn_dim, sample_out_dim=self.num_classes)


    def forward(self, input_features):
       
        ppm = self.ppm(input_features[3])

        # update features
        input_features[3] = ppm
        fpn = self.fpn(input_features)
        # print(fpn[0].shape, fpn[1].shape, fpn[2].shape, fpn[3].shape,)
        
        out_size = fpn[3].shape[2:]
        list_f = []
        list_f.append(fpn[3])
        list_f.append(F.interpolate(fpn[2], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn[1], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn[0], out_size, mode='bilinear', align_corners=False))
       
        x = torch.cat(list_f, dim=1)
       
        x = self.act(self.gn1(self.fuse(x)))
        x = self.act(self.gn2(self.seg_1(x)))
        x = self.seg_2(x)
       
        return x
        