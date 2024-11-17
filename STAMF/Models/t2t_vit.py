
"""
T2T-ViT
"""
import torch
from functools import partial
from torch import Tensor
from typing import Optional
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_
import numpy as np
from .token_transformer import Token_mamba

from .token_performer import Token_performer
from .transformer_mamba_block import create_block_for_t2t_vim, get_sinusoid_encoding, Block_for_t2t_vim, Block
from timm.models import load_checkpoint
from .ResNet_raw import *
from .ResNet import *
##mamba##
from .token_transformer import create_block_for_t2t_module, Token_transformer
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
from mamba_ssm.modules.mamba_simple import Mamba
import random
from .rope import *
import math

##
from .squence_all_direction import SquenceMultiDirection, ReconstructPatchImage
##

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
##mamba##



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='mamba', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            """
            There are three kinds of Soft Split (SS)
            """
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)  

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            #self.attention3 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        feature_map1 = x
        '''
        x is input image
        '''
        #print('x before SS0:', feature_map1.size())
        x = self.soft_split0(x).transpose(1, 2)     # T1
        # print('x after SS0:', x.size())
        # x [B, 56*56, 147=7*7*3]
        # iteration1: restricturization/reconstruction
        x_1_4 = self.attention1(x)          # ention1
        B, new_HW, C = x_1_4.shape

        x = x_1_4.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_map2 = x
        #print('Feature map before SS1:', feature_map2.size())
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)   # T2

        # iteration2: restricturization/reconstruction
        x_1_8 = self.attention2(x)              # attention2
        B, new_HW, C = x_1_8.shape
        x = x_1_8.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_map3 = x
        #print('Feature map before SS2:', feature_map3.size())
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)         # T3
        # final tokens
        x = self.project(x)
        #print('Feature map after SS2:', x.size())

        return x, x_1_8, x_1_4, feature_map1, feature_map2, feature_map3

    def forward1(self, x):
        feature_map1 = x
        return feature_map1

    def forward2(self, x):
        # step0: soft split

        ###x = x.CA_SA_Enhance_1()
        x = self.soft_split0(x).transpose(1, 2)

        # x [B, 56*56, 147=7*7*3]
        # iteration1: restricturization/reconstruction
        x_1_4 = self.attention1(x)
        B, new_HW, C = x_1_4.shape

        x = x_1_4.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_map2 = x
        #print('Feature map layer1:', feature_map2.size())
        return feature_map2, x_1_4

    def forward3(self, x):
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: restricturization/reconstruction
        x_1_8 = self.attention2(x)
        B, new_HW, C = x_1_8.shape
        x = x_1_8.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_map3 = x
        #print('Feature map layer2:', feature_map3.size())
        # iteration2: soft split
        ###x_enhanced2 = x.CA_SA_Enhance_3()
        return feature_map3, x_1_8

class T2T_for_mamba_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='mamba', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            """
            There are three kinds of Soft Split (SS)
            """
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            #self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            #self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.ssm1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.ssm2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'mamba':
            print('adopt mamba encoder for tokens-to-token-visionmamba')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.squence_patch_all_dir_1 = SquenceMultiDirection(embed_dim=147)
            self.squence_patch_all_dir_2 = SquenceMultiDirection(embed_dim=576)
            self.squence_patch_reconstruct_image_1 = ReconstructPatchImage()
            self.squence_patch_reconstruct_image_2 = ReconstructPatchImage()

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.ssm1 = create_block_for_t2t_module(in_dim=in_chans*7*7, d_model=token_dim)
            self.ssm2 = create_block_for_t2t_module(in_dim=token_dim*3*3, d_model=token_dim)
            #self.attention3 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)




        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        feature_map1 = x
        '''
        x is input image
        '''
        #print('x before SS0:', feature_map1.size())
        x = self.soft_split0(x).transpose(1, 2)     # T1
        # print('x after SS0:', x.size())
        # x [B, 56*56, 147=7*7*3]
        # iteration1: restricturization/reconstruction
        x = self.squence_patch_all_dir_1(x)
        ## 8个方向的patch序列 ##
        x_left_to_right = x["left_to_right"]
        x_right_to_left = x["right_to_left"]
        x_top_to_bottom = x["top_to_bottom"]
        x_bottom_to_top = x["bottom_to_top"]
        x_top_left_to_bottom_right = x["top_left_to_bottom_right"]
        x_bottom_right_to_top_left = x["bottom_right_to_top_left"]
        x_top_right_to_bottom_left = x["top_right_to_bottom_left"]
        x_bottom_left_to_top_right = x["bottom_left_to_top_right"]
        ## 8个方向的patch序列 ##
        ##分别进入SSM##
        x_1_4_left_to_right, _ = self.ssm1(x_left_to_right)
        x_1_4_right_to_left, _ = self.ssm1(x_right_to_left)
        x_1_4_top_to_bottom, _ = self.ssm1(x_top_to_bottom)
        x_1_4_bottom_to_top, _ = self.ssm1(x_bottom_to_top)
        x_1_4_top_left_to_bottom_right, _ = self.ssm1(x_top_left_to_bottom_right)
        x_1_4_bottom_right_to_top_left, _ = self.ssm1(x_bottom_right_to_top_left)
        x_1_4_top_right_to_bottom_left, _ = self.ssm1(x_top_right_to_bottom_left)
        x_1_4_bottom_left_to_top_right, _ = self.ssm1(x_bottom_left_to_top_right)

        x_1_4_dict = {
            "left_to_right": x_1_4_left_to_right,
            "right_to_left": x_1_4_right_to_left,
            "top_to_bottom": x_1_4_top_to_bottom,
            "bottom_to_top": x_1_4_bottom_to_top,
            "top_left_to_bottom_right": x_1_4_top_left_to_bottom_right,
            "bottom_right_to_top_left": x_1_4_bottom_right_to_top_left,
            "top_right_to_bottom_left": x_1_4_top_right_to_bottom_left,
            "bottom_left_to_top_right": x_1_4_bottom_left_to_top_right,
        }
        ##分别进入SSM##
        x = self.squence_patch_reconstruct_image_1(x_1_4_dict)
        # x_1_4, _ = self.ssm1(x)          # ention1
        # B, new_HW, C = x_1_4.shape
        #
        # x = x_1_4.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_map2 = x
        feature_map2_for_14_output_squence = feature_map2.flatten(2).transpose(1,2)
        #print('Feature map before SS1:', feature_map2.size())
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)   # T2  B：8, C：576, HW：784=28×28
        x = self.squence_patch_all_dir_2(x)
        ## 8个方向的patch序列 ##
        x_left_to_right = x["left_to_right"]
        x_right_to_left = x["right_to_left"]
        x_top_to_bottom = x["top_to_bottom"]
        x_bottom_to_top = x["bottom_to_top"]
        x_top_left_to_bottom_right = x["top_left_to_bottom_right"]
        x_bottom_right_to_top_left = x["bottom_right_to_top_left"]
        x_top_right_to_bottom_left = x["top_right_to_bottom_left"]
        x_bottom_left_to_top_right = x["bottom_left_to_top_right"]
        ## 8个方向的patch序列 ##
        ##分别进入SSM##
        x_1_8_left_to_right, _ = self.ssm2(x_left_to_right)
        x_1_8_right_to_left, _ = self.ssm2(x_right_to_left)
        x_1_8_top_to_bottom, _ = self.ssm2(x_top_to_bottom)
        x_1_8_bottom_to_top, _ = self.ssm2(x_bottom_to_top)
        x_1_8_top_left_to_bottom_right, _ = self.ssm2(x_top_left_to_bottom_right)
        x_1_8_bottom_right_to_top_left, _ = self.ssm2(x_bottom_right_to_top_left)
        x_1_8_top_right_to_bottom_left, _ = self.ssm2(x_top_right_to_bottom_left)
        x_1_8_bottom_left_to_top_right, _ = self.ssm2(x_bottom_left_to_top_right)

        x_1_8_dict = {
            "left_to_right": x_1_8_left_to_right,
            "right_to_left": x_1_8_right_to_left,
            "top_to_bottom": x_1_8_top_to_bottom,
            "bottom_to_top": x_1_8_bottom_to_top,
            "top_left_to_bottom_right": x_1_8_top_left_to_bottom_right,
            "bottom_right_to_top_left": x_1_8_bottom_right_to_top_left,
            "top_right_to_bottom_left": x_1_8_top_right_to_bottom_left,
            "bottom_left_to_top_right": x_1_8_bottom_left_to_top_right,
        }

        x = self.squence_patch_reconstruct_image_2(x_1_8_dict)

        feature_map3 = x
        feature_map3_for_18_output_squence = feature_map3.flatten(2).transpose(1, 2)
        x = self.soft_split2(x).transpose(1, 2)         # T3

        x = self.project(x)

        return x, feature_map3_for_18_output_squence, feature_map2_for_14_output_squence, feature_map1, feature_map2, feature_map3

    def forward1(self, x):
        feature_map1 = x
        return feature_map1

    def forward2(self, x):

        x = self.soft_split0(x).transpose(1, 2)

        x_1_4, _ = self.ssm1(x)
        B, new_HW, C = x_1_4.shape

        x = x_1_4.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_map2 = x

        return feature_map2, x_1_4

    def forward3(self, x):
        x = self.soft_split1(x).transpose(1, 2)
        x_1_8,_ = self.ssm2(x)
        B, new_HW, C = x_1_8.shape
        x = x_1_8.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_map3 = x
        return feature_map3, x_1_8

class PatchEmbed_for_flag4(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class T2t_ViM(nn.Module):
    def __init__(self,
                 img_size=224,
                 tokens_type='mamba',
                 stride=16,
                 in_chans=3,
                 depth=12,
                 embed_dim=768,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models


        self.tokens_to_token = T2T_for_mamba_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.flag4_RGB_ResNet = ResNet50_raw()
        ##for flag4 rgb
        self.rgb_patch_embed = PatchEmbed_for_flag4(
            img_size=img_size, patch_size=16, stride=stride, in_chans=channels, embed_dim=384)
        norm_layer = None
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        flag4_rgb_num_patches = self.rgb_patch_embed.num_patches
        self.flag4_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.flag4_pos_embed = nn.Parameter(torch.zeros(1, flag4_rgb_num_patches + self.num_tokens, self.embed_dim))
        self.flag4_pos_drop = nn.Dropout(p=drop_rate)
        self.rgb_channel_change = nn.Conv2d(1024, embed_dim, kernel_size=1, stride=1,bias=None)

        ##for flag rgb

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        ##flag4_rgb_dpr
        flag4_rgb_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 24)]  # stochastic depth decay rule
        flag4_rgb_inter_dpr = [0.0] + flag4_rgb_dpr
        #self.flag4_rgb_drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        ##flag4_rgb_dpr

        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block_for_t2t_vim(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        ###layer4_rgb_ssm_layer###
        self.layer4_RGB_layers = nn.ModuleList(
            [
                create_block_for_t2t_vim(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=flag4_rgb_inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(12)
            ]
        )
        ###layer4_rgb_ssm_layer###

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init

        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                self._init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.FlagForward = 0

    def _init_weights(self,
            module,
            n_layer,
            initializer_range=0.02,  # Now only used for embedding layer.
            rescale_prenorm_residual=True,
            n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x1, x2=None, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        if self.FlagForward == 0:
            B = x1.shape[0]
            x1, x_1_8, x_1_4, image1, image2, image3 = self.tokens_to_token(x1)
            B, M, _ = x1.shape
            if self.if_cls_token:
                if self.use_middle_cls_token:#执行这个
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x1 = torch.cat((x1[:, :token_position, :], cls_token, x1[:, token_position:, :]), dim=1)
                    M = x1.shape[1]
            if self.if_abs_pos_embed:
                x1 = x1 + self.pos_embed
                x1 = self.pos_drop(x1)

            if if_random_token_rank:

                shuffle_indices = torch.randperm(M)

                if isinstance(token_position, list):
                    print("original value: ", x1[0, token_position[0], 0], x1[0, token_position[1], 0])
                else:
                    print("original value: ", x1[0, token_position, 0])
                print("original token_position: ", token_position)

                # 执行 shuffle
                x1 = x1[:, shuffle_indices, :]

                if isinstance(token_position, list):
                    new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                    token_position = new_token_position
                else:
                    token_position = torch.where(shuffle_indices == token_position)[0].item()

                if isinstance(token_position, list):
                    print("new value: ", x1[0, token_position[0], 0], x1[0, token_position[1], 0])
                else:
                    print("new value: ", x1[0, token_position, 0])
                print("new token_position: ", token_position)

            if_flip_img_sequences = False
            if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
                x1 = x1.flip([1])
                if_flip_img_sequences = True
            # mamba impl
            residual = None
            hidden_states = x1
            if not self.if_bidirectional:
                for layer in self.layers:
                    hidden_states, residual = layer(
                        hidden_states, residual, inference_params=inference_params
                    )
            else:
                print('here,model has if_bidirectional')
            if not self.fused_add_norm:
                print('model has not fused _add_norm')
            else:
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

            # return only cls token if it exists
            if self.if_cls_token:
                if self.use_double_cls_token:
                    return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
                else:
                    if self.use_middle_cls_token:
                        now = torch.cat((hidden_states[:, :token_position, :],  hidden_states[:, token_position+1:, :]), dim=1)
                        return now, x_1_8, x_1_4, image1, image2, image3
                    elif if_random_cls_token_position:
                        return hidden_states[:, token_position, :]
                    else:
                        return hidden_states[:, token_position, :]

            if self.final_pool_type == 'none':
                return hidden_states[:, -1, :]
            elif self.final_pool_type == 'mean':
                return hidden_states.mean(dim=1), x_1_8, x_1_4, image1, image2, image3

            elif self.final_pool_type == 'max':
                return hidden_states
            elif self.final_pool_type == 'all':
                return hidden_states
            else:
                raise NotImplementedError
        elif self.FlagForward == 1:
            feature_map1 = self.tokens_to_token.forward1(x1)
            return feature_map1
        elif self.FlagForward == 2:
            feature_map2, x_1_4 = self.tokens_to_token.forward2(x1)
            return feature_map2, x_1_4
        elif self.FlagForward == 3:
            feature_map3, x_1_8 = self.tokens_to_token.forward3(x1)
            return feature_map3, x_1_8
        elif self.FlagForward == 4:
            B = x2.shape[0]
            M = x2.shape[1]
            x2 = self.flag4_RGB_ResNet(x2)
            x1 = self.tokens_to_token.soft_split2(x1).transpose(1, 2)
            # final tokens
            #print(x1.size())
            x = self.tokens_to_token.project(x1)
            #print(x.size())
            ########x2########
            x2 = self.rgb_channel_change(x2)
            x2 = x2.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x2 = self.norm(x2)
            x2_cls_token = self.flag4_cls_token.expand(B, -1, -1)
            token_position = M // 2
            x2 = torch.cat((x2[:, :token_position, :], x2_cls_token, x2[:, token_position:, :]), dim=1)
            x2 = x2 + self.flag4_pos_embed
            x2 = self.flag4_pos_drop(x2)
            #######x2#########

            #
            cls_token = self.cls_token.expand(B, -1, -1)
            token_position = M // 2
            x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
            #

            x = x + self.pos_embed
            x = self.pos_drop(x)

            if_flip_img_sequences = False
            if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
                x = x.flip([1])
                if_flip_img_sequences = True

            residual_rgb = None
            hidden_states_rgb = x2
            if not self.if_bidirectional:
                for RGB_layer in self.layer4_RGB_layers:
                    hidden_states_rgb, residual_rgb = RGB_layer(
                        hidden_states_rgb, residual_rgb, inference_params=inference_params
                    )


            # mamba impl
            residual = None
            hidden_states = x
            if not self.if_bidirectional:
                for layer in self.layers:
                    hidden_states_x1, residual_x1 = layer(
                        hidden_states, residual, inference_params=inference_params
                    )
            else:
                print('here,model has if_bidirectional')
            if not self.fused_add_norm:
                print('model has not fused _add_norm')
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn

                hidden_states = hidden_states_x1 + hidden_states_rgb
                residual = residual_x1 + residual_rgb

                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            # return only cls token if it exists
            if self.if_cls_token:
                if self.use_double_cls_token:
                    return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
                else:
                    if self.use_middle_cls_token:
                        hidden_states = torch.cat(
                            (hidden_states[:, :token_position, :], hidden_states[:, token_position + 1:, :]), dim=1)
                        return hidden_states
                    elif if_random_cls_token_position:
                        return hidden_states[:, token_position, :]
                    else:
                        return hidden_states[:, token_position, :]

            if self.final_pool_type == 'none':
                return hidden_states[:, -1, :]
            elif self.final_pool_type == 'mean':
                return hidden_states.mean(dim=1)
            elif self.final_pool_type == 'max':
                return hidden_states
            elif self.final_pool_type == 'all':
                return hidden_states
            else:
                raise NotImplementedError

    def forward(self, x1, x2=None, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False,layer_flag=0):
        self.FlagForward = layer_flag
        if self.FlagForward == 0:
            x, x_1_8, x_1_4, image1, image2, image3 = self.forward_features(x1, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
            return x, x_1_8, x_1_4, image1, image2, image3
        elif self.FlagForward == 1:
            feature_map1 = self.forward_features(x1, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
            return feature_map1
        elif self.FlagForward == 2:
            feature_map2, x_1_4 = self.forward_features(x1, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
            return feature_map2, x_1_4
        elif self.FlagForward == 3:
            feature_map3, x_1_8 = self.forward_features(x1, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
            return feature_map3, x_1_8
        elif self.FlagForward == 4:
            final_vit = self.forward_features(x1, x2, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
            return final_vit
#####T2T_VIM#####

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='mamba', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.FlagForward = 0  ####这个是加上的参数

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x1, x2=None):
        if self.FlagForward == 0:
            B = x1.shape[0]
            x1, x_1_8, x_1_4, image1, image2, image3 = self.tokens_to_token(x1)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x1 = torch.cat((cls_tokens, x1), dim=1)
            x1 = x1 + self.pos_embed
            x1 = self.pos_drop(x1)

            # T2T-ViT backbone
            for blk in self.blocks:
                x1 = blk(x1)

            x1 = self.norm(x1)
            # return x[:, 0]
            return x1[:, 1:, :], x_1_8, x_1_4, image1, image2, image3
        elif self.FlagForward == 1:
            feature_map1 = self.tokens_to_token.forward1(x1)
            return feature_map1
        elif self.FlagForward == 2:
            feature_map2, x_1_4 = self.tokens_to_token.forward2(x1)
            return feature_map2, x_1_4
        elif self.FlagForward == 3:
            feature_map3, x_1_8 = self.tokens_to_token.forward3(x1)
            return feature_map3, x_1_8
        elif self.FlagForward == 4:
            B = x2.shape[0]
            x1 = self.tokens_to_token.soft_split2(x1).transpose(1, 2)
            # final tokens
            #print(x1.size())
            x = self.tokens_to_token.project(x1)
            #print(x.size())

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            # T2T-ViT backbone
            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            # return x[:, 0]
            return x[:, 1:, :]



    def forward(self, x1, x2=None, layer_flag=0):
        """
        @brief:
        """
        self.FlagForward = layer_flag
        if self.FlagForward == 0:
            x, x_1_8, x_1_4, image1, image2, image3 = self.forward_features(x1)
            # x = self.head(x)
            return x, x_1_8, x_1_4, image1, image2, image3
        elif self.FlagForward == 1:
            feature_map1 = self.forward_features(x1)
            return feature_map1
        elif self.FlagForward == 2:
            feature_map2, x_1_4 = self.forward_features(x1)
            return feature_map2, x_1_4
        elif self.FlagForward == 3:
            feature_map3, x_1_8 = self.forward_features(x1)
            return feature_map3, x_1_8
        elif self.FlagForward == 4:
            final_vit = self.forward_features(x1, x2)
            return final_vit


@register_model
def T2t_vit_t_14(pretrained=True, **kwargs):  # adopt transformers for tokens to token

    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    args = kwargs['args']
    if pretrained:
        load_checkpoint(model, args.pretrained_model, use_ema=True)
        print('Model loaded from {}'.format(args.pretrained_model))
    return model

@register_model
def T2t_Vision_Mamba(pretrained=False, **kwargs):
    model = T2t_ViM(
        tokens_type='mamba', embed_dim=384, depth=14, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def T2t_vit_t_14_d(pretrained=True, **kwargs):  # adopt transformers for tokens to token

    model = T2T_ViT(tokens_type='convolution', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    args = kwargs['args']
    if pretrained:
        load_checkpoint(model, args.pretrained_model, use_ema=True)
        print('Model loaded from {}'.format(args.pretrained_model))
    return model

@register_model
def T2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3.)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3.)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_7(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_10(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_12(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def T2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_19(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_24(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


# rexnext and wide structure
@register_model
def T2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def T2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
