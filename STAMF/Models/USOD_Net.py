from .Transformer_mamba_depth import Transformer
from .Transformer import token_Transformer
from .Transformer_mamba_depth import ViMamba
from .DAM_module import *
from .Decoder_Dconv import Decoder


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        # Cross modality fusion
        self.MCMF1 = Mamba_fusion_enhancement_module(3)
        self.MCMF2 = Mamba_fusion_enhancement_module(64)
        self.MCMF3 = Mamba_fusion_enhancement_module(64)
        # Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.polar_backbone = T2t_Vision_Mamba(pretrained=False, args=args)

        self.mamba = ViMamba(patch_size=16, embed_dim=384, depth=12, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True)

        self.decoder = Decoder()


    def forward(self, image_Input, polar_Input):
        B, _, _, _ = image_Input.shape

        feature_map1 = self.rgb_backbone(image_Input, layer_flag=1)
        pol_layer3_vit, _, _, pol_layer1, pol_layer2, pol_layer3 = self.polar_backbone(polar_Input)

        img_cmf1 = self.MCMF1(feature_map1, pol_layer1)
        img_layer_cat1 = feature_map1 + img_cmf1
        feature_map2, rgb_fea_1_4 = self.rgb_backbone(img_layer_cat1, layer_flag=2)

        img_cmf2 = self.MCMF2(feature_map2, pol_layer2)
        img_layer_cat2 = feature_map2 + img_cmf2
        feature_map3, rgb_fea_1_8 = self.rgb_backbone(img_layer_cat2, layer_flag=3)

        img_cmf3 = self.MCMF3(feature_map3, pol_layer3)
        img_layer_cat3 = feature_map3 + img_cmf3

        img_layer3_vit = self.rgb_backbone(img_layer_cat3, image_Input, layer_flag=4)

        rgb_fea_1_16, polar_fea_1_16 = self.mamba(img_layer3_vit, pol_layer3_vit)

        rgb_fea_1_16 = rgb_fea_1_16.transpose(1, 2).reshape(B, 384, 14, 14)
        polar_fea_1_16 = polar_fea_1_16.transpose(1, 2).reshape(B, 384, 14, 14)

        outputs = self.decoder.forward(rgb_fea_1_16, polar_fea_1_16,  feature_map3, feature_map2, feature_map1)


        return outputs

