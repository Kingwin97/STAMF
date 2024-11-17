import torch
import torch.nn as nn
import numpy as np


class Mamba_fusion_enhancement_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, token_dim=64):
        super().__init__()
        # self.ssm_layer_first, _ = mamba_block_for_fusion(d_model=token_dim)
        # self.ssm_layer_last, _  = mamba_block_for_fusion(d_model=token_dim)
        self.norm = nn.LayerNorm(token_dim, token_dim)

    def forward(self, x, y):

        x = x.flatten(2).transpose(1, 2)  #B C H W -> B HW(N) C: RGB
        y = y.flatten(2).transpose(1, 2)  #B C H W -> B HW(N) C: polarization
        fusion_first = x + y
        res = fusion_first  #留个残差
        B, HW, C = fusion_first.shape
        fusion_last = torch.flip(fusion_first, [1])

        # fusion_first = self.ssm_layer_first(fusion_first)
        # fusion_last = self.ssm_layer_last(fusion_last)

        fusion_last_jiaozheng = torch.flip(fusion_last, [1])
        fusion_token = fusion_last_jiaozheng + fusion_first
        fusion_token = fusion_token + res
        #fusion_token = self.norm(fusion_token)

        fusion_image = fusion_token.transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        return  fusion_image

input_tensor_x = torch.tensor([[[[1,2,3], [4,5,6], [7,8,9]]]])
input_tensor_y = torch.tensor([[[[10,11,12], [13,14,15], [16,17,18]]]])
ceshi = Mamba_fusion_enhancement_module(1)
output = ceshi(input_tensor_x, input_tensor_y)