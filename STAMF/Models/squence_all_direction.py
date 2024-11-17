import torch
import torch.nn as nn
import numpy as np

class SquenceMultiDirection(nn.Module):
    """
    2D Image to Patch Embedding with multi-directional flattening.
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        #img_size = (img_size, img_size)
        #patch_size = (patch_size, patch_size)
        #self.img_size = img_size
        #self.patch_size = patch_size
        #self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        #self.num_patches = self.grid_size[0] * self.grid_size[1]

        # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        #x[B, 56 * 56, 147 = 7 * 7 * 3]
        B, HW, C = x.shape
        # Project and flatten the patches
        patch_embeds = x.transpose(1, 2)  #x[B, 147 = 7 * 7 * 3, 56 * 56]
        x_image = x
        x_image = x_image.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        grid_B, grid_C, grid_H, grid_W = x_image.shape

        # 1. Left to right
        left_to_right_1 = patch_embeds.transpose(1, 2)

        # 2. Right to left
        right_to_left_2 = torch.flip(patch_embeds, [2]).transpose(1, 2)

        # 3. Top to bottom
        top_to_bottom_3 = patch_embeds.view(B, -1, grid_W).transpose(1, 2).reshape(B, HW, -1)

        # 4. Bottom to top
        bottom_to_top_4 = torch.flip(top_to_bottom_3, [1])

        # 5. Top-left to bottom-right
        indices = []
        for s in range(grid_H + grid_W - 1):
            for y in range(max(0, s - grid_W + 1), min(grid_H, s + 1)):
                x = s - y
                indices.append(y * grid_W + x)

        # 使用索引重新排序张量
        top_left_to_bottom_right_5 = patch_embeds[:, :, indices].transpose(1, 2)

        # 6. Bottom-right to top-left
        indices6 = []
        for s in range(grid_H + grid_W - 1):
            for x in range(min(grid_W - 1, s), max(-1, s - grid_H), -1):
                y = s - x
                if y < grid_H and x < grid_W:
                    indices6.append(y * grid_W + x)
        # 逆序索引列表以确保正确的顺序
        indices6.reverse()
        bottom_right_to_top_left_6 = patch_embeds[:, :, indices6].transpose(1, 2)

        # 7. top_right_to_bottom_left
        indices7 = []
        for s in range(grid_W + grid_H - 1):
            for y in range(max(0, s - grid_W + 1), min(grid_H, s + 1)):
                x = s - y
                indices7.append(y * grid_W + (grid_W - 1 - x))  # Adjusting x to start from the right

        # 使用索引重新排序张量
        top_right_to_bottom_left_7 = patch_embeds[:, :, indices7].transpose(1, 2)

        # 8. bottom_left_to_top_right
        indices8 = []
        # 从每一行的左下角开始到右上角
        for s in range(grid_W + grid_H - 1):
            for y in range(max(0, s - grid_W + 1), min(grid_H, s + 1)):
                x = s - y
                indices8.append(y * grid_W + (grid_W - 1 - x))  # 调整 x 从右向左递减
        # 使用索引重新排序张量
        bottom_left_to_top_right_reverse = patch_embeds[:, :, indices8]
        bottom_left_to_top_right_8 = bottom_left_to_top_right_reverse.flip(dims=(2,)).transpose(1, 2)

        # Apply normalization
        outputs = {
            "left_to_right": self.norm(left_to_right_1),
            "right_to_left": self.norm(right_to_left_2),
            "top_to_bottom": self.norm(top_to_bottom_3),
            "bottom_to_top": self.norm(bottom_to_top_4),
            "top_left_to_bottom_right": self.norm(top_left_to_bottom_right_5),
            "bottom_right_to_top_left": self.norm(bottom_right_to_top_left_6),
            "top_right_to_bottom_left": self.norm(top_right_to_bottom_left_7),
            "bottom_left_to_top_right": self.norm(bottom_left_to_top_right_8),
        }

        return outputs

import torch

class ReconstructPatchImage(nn.Module):
    """
    Reconstruct the image from multi-directional patch embeddings.
    """
    def __init__(self, img_size=224, patch_size=16):
        super().__init__()
        #img_size = (img_size, img_size)
        #patch_size = (patch_size, patch_size)
        #self.img_size = img_size
        #self.patch_size = patch_size
        #self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        #self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, outputs):
        # Reverse the normalization
        left_to_right = outputs["left_to_right"]
        right_to_left = outputs["right_to_left"]
        top_to_bottom = outputs["top_to_bottom"]
        bottom_to_top = outputs["bottom_to_top"]
        top_left_to_bottom_right = outputs["top_left_to_bottom_right"]
        bottom_right_to_top_left = outputs["bottom_right_to_top_left"]
        top_right_to_bottom_left = outputs["top_right_to_bottom_left"]
        bottom_left_to_top_right = outputs["bottom_left_to_top_right"]
        B, HW, C = left_to_right.shape
        x_image = left_to_right.transpose(1,2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        grid_B, grid_C, grid_H, grid_W = x_image.shape


        # Reverse the transformations
        left_to_right = left_to_right.transpose(1, 2).view(-1, left_to_right.size(2), grid_H, grid_W)
#2
        right_to_left_2 = right_to_left.transpose(1, 2)
        right_to_left = torch.flip(right_to_left_2, [2])
        right_to_left = right_to_left.view(-1, right_to_left.size(1), grid_H, grid_W)
#3
        top_to_bottom = top_to_bottom.view(B, -1, grid_W).transpose(1, 2).reshape(B, HW, -1).transpose(1, 2)
        top_to_bottom = top_to_bottom.view(-1, top_to_bottom.size(1), grid_H, grid_W)
#4
        bottom_to_top = torch.flip(bottom_to_top, [1])
        bottom_to_top = bottom_to_top.view(B, -1, grid_W).transpose(1, 2).reshape(B, HW, -1).transpose(1, 2)
        bottom_to_top = bottom_to_top.view(-1, top_to_bottom.size(1), grid_H, grid_W)
#5
        indices = []
        for s in range(grid_H + grid_W - 1):
            for y in range(max(0, s - grid_W + 1), min(grid_H, s + 1)):
                x = s - y
                indices.append(y * grid_W + x)
        # 使用索引重新排序张量
        top_left_to_bottom_right = top_left_to_bottom_right.transpose(1,2)
        inverse_indices5 = torch.argsort(torch.tensor(indices))
        # # 使用逆序索引对top_left_to_bottom_right_5进行排序
        patch_embeds_inverse = top_left_to_bottom_right[:, :, inverse_indices5]
        # # 对排序后的张量执行逆转置操作
        top_left_to_bottom_right = patch_embeds_inverse.view(-1, patch_embeds_inverse.size(1),
                                                                  grid_H, grid_W)
#6
        indices6 = []
        for s in range(grid_H + grid_W - 1):
            for x in range(min(grid_W - 1, s), max(-1, s - grid_H), -1):
                y = s - x
                if y < grid_H and x < grid_W:
                    indices6.append(y * grid_W + x)
        # 逆序索引列表以确保正确的顺序
        indices6.reverse()
        bottom_right_to_top_left = bottom_right_to_top_left.transpose(1,2)
        # # 获取indices6的逆序索引
        inverse_indices6 = torch.argsort(torch.tensor(indices6))
        # # 使用逆序索引对bottom_right_to_top_left_6进行排序
        patch_embeds_inverse = bottom_right_to_top_left[:, :, inverse_indices6]
        bottom_right_to_top_left = patch_embeds_inverse.view(-1, patch_embeds_inverse.size(1),
                                                                 grid_H, grid_W)
        #####7
        indices7 = []
        for s in range(grid_W + grid_H - 1):
            for y in range(max(0, s - grid_W + 1), min(grid_H, s + 1)):
                x = s - y
                indices7.append(y * grid_W + (grid_W - 1 - x))  # Adjusting x to start from the right
        # 使用索引重新排序张量
        inverse_indices7 = torch.argsort(torch.tensor(indices7))
        # 使用逆序索引对top_right_to_bottom_left_7进行排序
        top_right_to_bottom_left = top_right_to_bottom_left.transpose(1,2)
        top_right_to_bottom_left = top_right_to_bottom_left[:, :, inverse_indices7]
        top_right_to_bottom_left = top_right_to_bottom_left.view(-1, top_right_to_bottom_left.size(1),
                                                                 grid_H, grid_W)
#####8
        indices8 = []
        for s in range(grid_W + grid_H - 1):
            for y in range(max(0, s - grid_W + 1), min(grid_H, s + 1)):
                x = s - y
                indices8.append(y * grid_W + (grid_W - 1 - x))  # 调整 x 从右向左递减
        # 使用索引重新排序张量
        bottom_left_to_top_right = bottom_left_to_top_right.transpose(1,2)
        # 获取indices8的逆序索引
        inverse_indices8 = torch.argsort(torch.tensor(indices8))
        patch_embeds_inverse = bottom_left_to_top_right[:, :, inverse_indices8]
        bottom_left_to_top_right = patch_embeds_inverse
        bottom_left_to_top_right = torch.flip(bottom_left_to_top_right, dims=[-1])
        bottom_left_to_top_right = bottom_left_to_top_right.view(-1, bottom_left_to_top_right.size(1),
                                                                 grid_H, grid_W)

        # Combine the patches into the original image
        # reconstructed_image = torch.cat([
        #     left_to_right, right_to_left,
        #     top_to_bottom, bottom_to_top,
        #     top_left_to_bottom_right, bottom_right_to_top_left,
        #     top_right_to_bottom_left, bottom_left_to_top_right
        # ], dim=0)
        reconstructed_image = left_to_right + right_to_left + top_to_bottom + bottom_to_top + top_left_to_bottom_right + bottom_right_to_top_left + top_right_to_bottom_left + bottom_left_to_top_right


        # Reshape the image
        #reconstructed_image = reconstructed_image.view(8, -1, self.patch_size[0], self.patch_size[1])

        return reconstructed_image

#####################
#input_tensor = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32)
# input_tensor = torch.rand(8, 3, 224, 224)
# #patch_embedder = PatchEmbedMultiDirection(img_size=3, patch_size=1, in_c=1, embed_dim=1)
# patch_embedder = PatchEmbedMultiDirection()
# x = patch_embedder(input_tensor)
# recon = ReconstructImage()
# y = recon(x)
#
# tensor = torch.randn(8, 3, 224, 224)
# patch_embedder = PatchEmbedMultiDirection()
# x = patch_embedder(tensor)





#######################################################
import torch
import torch.nn as nn

class PatchReconstructor(nn.Module):
    """
    Reconstructs the image from multiple flattened sequences.
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

    def reconstruct(self, outputs):
        """
        Reconstructs the image from multiple sequences.
        """
        # Create empty tensor to store reconstructed patches
        reconstructed = torch.zeros(
            (self.grid_size[0], self.grid_size[1], self.embed_dim)
        )

        # Place the flattened sequences back into the grid
        # Following the original order
        reconstructed[0, :] = outputs["left_to_right"].view(self.grid_size[1], -1)
        reconstructed[:, -1] = outputs["right_to_left"].view(-1, self.grid_size[1])[:, ::-1]

        # Arrange top-to-bottom
        for i in range(self.grid_size[1]):
            reconstructed[:, i] = outputs["top_to_bottom"][i]

        # Arrange bottom-to-top
        for i in range(self.grid_size[1]):
            reconstructed[::-1, i] = outputs["bottom_to_top"][i]

        # Arrange top-left to bottom-right
        for i in range(self.grid_size[0]):
            indices = [j for j in range(i, self.grid_size[0])]
            reconstructed[indices, indices] = outputs["top_left_to_bottom_right"]

        # Arrange bottom-right to top-left
        for i in range(self.grid_size[0]):
            indices = [j for j in range(self.grid_size[0] - i - 1)]
            reconstructed[indices, indices[::-1]] = outputs["bottom_right_to_top_left"]

        # Arrange bottom-left to top-right
        for i in range(self.grid_size[0]):
            indices = [j for j in range(self.grid_size[0] - i - 1)]
            reconstructed[::-1, indices] = outputs["bottom_left_to_top_right"]

        # Arrange top-right to bottom-left
        for i in range(self.grid_size[0]):
            indices = [j for j in range(i, self.grid_size[0])]
            reconstructed[indices, indices[::-1]] = outputs["top_right_to_bottom_left"]

        return reconstructed




# import torch
# import torch.nn as nn
#
# # PatchEmbedMultiDirection class with multi-directional flattening
# class PatchEmbedMultiDirection(nn.Module):
#     """
#     2D Image to Patch Embedding with multi-directional flattening.
#     """
#     def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
#         super().__init__()
#         self.img_size = (img_size, img_size)
#         self.patch_size = (patch_size, patch_size)
#         self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#
#         # Project and flatten the patches
#         patch_embeds = self.proj(x).flatten(2)
#
#         # Generate different flattening sequences
#         outputs = {}
#
#         # Left to right
#         outputs['left_to_right'] = patch_embeds.transpose(1, 2)
#
#         # Right to left
#         outputs['right_to_left'] = patch_embeds[:, :, ::-1].transpose(1, 2)
#
#         # Top to bottom
#         top_to_bottom = patch_embeds.view(B, -1, self.grid_size[1]).transpose(1, 2)
#         outputs['top_to_bottom'] = top_to_bottom
#
#         # Bottom to top
#         bottom_to_top = patch_embeds.view(B, -1, self.grid_size[1]).transpose(1, 2)[:, ::-1]
#         outputs['bottom_to_top'] = bottom_to_top
#
#         # Top-left to bottom-right
#         top_left_to_bottom_right = []
#         for i in range(self.grid_size[0]):
#             indices = [j * self.grid_size[0] + (i + j) for j in range(self.grid_size[1] - i)]
#             top_left_to_bottom_right.append(patch_embeds[:, indices].transpose(1, 2))
#         outputs['top_left_to_bottom_right'] = torch.cat(top_left_to_bottom_right, dim=1)
#
#         # Bottom-right to top-left
#         bottom_right_to_top_left = []
#         for i in range(self.grid_size[0]):
#             indices = [j * self.grid_size[0] + (self.grid_size[0] - i - j - 1) for j in range(self.grid_size[1] - i)]
#             bottom_right_to_top_left.append(patch_embeds[:, indices].transpose(1, 2))
#         outputs['bottom_right_to top_left'] = torch.cat(bottom_right_to top_left, dim=1)
#
#         # Bottom-left to top-right
#         bottom_left_to top_right = []
#         for i in range(self grid size[0]):
#             indices = [j for j inrange(self grid size[1])]
#             if all(idx < patch embeds.shape[2]):
#                 bottom_left_to top right.append(patch embeds[:, indices].transpose(1, 2))
#         bottom_left_to top right = torch cat(bottom左至上right, dim=1)
#
#         # Top-right to bottom-left
#         top_right至底left = []
#         for i inrange(indices):
#             indices = [j for j inrange(indices)]
#             if all(0 <= idx < patch embeds.shape[2]):
#                 top右至左bottom left.append(patch embeds[:, indices].transpose(1, 2))
#         top右至底left = torch cat(top右至左bottom left, dim=1)
#
#         return outputs


