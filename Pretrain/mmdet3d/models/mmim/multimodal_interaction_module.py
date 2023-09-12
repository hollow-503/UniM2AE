import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import math

from mmdet3d.models.builder import FUSERS

from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet3d.models.mmim.point_generator import MlvlPointGenerator
from mmdet3d.models.mmim.deform_fusion_module import MultiScaleDeformableAttention3D
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from mmcv.runner import force_fp32

__all__ = ["MMIM"]


@FUSERS.register_module()
class MMIM(nn.Module):
    def __init__(
        self, 
        volume_h,
        volume_w,
        volume_z,
        embed_dims,
        positional_encoding,
        strides,
        encoder,
    ):
        super(MMIM, self).__init__()
        
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        
        fusion_channels = embed_dims
        self.postional_encoding = build_positional_encoding(positional_encoding)
        self.strides = strides
        
        self.num_encoder_levels = encoder.transformerlayers.attn_cfgs.num_levels
        self.level_encoding = nn.Embedding(self.num_encoder_levels, fusion_channels)
        self.point_generator = MlvlPointGenerator(self.strides)
        self.fusion_encoder = build_transformer_layer_sequence(encoder)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        # v2 projection method
        # +----------------------------------------------------------------
        # self.proj_cam_downsample = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(192),
        #             nn.ReLU(inplace=True),
        #         ),
        #         nn.Sequential(
        #             nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
        #             nn.BatchNorm2d(192),
        #             nn.ReLU(inplace=True),
        #         ),
        #         nn.Sequential(
        #             nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
        #             nn.BatchNorm2d(384),
        #             nn.ReLU(inplace=True),
        #         ),
        #         nn.Sequential(
        #             nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1),
        #             nn.BatchNorm2d(768),
        #             nn.ReLU(inplace=True),
        #         )
        #     ]
        # )
        # +----------------------------------------------------------------
        
        # v1 projection method
        # +----------------------------------------------------------------
        self.proj_cam_downsample = nn.Conv2d(fusion_channels, 768, kernel_size=1)
        # +----------------------------------------------------------------
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.fusion_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        for layer in self.fusion_encoder.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention3D):
                    attn.init_weights()
                    
    @force_fp32(apply_to=('volume_coors', 'img_metas'))
    def cam_transform(self, cam_volume_feat, volume_coors, batch_mask, img_metas):
        img_augs = []
        lidar2imgs = []
        for img_meta in img_metas:
            lidar2imgs.append(img_meta['lidar2image'])
            img_augs.append(img_meta["imgs_aug"])
        lidar2imgs = np.asarray(lidar2imgs)
        lidar2imgs = cam_volume_feat.new_tensor(lidar2imgs)  # (B, N, 4, 4)
        batch_size = lidar2imgs.shape[0]
        
        cam_feats = []
        C = cam_volume_feat.size(1)
        resize_dims = img_metas[0]["img_shape"][::-1]
        
        for i in range(batch_size):
            cam_volume_feat_per_batch = cam_volume_feat[batch_mask[i], :]
            
            resize = [aug["resize"] for aug in img_augs[i]]
            crop = [aug["crop"] for aug in img_augs[i]]
            flip = [aug["flip"] for aug in img_augs[i]]
            
            lidar2img = lidar2imgs[i]
            this_coors = volume_coors[batch_mask[i], :]
            this_coors = torch.flip(this_coors[..., 1:], dims=[-1])
            
            this_coors[..., 0] = this_coors[..., 0] * 0.5 - 50.0
            this_coors[..., 1] = this_coors[..., 1] * 0.5 - 50.0
            this_coors[..., 2] = this_coors[..., 2] * 4 - 5.0
            
            this_coors = torch.cat((this_coors, torch.ones_like(this_coors[..., :1])), -1)
            num_query = this_coors.size(0)
            num_cam = lidar2img.size(0)

            this_coors = this_coors.view(1, num_query, 4).repeat(num_cam, 1, 1).unsqueeze(-1)
            lidar2img = lidar2img.view(num_cam, 1, 4, 4).repeat(1, num_query, 1, 1)
            this_coors = torch.matmul(lidar2img.to(torch.float32),
                                      this_coors.to(torch.float32)).squeeze(-1)
            this_coors = this_coors[..., 0:2] / torch.maximum(
                this_coors[..., 2:3], torch.ones_like(this_coors[..., 2:3]) * 1e-5)
            
            cam_feats_per_cam = []
            for cam_it in range(num_cam):
                this_coor = this_coors[cam_it]
                H, W = resize_dims
                this_coor[:, :2] = this_coor[:, :2] * resize[cam_it]
                this_coor[:, 0] -= crop[cam_it][0]
                this_coor[:, 1] -= crop[cam_it][1]
                if flip[cam_it]:
                    this_coor[:, 0] = resize_dims[1] - this_coor[:, 0]

                this_coor[:, 0] -= W / 2.0
                this_coor[:, 1] -= H / 2.0

                h = 0.
                rot_matrix = this_coor.new_tensor([
                    [math.cos(h), math.sin(h)],
                    [-math.sin(h), math.cos(h)],
                ])
                this_coor[:, :2] = torch.matmul(rot_matrix, this_coor[:, :2].T).T

                this_coor[:, 0] += W / 2.0
                this_coor[:, 1] += H / 2.0

                depth_coords = this_coor[:, :2].type(torch.long)

                cam_feat = cam_volume_feat_per_batch.new_zeros(resize_dims+(C, ))
                valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                            & (depth_coords[:, 0] < resize_dims[1])
                            & (depth_coords[:, 1] >= 0)
                            & (depth_coords[:, 0] >= 0))
                cam_feat[depth_coords[valid_mask, 1],
                        depth_coords[valid_mask, 0], :] = cam_volume_feat_per_batch[valid_mask]
                cam_feats_per_cam.append(cam_feat.permute(2, 0, 1))
                
            cam_feats.append(torch.stack(cam_feats_per_cam))
        
        cam_pred = torch.cat(cam_feats)
        cam_pred = F.interpolate(cam_pred, scale_factor=1/32, mode="bicubic")
        cam_pred = self.proj_cam_downsample(cam_pred)

        return cam_pred
    
    def get_reference_points(self, H, W, Z, device='cuda', dtype=torch.float):
        zs = torch.linspace(0., Z-1, Z, dtype=dtype,
                            device=device).view(1, 1, Z).expand(H, W, Z)
        xs = torch.linspace(0., W-1, W, dtype=dtype,
                            device=device).view(1, W, 1).expand(H, W, Z)
        ys = torch.linspace(0., H-1, H, dtype=dtype,
                            device=device).view(H, 1, 1).expand(H, W, Z)
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        return ref_3d

    @force_fp32(apply_to=('volume_coors', 'img_metas'))
    def cam_transform_full(self, cam_volume_feat, volume_coors, img_metas):
        img_augs = []
        lidar2imgs = []
        for img_meta in img_metas:
            lidar2imgs.append(img_meta['lidar2image'])
            img_augs.append(img_meta["imgs_aug"])
        lidar2imgs = np.asarray(lidar2imgs)
        lidar2imgs = cam_volume_feat.new_tensor(lidar2imgs)  # (B, N, 4, 4)
        batch_size = lidar2imgs.shape[0]
        
        cam_feats = []
        C = cam_volume_feat.size(2)
        resize_dims = img_metas[0]["img_shape"][::-1]
        
        for i in range(batch_size):
            this_coors = volume_coors.clone()
            cam_volume_feat_per_batch = cam_volume_feat[i]
            
            resize = [aug["resize"] for aug in img_augs[i]]
            crop = [aug["crop"] for aug in img_augs[i]]
            flip = [aug["flip"] for aug in img_augs[i]]
            
            lidar2img = lidar2imgs[i]
            
            this_coors[..., 0] = this_coors[..., 0] * 0.5 - 50.0
            this_coors[..., 1] = this_coors[..., 1] * 0.5 - 50.0
            this_coors[..., 2] = this_coors[..., 2] * 4 - 5.0
            
            this_coors = torch.cat((this_coors, torch.ones_like(this_coors[..., :1])), -1)
            num_query = this_coors.size(0)
            num_cam = lidar2img.size(0)

            this_coors = this_coors.view(1, num_query, 4).repeat(num_cam, 1, 1).unsqueeze(-1)
            lidar2img = lidar2img.view(num_cam, 1, 4, 4).repeat(1, num_query, 1, 1)
            this_coors = torch.matmul(lidar2img.to(torch.float32),
                                      this_coors.to(torch.float32)).squeeze(-1)
            this_coors = this_coors[..., 0:2] / torch.maximum(
                this_coors[..., 2:3], torch.ones_like(this_coors[..., 2:3]) * 1e-5)
            
            cam_feats_per_cam = []
            for cam_it in range(num_cam):
                this_coor = this_coors[cam_it]
                H, W = resize_dims
                this_coor[:, :2] = this_coor[:, :2] * resize[cam_it]
                this_coor[:, 0] -= crop[cam_it][0]
                this_coor[:, 1] -= crop[cam_it][1]
                if flip[cam_it]:
                    this_coor[:, 0] = resize_dims[1] - this_coor[:, 0]

                this_coor[:, 0] -= W / 2.0
                this_coor[:, 1] -= H / 2.0

                h = 0.
                rot_matrix = this_coor.new_tensor([
                    [math.cos(h), math.sin(h)],
                    [-math.sin(h), math.cos(h)],
                ])
                this_coor[:, :2] = torch.matmul(rot_matrix, this_coor[:, :2].T).T

                this_coor[:, 0] += W / 2.0
                this_coor[:, 1] += H / 2.0

                depth_coords = this_coor[:, :2].type(torch.long)

                cam_feat = cam_volume_feat_per_batch.new_zeros(resize_dims+(C, ))
                valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                            & (depth_coords[:, 0] < resize_dims[1])
                            & (depth_coords[:, 1] >= 0)
                            & (depth_coords[:, 0] >= 0))
                cam_feat[depth_coords[valid_mask, 1],
                        depth_coords[valid_mask, 0], :] = cam_volume_feat_per_batch[valid_mask]
                cam_feats_per_cam.append(cam_feat.permute(2, 0, 1))
                
            cam_feats.append(torch.stack(cam_feats_per_cam))
        
        cam_pred = torch.cat(cam_feats)
        
        for layer in self.proj_cam_downsample:
            cam_pred = layer(cam_pred)

        cam_pred = F.interpolate(cam_pred, scale_factor=1/4, mode="bicubic")

        return cam_pred

    def fuse(self, volume_feats):
        batch_size, C, H, W, Z = volume_feats[0].shape
        
        encoder_inputs_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shape_list = []
        reference_points_list = []
        
        for i in range(self.num_encoder_levels):
            feat = volume_feats[i]
            # X, Y, Z
            volume_shape = volume_feats[i].shape[-3:]
            padding_mask_resized = feat.new_zeros((batch_size,)+volume_shape, 
                                                  dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1, 1) + pos_embed
            
            reference_points = self.point_generator.single_level_grid_priors(
                volume_shape, i, device=feat.device)
            
            factor = feat.new_tensor([volume_shape[::-1]]) * self.strides[i]
            reference_points = reference_points / factor
            
            volume_projected = feat.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)
            
            encoder_inputs_list.append(volume_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shape_list.append(volume_shape)
            reference_points_list.append(reference_points)
            
        padding_mask = torch.cat(padding_mask_list, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shape_list, dtype=torch.long, device=volume_feats[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        encoder_inputs = torch.cat(encoder_inputs_list, dim=0)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=0)
        
        volume_feat = self.fusion_encoder(
            query=encoder_inputs,
            key=None,
            value=None,
            query_pos=level_positional_encodings,
            key_pos=None,
            attn_masks=None,
            key_padding_mask=None,
            query_key_padding_mask=padding_mask,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_radios=valid_radios)
        
        lidar_feat, camera_feat = torch.split(volume_feat, level_start_index[1])
        
        lidar_feat = lidar_feat.permute(1, 2, 0).view(batch_size, -1, H, W, Z)
        lidar_feat = self.relu1(lidar_feat+volume_feats[0])
        camera_feat = camera_feat.permute(1, 2, 0).view(batch_size, -1, H, W, Z)
        camera_feat = self.relu2(camera_feat+volume_feats[1])
        
        return lidar_feat, camera_feat
    
    def forward(self, volume_feats, lidar_x, img_metas):
        lidar_feat, camera_feat = self.fuse(volume_feats)
        
        # The performance of v1 projection is slightly lower than v2 projection method
        # But the released weight was trained with v1 projection method
        # Therefore, we matain the usage of the v1 projection method here
        
        # v1 projection method
        # +----------------------------------------------------------------
        camera_feat = camera_feat.flatten(-3)
        C = camera_feat.size(1)
        cam2token = lidar_x[0]['output'].new_zeros([lidar_x[3], C])
        cam2token_bev = [i[:, lidar_x[1][n]].t() for n, i in enumerate(camera_feat)]
        for n, batch in enumerate(cam2token_bev):
            cam2token[lidar_x[2][n]] = batch
        
        cameara_proj_feat = self.cam_transform(cam2token, lidar_x[0]['voxel_coors'], lidar_x[2], img_metas)
        # +----------------------------------------------------------------
        
        # v2 projection method
        # +----------------------------------------------------------------
        # camera_feat = camera_feat.flatten(2).permute(0, 2, 1)
        # volume_index = self.get_reference_points(
        #     self.volume_h, 
        #     self.volume_w, 
        #     self.volume_z, 
        #     device=camera_feat.device
        # )
        # cameara_proj_feat = self.cam_transform_full(camera_feat, volume_index, img_metas)
        # +----------------------------------------------------------------
        
        lidar_feat = lidar_feat.flatten(-3)
        C = lidar_feat.size(1)
        lidar_proj_feat = lidar_x[0]['output'].new_zeros([lidar_x[3], C])
        lidar_proj_feat_volume = [i[:, lidar_x[1][n]].t() for n, i in enumerate(lidar_feat)]
        for n, batch in enumerate(lidar_proj_feat_volume):
            lidar_proj_feat[lidar_x[2][n]] = batch

        return cameara_proj_feat, lidar_proj_feat