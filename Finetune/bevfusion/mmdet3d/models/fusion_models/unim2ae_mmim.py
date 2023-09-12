import torch.nn as nn
import torch
from mmdet3d.models import FUSIONMODELS
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from mmcv.cnn import build_conv_layer, build_norm_layer

from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet3d.models.mmim.point_generator import MlvlPointGenerator

from mmcv.runner import auto_fp16, force_fp32
import math
import numpy as np
import torch.nn.functional as F
from mmdet3d.models.mmim.deform_fusion_module import MultiScaleDeformableAttention3D
from .base import Base3DFusionModel

from typing import Any, Dict 

from mmdet3d.models.builder import (
    build_backbone,
    build_head,
    build_neck,
    build_voxel_encoder,
    build_vtransform,
    build_middle_encoder,
)
from mmdet3d.ops import Voxelization, DynamicScatter


@FUSIONMODELS.register_module()
class UniM2AE_MMIM(Base3DFusionModel):
    def __init__(
        self, 
        encoders: Dict[str, Any],
        fusion_module: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ):
        super(UniM2AE_MMIM, self).__init__()
        
        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0 or \
            encoders["lidar"]["voxelize"].get("Voxelization", False):
                if 'Voxelization' in encoders["lidar"]["voxelize"]:
                    encoders["lidar"]["voxelize"].pop("Voxelization")
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "voxel_encoder": build_voxel_encoder(encoders["lidar"]["voxel_encoder"]) if encoders["lidar"].get("voxel_encoder", None) else None,
                    "middle_encoder": build_middle_encoder(encoders["lidar"]["middle_encoder"]) if encoders["lidar"].get("middle_encoder", None) else None,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
        
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        conv_cfg=dict(type='Conv3d', bias=False)
        
        self.deblock_camera = nn.Sequential(
            build_conv_layer(conv_cfg,
                in_channels=80,
                out_channels=192,
                kernel_size=3,
                stride=2,
                padding=1),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True),
        )
        
        self.deblock_lidar = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels=128,
                out_channels=192,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True),
        )

        self.deblock_fusion = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False),
            build_norm_layer(dict(type='BN', eps=1.0e-3, momentum=0.01), 192)[1],
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, bias=False),
            build_norm_layer(dict(type='BN', eps=1.0e-3, momentum=0.01), 192)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=1, stride=1, bias=False),
            build_norm_layer(dict(type='BN', eps=1.0e-3, momentum=0.01), 128)[1],
            nn.ReLU(inplace=True),
        )
        
        self.postional_encoding = build_positional_encoding(fusion_module["positional_encoding"])
        self.strides = fusion_module["strides"]
        self.num_encoder_levels = fusion_module["encoder"].transformerlayers.attn_cfgs.num_levels
        self.level_encoding = nn.Embedding(self.num_encoder_levels, fusion_module["embed_dims"])
        self.point_generator = MlvlPointGenerator(self.strides)
        self.fuser = build_transformer_layer_sequence(fusion_module["encoder"])
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.init_weights()

    def init_weights(self):
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()
        
        for p in self.fuser.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        for layer in self.fuser.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention3D):
                    attn.init_weights()
    
    def volume_vtransform_embed(self, camera_x, img_metas):
        B, N, C, H, W = camera_x.shape
        dtype = camera_x.dtype
        volume_queries = self.volume_embedding.weight.to(dtype)
        
        volume_queries = volume_queries.unsqueeze(1).repeat(1, B, 1)
        view_features = self.transfer_conv(camera_x.view(B*N, C, H, W))
        view_features = view_features.view(B, N, -1, H, W).flatten(3).permute(1, 0, 3, 2)
        view_features = view_features + self.cams_embeds[:, None, None, :].to(view_features.dtype)
        view_features = view_features + self.level_embeds[None, None, 0:1, :].to(view_features.dtype)
        spatial_shapes = torch.as_tensor([[H, W]], dtype=torch.long, device=view_features.device)
        level_start_index = spatial_shapes.new_zeros((1,))
        view_features = view_features.permute(0, 2, 1, 3)
        
        volume_embed = self.encoders["camera"]["vtransform"](
            volume_queries,
            view_features,
            view_features,
            volume_h=self.volume_h,
            volume_w=self.volume_w,
            volume_z=self.volume_z,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas
        ).reshape(B, self.volume_z, self.volume_h, self.volume_w, -1).permute(0, 4, 3, 2, 1)
        
        volume_embed = self.deblock_camera(volume_embed)

        return volume_embed
    
    def fusion_module(self, volume_feats):
        batch_size, C, W, H, Z = volume_feats[0].shape
        
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
        
        volume_feat = self.fuser(
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
            valid_radios=valid_radios
        )
        
        lidar_feat, camera_feat = torch.split(volume_feat, level_start_index[1])
        
        lidar_feat = lidar_feat.permute(1, 2, 0).view(batch_size, -1, W, H, Z)
        lidar_feat = self.relu1(lidar_feat+volume_feats[0])
        camera_feat = camera_feat.permute(1, 2, 0).view(batch_size, -1, W, H, Z)
        camera_feat = self.relu2(camera_feat+volume_feats[1])
        
        return lidar_feat, camera_feat
    
    def extract_camera_features(
        self, 
        x, 
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        
        if not isinstance(x, torch.Tensor):
            x = x[0]
            
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        # x = self.volume_vtransform_embed(x, img_metas)
        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        
        x = self.deblock_camera(x)
         
        return x
    
    @torch.no_grad()
    @force_fp32()
    def voxelize_sst(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.encoders["lidar"]["voxelize"](res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    
    def extract_lidar_features(self, x) -> torch.Tensor:
        voxels, coors = self.voxelize_sst(x)
        batch_size = coors[-1, 0].item() + 1
        voxel_features, feature_coors = self.encoders["lidar"]["voxel_encoder"](voxels, coors)
        x = self.encoders["lidar"]["middle_encoder"](voxel_features, feature_coors, batch_size)
        x = self.encoders["lidar"]["backbone"](x)
        return x
    
    @auto_fp16(apply_to=('img, points'))
    def forward(self,
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                gt_masks_bev=None,
                **kwargs,
                ):
        
        if isinstance(img, list):
            raise NotImplementedError
        else:
            output = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_masks_bev
            )
            return output
    
    @auto_fp16(apply_to=("img, points"))
    def forward_single(self, 
                       img, 
                       points,
                       camera2ego,
                       lidar2ego,
                       lidar2camera,
                       lidar2image,
                       camera_intrinsics,
                       camera2lidar,
                       img_aug_matrix,
                       lidar_aug_matrix,
                       metas,
                       gt_bboxes_3d=None,
                       gt_labels_3d=None,
                       gt_masks_bev=None,
    ):
        
        features = []
        for sensor in (
            self.encoders if not self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img, 
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
                feature = self.deblock_lidar(feature)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
        
        if not self.training:
            # avoid OOM
            features = features[::-1]
            
        x = self.fusion_module(features)
        x = torch.cat(x, dim=1)
        x = torch.cat(x.unbind(dim=-1), 1)
        
        x = self.deblock_fusion(x)
        
        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            # viz = True
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    if isinstance(bboxes, torch.Tensor):
                        return bboxes
                    else:
                        for k, (boxes, scores, labels) in enumerate(bboxes):
                            outputs[k].update(
                                {
                                    "boxes_3d": boxes.to("cpu"),
                                    "scores_3d": scores.cpu(),
                                    "labels_3d": labels.cpu(),
                                }
                            )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
