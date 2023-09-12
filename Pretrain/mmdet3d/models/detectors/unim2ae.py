import torch.nn as nn
import torch
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.dynamic_voxelnet import DynamicVoxelNet
from mmdet3d.models.builder import build_backbone, build_head

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.models.builder import build_vtransform, build_fuser

import numpy as np

import matplotlib.pyplot as plt


def show_image(image, title=''):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def vis_image(ori_img, pred_img, mask, model, out_dir):
    ori_img = model.camera_decoder.patchify(ori_img)
    mean = ori_img.mean(dim=-1, keepdim=True)
    var = ori_img.var(dim=-1, keepdim=True)
    ori_img = model.camera_decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    # run MAE
    pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = model.camera_decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.camera_decoder.final_patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = model.camera_decoder.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 12]
    
    for i in range(x.shape[0]):
        plt.subplot(6, 4, i*4+1)
        show_image(x[i], "original")

        plt.subplot(6, 4, i*4+2)
        show_image(im_masked[i], "masked")

        plt.subplot(6, 4, i*4+3)
        show_image(y[i], "reconstruction")

        plt.subplot(6, 4, i*4+4)
        show_image(im_paste[i], "reconstruction + visible")

    plt.savefig(f"{out_dir}.png")


@DETECTORS.register_module()
class UniM2AE(DynamicVoxelNet):
    def __init__(self, 
                 voxel_layer, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone,
                 fusion_module,
                 camera_backbone,
                 camera_vtransform,
                 camera_decoder=None,
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None, 
                 freeze=None,
                 ):
        super(UniM2AE, self).__init__(
            voxel_layer=voxel_layer, 
            voxel_encoder=voxel_encoder, 
            middle_encoder=middle_encoder, 
            backbone=backbone, 
            neck=neck, 
            bbox_head=bbox_head, 
            train_cfg=train_cfg, 
            test_cfg=test_cfg, 
            pretrained=pretrained, 
            init_cfg=init_cfg, 
            freeze=freeze,
            )
        self.init_weights()

        self.camera_encoder = nn.ModuleDict(
            {
                "backbone": build_backbone(camera_backbone),
                "vtransform": build_vtransform(camera_vtransform),
            }
        )
        self.relu = nn.ReLU(inplace=True)
        self.camera_decoder = build_head(camera_decoder)
        
        self.fusion_module = build_fuser(fusion_module)
        
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        self.deblock_lidar = nn.Sequential(
                build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=128,
                out_channels=192,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True),
        )
    
    def extract_feat(self, points, img_metas):
        lidar_x = super().extract_feat(points, img_metas)[1]
        lidar_volume_embed = self.deblock_lidar(lidar_x[0]['output'])
        
        return lidar_x, lidar_volume_embed
    
    @torch.no_grad()
    def test_pretrain(self, img, 
                      points, 
                      img_metas,
                      out_dir):
        """Test function without augmentaiton."""
        batch_size = len(points)
        vx, vy, vz = self.middle_encoder.sparse_shape

        lidar_x, lidar_volume_embed = self.extract_feat(points, img_metas)
        
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        camera_x, camera_mask, camera_ids_restore = self.camera_encoder["backbone"](img, camera_only=True)
        camera_volume_embed, camera_x = self.camera_encoder["vtransform"](
            camera_x, 
            (B, N, C, H, W),
            camera_ids_restore, 
            img_metas,
        )

        cam_proj_feat, lidar_proj_feat = self.fusion_module(
            [lidar_volume_embed, camera_volume_embed], 
            lidar_x,
            img_metas
        )
        
        cam_pred = self.relu(cam_proj_feat + camera_x.view(B*N, -1, H//32, W//32))
        cam_pred = cam_pred.flatten(2).permute(0, 2, 1)
        cam_pred = self.camera_decoder(cam_pred, camera_ids_restore)
        
        lidar_x[0]['output'] = lidar_proj_feat
        
        vis_image(img, cam_pred, camera_mask, self, out_dir)
        
        outs = self.bbox_head(lidar_x[0], show=True)
        pred_dict = outs[0]
        voxel_coors = pred_dict["voxel_coors"]
        masked_voxel_coors = pred_dict["masked_voxel_coors"]
        unmasked_voxel_coors = pred_dict["unmasked_voxel_coors"]

        occupied = None
        if "pred_occupied" in pred_dict:
            occupied = -torch.ones((batch_size, vx, vy), dtype=torch.long, device=pred_dict["pred_occupied"].device)
            index = (voxel_coors[:, 0], voxel_coors[:, 3], voxel_coors[:, 2])  # b ,x, y
            unmasked_index = (
                unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])
            gt_occupied = pred_dict["gt_occupied"].long()+1  # 1 -> real voxel, 2 -> fake voxel
            occupied[index] = 2 * gt_occupied  # 2 -> real voxel, 4 -> fake voxel
            occupied[unmasked_index] -= 2  # 0 -> unmasked voxels 2 -> masked voxel, 4 -> fake voxel
            occupied[index] += (torch.sigmoid(pred_dict["pred_occupied"]) + 0.5).long()
            # 0 -> unmasked voxel predicted as real,
            # 1 -> unmasked voxel predicted as fake,
            # 2 -> masked voxel predicted as real,
            # 3 -> masked voxel predicted as fake,
            # 4 -> fake voxel predicted as real,
            # 5 -> fake voxel predicted as fake

        gt_num_points = None
        diff_num_points = None
        if "pred_num_points_masked" in pred_dict:
            device = pred_dict["pred_num_points_masked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device)
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device)
            index = (masked_voxel_coors[:, 0], masked_voxel_coors[:, 3], masked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_masked = pred_dict["pred_num_points_masked"]
            gt_num_points_masked = pred_dict["gt_num_points_masked"]
            gt_num_points[index] = gt_num_points_masked.long()
            diff_num_points[index] = gt_num_points_masked.float()-pred_num_points_masked
        if "pred_num_points_unmasked" in pred_dict:
            device = pred_dict["pred_num_points_unmasked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device) if gt_num_points is None else gt_num_points
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device) if diff_num_points is None else diff_num_points
            index = (unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_unmasked = pred_dict["pred_num_points_unmasked"]
            gt_num_points_unmasked = pred_dict["gt_num_points_unmasked"]
            gt_num_points[index] = gt_num_points_unmasked.long()
            diff_num_points[index] = gt_num_points_unmasked.float() - pred_num_points_unmasked

        points = []
        batch = []
        if "pred_points_masked" in pred_dict:
            pred_points_masked = pred_dict["pred_points_masked"].clone()  # M, num_chamfer_points, 3
            M, n, C = pred_points_masked.shape
            x_shift = (masked_voxel_coors[:, 3].type_as(pred_points_masked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset)  # M
            y_shift = (masked_voxel_coors[:, 2].type_as(pred_points_masked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset)  # M
            z_shift = (masked_voxel_coors[:, 1].type_as(pred_points_masked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset)  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            pred_points_masked[..., 0] = pred_points_masked[..., 0] * self.voxel_encoder.vx / 2  # [-1, 1] -> [voxel_encoder.vx/2, voxel_encoder.vx/2]
            pred_points_masked[..., 1] = pred_points_masked[..., 1] * self.voxel_encoder.vy / 2  # [-1, 1] -> [voxel_encoder.vy/2, voxel_encoder.vy/2]
            pred_points_masked[..., 2] = pred_points_masked[..., 2] * self.voxel_encoder.vz / 2  # [-1, 1] -> [voxel_encoder.vz/2, voxel_encoder.vz/2]
            batch.append(masked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            
            gt_points_unmasked = pred_dict["gt_points_unmasked"].clone()
            gt_points_unmasked[..., 0] = gt_points_unmasked[..., 0] * self.voxel_encoder.vx / 2  # [-1, 1] -> [voxel_encoder.vx/2, voxel_encoder.vx/2]
            gt_points_unmasked[..., 1] = gt_points_unmasked[..., 1] * self.voxel_encoder.vy / 2  # [-1, 1] -> [voxel_encoder.vy/2, voxel_encoder.vy/2]
            gt_points_unmasked[..., 2] = gt_points_unmasked[..., 2] * self.voxel_encoder.vz / 2  # [-1, 1] -> [voxel_encoder.vz/2, voxel_encoder.vz/2]
            x_shift_m = (unmasked_voxel_coors[:, 3].type_as(pred_points_masked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset)  # M
            y_shift_m = (unmasked_voxel_coors[:, 2].type_as(pred_points_masked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset)  # M
            z_shift_m = (unmasked_voxel_coors[:, 1].type_as(pred_points_masked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset)  # M
            shift_m = torch.cat([x_shift_m.unsqueeze(-1), y_shift_m.unsqueeze(-1), z_shift_m.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            gt_points_unmasked = gt_points_unmasked + shift_m
            
            points.append((pred_points_masked + shift).reshape(-1, 3))
            points.append(gt_points_unmasked.reshape(-1, 3))

        if "pred_points_unmasked" in pred_dict:
            pred_points_unmasked = pred_dict["pred_points_unmasked"]  # N-M, num_chamfer_points, 3
            M, n, C = pred_points_unmasked.shape
            x_shift = unmasked_voxel_coors[:, 3].type_as(pred_points_unmasked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset  # M
            y_shift = unmasked_voxel_coors[:, 2].type_as(pred_points_unmasked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset  # M
            z_shift = unmasked_voxel_coors[:, 1].type_as(pred_points_unmasked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            batch.append(unmasked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            points.append((pred_points_unmasked + shift).reshape(-1, 3))
        points = torch.cat(points, dim=0) if points else None
        batch = torch.cat(batch, dim=0) if batch else None

        return {
            "occupied_bev": occupied,
            "gt_num_points_bev": gt_num_points,
            "diff_num_points_bev": diff_num_points,
            "points": points,
            "points_batch": batch,
            "gt_points": pred_dict["gt_points"],
            "gt_points_unmasked": gt_points_unmasked,
            "gt_points_batch":  pred_dict["gt_point_coors"][:, 0],
            "point_cloud_range": self.voxel_encoder.point_cloud_range,
            "voxel_shape": (self.voxel_encoder.vx, self.voxel_encoder.vy, self.voxel_encoder.vz)
        }
    
    def forward(self,
                img,
                points,
                img_metas,
                return_loss=True, 
                pretrain=False,
                out_dir=None,
                **kwargs,
                ):
        
        if return_loss:
            lidar_x, lidar_volume_embed = self.extract_feat(points, img_metas)
            
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
            camera_x, camera_mask, camera_ids_restore = self.camera_encoder["backbone"](img, camera_only=True)
            camera_volume_embed, camera_x = self.camera_encoder["vtransform"](
                camera_x, 
                (B, N, C, H, W),
                camera_ids_restore, 
                img_metas
            )

            cam_proj_feat, lidar_proj_feat = self.fusion_module(
                [lidar_volume_embed, camera_volume_embed], 
                lidar_x,
                img_metas
            )
            
            cam_pred = self.relu(cam_proj_feat + camera_x.view(B*N, -1, H//32, W//32))
            cam_pred = cam_pred.flatten(2).permute(0, 2, 1)
            cam_pred = self.camera_decoder(cam_pred, camera_ids_restore)

            lidar_x[0]['output'] = lidar_proj_feat

            lidar_pred = self.bbox_head(lidar_x[0])
            losses = self.bbox_head.loss(*lidar_pred)
            losses['camera_loss'] = self.camera_decoder.forward_loss(img, cam_pred, camera_mask)
            
            return losses
        
        elif pretrain:
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))
                    
            img = [img] if img is None else img
            return self.test_pretrain(img[0], points[0], img_metas[0], out_dir)
        
        else:
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))

            num_augs = len(points)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(points), len(img_metas)))
                
            if num_augs == 1:
                img = [img] if img is None else img
                return self.simple_test(points[0], img_metas[0], img[0])
            else:
                return self.aug_test(points, img_metas, img)

