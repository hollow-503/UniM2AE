import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


__all__ = ["VolumeTransform"]


@VTRANSFORMS.register_module()
class VolumeTransform(nn.Module):
    def __init__(
        self,
        volume_h,
        volume_w,
        volume_z,
        embed_dims,
        in_channels,
        volume_encoder,
        mask_ratio,
    ) -> None:
        super(VolumeTransform, self).__init__()
        
        self.mask_token = None
        if mask_ratio > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.volume_embedding = nn.Embedding(
            self.volume_h* self.volume_w* self.volume_z, 
            embed_dims
        )
        
        self.transfer_conv = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d', bias=True),
                in_channels=in_channels,
                out_channels=embed_dims,
                kernel_size=1,
                stride=1
            ), 
            nn.ReLU(inplace=True)
        )
        self.level_embeds = nn.Parameter(torch.Tensor(1, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(6, embed_dims))
        
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg=dict(type='deconv3d', bias=False)

        self.deblock_camera = nn.Sequential(
            build_upsample_layer(
                upsample_cfg,
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=256,
                out_channels=192,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True)
        )
        
        self.volume_encoder = build_transformer_layer_sequence(volume_encoder)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)
        
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=.02)
        
    def forward(
        self, 
        camera_x,
        img_shape,
        camera_ids_restore,
        img_metas
    ):
        B, N, C, H, W = img_shape
        if self.mask_token is not None:
            mask_tokens = self.mask_token.repeat(camera_x.shape[0], camera_ids_restore.shape[1] - camera_x.shape[1], 1)
            camera_x = torch.cat([camera_x, mask_tokens], dim=1)

        camera_x = torch.gather(camera_x, dim=1, index=camera_ids_restore.unsqueeze(-1).repeat(1, 1, camera_x.shape[2]))  # unshuffle
        camera_x = camera_x.permute(0, 2, 1).view(B, N, -1, H//32, W//32)
        
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
        
        volume_embed = self.volume_encoder(
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

        return volume_embed, camera_x