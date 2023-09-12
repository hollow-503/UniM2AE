import torch.nn as nn
from timm.models.vision_transformer import Block
import torch
from mmdet3d.models.utils.pos_embed import get_2d_sincos_pos_embed
from mmdet3d.models.builder import HEADS
from functools import partial
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


@HEADS.register_module()
class MAESwinDecoder(nn.Module):
    def __init__(self, num_patches, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], mlp_ratio=4.,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 norm_pix_loss=False, block_cls=Block, decoder=None):
        super().__init__()
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_patches = num_patches
        embed_dim = embed_dim * 2**(len(depths) - 1)
        patch_size = patch_size * 2**(len(depths) - 1)
        self.final_patch_size = patch_size
        self.decoder_embed = nn.Identity() if embed_dim == decoder_embed_dim else nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches[0] * num_patches[1], decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            block_cls(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # encoder to decoder

        self.cross_modality_module = None 
        if decoder is not None:
            self.cross_modality_module = build_transformer_layer_sequence(decoder)
            self.level_start_index = nn.Parameter(torch.as_tensor((0), dtype=torch.long), requires_grad=False)
            self.valid_ratios = nn.Parameter(torch.tensor([[[1., 1.]]], dtype=torch.float), requires_grad=False)
            self.decoder_pos_embed = nn.Parameter(torch.randn(num_patches[0] * num_patches[1], 1, decoder_embed_dim))
            self.reference_camera = nn.Linear(decoder_embed_dim, 2)
            self.lidar2token = nn.Conv2d(128, decoder_embed_dim, kernel_size=1)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.cross_modality_module is None:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self, imgs, patch_size=None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size or self.final_patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size or self.final_patch_size
        h = self.num_patches[0]
        w = self.num_patches[1]
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
        
    def forward(self, x, ids_restore, lidar_x=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if lidar_x is not None:
            _, _, H, W = lidar_x.shape
            lidar_x = self.lidar2token(lidar_x).flatten(-2)
            lidar_x = torch.cat([i.repeat(6, 1, 1) for i in lidar_x])
            lidar_x = lidar_x.permute(0, 2, 1)

            spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=lidar_x.device)
            valid_ratios = self.valid_ratios.repeat(x.shape[0], 1, 1)

            reference_point = self.reference_camera(x_).sigmoid()

            x_c, _ = self.cross_modality_module(query=x_.permute(1, 0, 2), 
                                                key=None, 
                                                value=lidar_x.permute(1, 0, 2), 
                                                query_pos=self.decoder_pos_embed,
                                                key_padding_mask=None,
                                                reference_points=reference_point,
                                                spatial_shapes=spatial_shapes,
                                                level_start_index=self.level_start_index,
                                                valid_ratios=valid_ratios,
                                                reg_branches=None)
            x = x_c.permute(1, 0, 2)
        else:
            # add pos embed
            x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
