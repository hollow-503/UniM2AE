from mmdet.models import BACKBONES

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch.utils.checkpoint import checkpoint
from mmdet3d.ops import flat2window_v2, window2flat_v2


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class WindowAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, batch_first=False, layer_id=None):
        super().__init__()
        self.nhead = nhead

        # from mmdet3d.models.transformer.my_multi_head_attention import MyMultiheadAttention
        # self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.exe_counter = 0

        self.layer_id = layer_id

    def forward(self, feat_2d, pos_dict, ind_dict, key_padding_dict):
        '''
        Args:

        Out:
            shifted_feat_dict: the same type as window_feat_dict
        '''

        out_feat_dict = {}

        feat_3d_dict = flat2window_v2(feat_2d, ind_dict)

        for name in feat_3d_dict:
            #  [n, num_token, embed_dim]
            pos = pos_dict[name]

            feat_3d = feat_3d_dict[name]
            feat_3d = feat_3d.permute(1, 0, 2)

            v = feat_3d

            if pos is not None:
                pos = pos.permute(1, 0, 2)
                assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape}, feat_shape:{feat_3d.shape}'
                q = k = feat_3d + pos
            else:
                q = k = feat_3d

            key_padding_mask = key_padding_dict[name]
            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)

        results = window2flat_v2(out_feat_dict, ind_dict)
        
        return results


class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, layer_id=None, mlp_dropout=0, layer_cfg=dict()):
        super().__init__()
        assert not batch_first, 'Current version of PyTorch does not support batch_first in MultiheadAttention. After upgrading pytorch, do not forget to check the layout of MLP and layer norm to enable batch_first option.'
        self.batch_first = batch_first
        self.win_attn = WindowAttention(d_model, nhead, dropout, layer_id=layer_id)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        use_bn = layer_cfg.get('use_bn', False)
        if use_bn:
            self.norm1 = build_norm_layer(dict(type='naiveSyncBN1d'), d_model)[1]
            self.norm2 = build_norm_layer(dict(type='naiveSyncBN1d'), d_model)[1]
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)

        self.activation = _get_activation_fn(activation)
        self.post_norm = layer_cfg.get('post_norm', True)
        self.fp16_enabled=False

    @auto_fp16(apply_to=('src'))
    def forward(
        self,
        src,
        pos_dict,
        ind_dict,
        key_padding_mask_dict,
        ):
        if self.post_norm:
            src2 = self.win_attn(src, pos_dict, ind_dict, key_padding_mask_dict) #[N, d_model]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        else:
            src2 = self.norm1(src)
            src2 = self.win_attn(src2, pos_dict, ind_dict, key_padding_mask_dict) #[N, d_model]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        return src


class BasicShiftBlockV2(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, block_id=-100, layer_cfg=dict()):
        super().__init__()

        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 0, layer_cfg=layer_cfg)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 1, layer_cfg=layer_cfg)
        # BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i], dropout, activation, batch_first=False)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
        self,
        src,
        pos_dict_list,
        ind_dict_list,
        key_mask_dict_list,
        using_checkpoint=False,
        ):
        num_shifts = len(pos_dict_list)
        assert num_shifts in (1, 2)

        output = src
        for i in range(2):

            this_id = i % num_shifts
            pos_dict = pos_dict_list[this_id]
            ind_dict = ind_dict_list[this_id]
            key_mask_dict = key_mask_dict_list[this_id]

            layer = self.encoder_list[i]
            if using_checkpoint and self.training:
                output = checkpoint(layer, output, pos_dict, ind_dict, key_mask_dict)
            else:
                output = layer(output, pos_dict, ind_dict, key_mask_dict)

        return output


@BACKBONES.register_module()
class SSTv2(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[],
        nhead=[],
        num_blocks=6,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        num_attached_conv=2,
        conv_in_channel=64,
        conv_out_channel=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        in_channel=None,
        conv_kwargs=None,
        checkpoint_blocks=[],
        layer_cfg=dict(),
        masked=False,
        ):
        super().__init__()
        if conv_kwargs == None:
            conv_kwargs=[
                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                dict(kernel_size=3, dilation=2, padding=2, stride=1),
            ]
        
        self.d_model = d_model
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )

        self.block_list = nn.ModuleList(block_list)
            
        self._reset_parameters()

        self.output_shape = output_shape

        self.debug = debug

        self.masked = masked

        self.num_attached_conv = num_attached_conv

        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                    )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel)[1],
                        nn.ReLU(inplace=True)
                    )
                # delattr(convnormrelu[1], 'fp16_enabled')
                conv_list.append(convnormrelu)
            
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, voxel_info):
        '''
        '''
        num_shifts = 2 
        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

        device = voxel_info['voxel_coors'].device
        batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        voxel_feat = voxel_info['voxel_feats']
        ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)
        for i, block in enumerate(self.block_list):
            output = block(output, pos_embed_list, ind_dict_list, 
                padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)

        # If masked we want to send the output to the decoder and not a FPN how requires dense bev image
        if not self.masked:
            if len(self.output_shape) == 2:
                output, indices_back, _ = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)
            elif len(self.output_shape) == 3:
                output, indices_back, _ = self.recover_volume(output, voxel_info['voxel_coors'], batch_size)
            else:
                raise ValueError("The output_shape should be [H, W, Z] or [H, W]")

            if self.num_attached_conv > 0:
                for conv in self.conv_layer:
                    output = conv(output)

            return output
        else:
            if self.num_attached_conv > 0:
                if len(self.output_shape) == 2:
                    bev_out, indices_back, batch_masks = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)
                elif len(self.output_shape) == 3:
                    output, indices_back, _ = self.recover_volume(output, voxel_info['voxel_coors'], batch_size)
                else:
                    raise ValueError("The output_shape should be [H, W, Z] or [H, W]")
                
                for conv in self.conv_layer:
                    bev_out = conv(bev_out)
                voxel_info["output"] = bev_out
                
                return voxel_info, indices_back, batch_masks, output.shape[0]

            voxel_info["output"] = output
            return voxel_info

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def recover_volume(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, nx, ny, nz]
        '''
        nx, ny, nz = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        indices_back = []
        batch_masks = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny * nz,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 3] * ny * nz + this_coors[:, 2] * nz + this_coors[:, 1]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)
            indices_back.append(indices)
            batch_masks.append(batch_mask)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, nx, ny, nz)

        return batch_canvas, indices_back, batch_masks

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, nx, ny]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        indices_back = []
        batch_masks = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 3] * ny + this_coors[:, 2]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)
            indices_back.append(indices)
            batch_masks.append(batch_mask)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, nx, ny)

        return batch_canvas, indices_back, batch_masks
    
    def recover_bev_indices(self, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        _, nx = self.output_shape

        indices_back = []
        batch_masks = []
        for batch_itt in range(batch_size):
            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)

            indices_back.append(indices)
            batch_masks.append(batch_mask)

        return indices_back, batch_masks
