_base_ = [
    'sst_base.py',
    'nuscenes.py',
    'cosine_2x.py',
    'default_runtime.py',
]

# -------------------model--------------------
voxel_size = (0.5, 0.5, 4)
sparse_shape = (200, 200, 2)
point_cloud_range = [-50, -50, -5, 50, 50, 3]
encoder_blocks = 8
img_size = (256, 704)

use_chamfer, use_num_points, use_fake_voxels = True, True, True
relative_error = False
masking_ratio = 0.7
masking_ratio_img = 0.75
fake_voxels_ratio = 0.1
loss_weights = dict(
    loss_occupied=1.,
    loss_num_points_masked=1.,
    loss_chamfer_src_masked=1.,
    loss_chamfer_dst_masked=1.,
    loss_num_points_unmasked=0.,
    loss_chamfer_src_unmasked=0.,
    loss_chamfer_dst_unmasked=0.
)
window_shape = (16, 16, 1) # 12 * 0.32m
shifts_list = [(0, 0), (window_shape[0]//2, window_shape[1]//2)]
drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    type='UniM2AE',

    fusion_module=dict(
        type='MMIM',
        volume_h=sparse_shape[0],
        volume_w=sparse_shape[1],
        volume_z=sparse_shape[2],
        embed_dims=192,
        strides=[1, 1],
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=192 // 3,
            normalize=True
        ),
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=3,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=192,
                    num_heads=8,
                    num_levels=2,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    embed_dims=192),
                feedforward_channels=192 * 4,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None
        ),
    ),

    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    
    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        return_gt_points=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='SSTInputLayerV2Masked',
        window_shape=window_shape,
        sparse_shape=sparse_shape,
        voxel_size=voxel_size,
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        mute=True,
        masking_ratio=masking_ratio,
        drop_points_th=100,
        pred_dims=3,  # x, y, z
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        fake_voxels_ratio=fake_voxels_ratio
    ),
    
    backbone=dict(
        type='SSTv2',
        d_model=[128, ] * encoder_blocks,
        nhead=[8, ] * encoder_blocks,
        num_blocks=encoder_blocks,
        dim_feedforward=[256, ] * encoder_blocks,
        output_shape=[200, 200],  # tot_point_cloud_range / voxel_size (50+50)/0.5
        num_attached_conv=0,
        # checkpoint_blocks=[0, 1, 2, 3, 4, 5, 6, 7], # Save the gpu memory but get slower
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
        masked=True,
    ),

    neck=dict(
        _delete_=True,
        type='SSTv2Decoder',
        d_model=[128, ] * 6,
        nhead=[8, ] * 6,
        num_blocks=6,
        dim_feedforward=[256, ] * 6,
        output_shape=sparse_shape,
        num_attached_conv=-1,
        debug=True,
        use_fake_voxels=use_fake_voxels,
    ),

    bbox_head=dict(
        _delete_=True,
        type='ReconstructionHead',
        in_channels=192,
        feat_channels=192,
        num_chamfer_points=10,
        pred_dims=3,
        only_masked=True,
        relative_error=relative_error,
        loss_weights=loss_weights,
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
    ),
    
    camera_backbone=dict(
        type='MAESwinEncoder',
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=True, # Save the gpu memory but get slower
        mask_ratio=masking_ratio_img,
    ),
    
    camera_vtransform=dict(
        type='VolumeTransform',
        volume_h=sparse_shape[0]//2,
        volume_w=sparse_shape[1]//2,
        volume_z=sparse_shape[2]//2,
        embed_dims=256,
        in_channels=768,
        mask_ratio=masking_ratio_img,
        volume_encoder=dict(
            type='volumeEncoder',
            num_layers=6,
            pc_range=point_cloud_range,
            return_intermediate=False,
            transformerlayers=dict(
                type='volumeLayer',
                attn_cfgs=[
                    dict(
                        type='SpatialCrossAttention',
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=256,
                            num_points=4,
                            num_levels=1),
                        embed_dims=256,
                    )
                ],
                feedforward_channels=256*2,
                ffn_dropout=0.1,
                embed_dims=256,
                conv_num=2,
                operation_order=('cross_attn', 'norm',
                                    'ffn', 'norm', 'conv')
            ),
        ),
    ),
    
    camera_decoder=dict(
        type='MAESwinDecoder',
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        mlp_ratio=4,
        # decoder settings
        num_patches=(8, 22),
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        norm_pix_loss=True,
    ),

)


# optimizer
lr = 2.5e-4  # max learning rate
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-7)

momentum_config = None

# runtime settings
epochs = 200
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=epochs+1)  # Don't evaluate when doing pretraining
workflow = [("train", 1)]
checkpoint_config = dict(
    interval=5,
    max_keep_ckpts=5,
)

fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=12,
)
