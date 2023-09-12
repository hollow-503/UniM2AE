# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-49.95, -49.95, -4.95, 49.95, 49.95, 2.95]
number_of_sweeps = 9  # Extra sweeps to be merged. Max is 9 extra.
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0,
        test_mode=True,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ImageAug3D', 
         final_dim=[256, 704],
         resize_lim=[0.44, 0.61],
         bot_pct_lim=[0.0, 0.0],
         rand_flip=True,
         is_train=True,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3Dv2', flip_ratio_bev_horizontal=0.5, sync_2d=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    # dict(type='ImageNormalize_mae', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego',
                    'lidar2camera', 'camera2lidar', 'lidar2image', 
                    'img_aug_matrix', 'lidar_aug_matrix'],
         )
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0,
        test_mode=True,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(256, 704),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='ImageAug3D', 
                final_dim=[256, 704],
                resize_lim=[0.48, 0.48],
                bot_pct_lim=[0.0, 0.0],
                rand_flip=False,
                is_train=False,
            ),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]
            ),
            dict(type='RandomFlip3Dv2', sync_2d=False),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False
            ),
            dict(
                type='Collect3D', 
                keys=['points', 'img'],
                meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego',
                           'lidar2camera', 'camera2lidar', 'lidar2image', 
                           'img_aug_matrix', 'lidar_aug_matrix']
            ),
        ]
    )
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0,
        test_mode=True,
        use_dim=[0,1,2,3,4]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=24, pipeline=eval_pipeline)
