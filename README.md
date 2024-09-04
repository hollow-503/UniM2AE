# UniM<sup>2</sup>AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving

### [Paper](https://arxiv.org/abs/2308.10421) | [BibTeX](#citation)

This is the official PyTorch implementation of the paper - UniM<sup>2</sup>AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving.

![pipeline](Assets/pipeline.png)

## Results

### Pre-training

We provide our pretrained weights. You can load the pretrained UniM<sup>2</sup>AE(UniM<sup>2</sup>AE for BEVFusion and UniM<sup>2</sup>AE-sst-pre for SST) to train the multi-modal detector(BEVFusion) or the LiDAR-only detector(SST).

|        Model         | Modality | Checkpoint  |
| :------------------: | :------: | :---------: |
| UniM<sup>2</sup>AE | C+L | [Link](https://drive.google.com/file/d/1NaD0zNuxpIXwSyBfvuouTGsH97rJH6r9/view?usp=drive_link) |
| UniM<sup>2</sup>AE-sst-pre | L | [Link](https://drive.google.com/file/d/19FcDMD_GkS6gMVCUXO463maQlCJj_Li2/view?usp=drive_link) |
| swint-nuImages | C | [Link](https://bevfusion.mit.edu/files/pretrained_updated/swint-nuimages-pretrained.pth) |

*Note:* The checkpoint(denoted as swint-nuImages) pretrained on nuImages is provided by [BEVFusion](https://github.com/mit-han-lab/bevfusion).

### 3D Object Detection (on nuScenes validation)

|        Model         | Modality | mAP  | NDS  | Checkpoint  |
| :------------------: | :------: | :--: | :--: | :---------: |
| [TransFusion-L-SST](Finetune/bevfusion/configs/nuscenes/det/transfusion/secfpn/lidar/sstv2.yaml) | L | 65.0 | 69.9 | [Link](https://drive.google.com/file/d/1WWvSqdBbsVchn96fb3x_PvoaPCKHan8e/view?usp=drive_link) |
| [UniM<sup>2</sup>AE-L](Finetune/bevfusion/configs/nuscenes/det/transfusion/secfpn/lidar/sstv2.yaml) | L | 65.7 | 70.4 | [Link](https://drive.google.com/file/d/1UC18fgNv_LWSFSAhJVOOPnP29ibd-LzF/view?usp=drive_link) |
| [BEVFusion-SST](Finetune/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevfusion_sst.yaml) | C+L | 68.2 | 71.5 | [Link](https://drive.google.com/file/d/1RSGqknmqsnVUj1CgjmsRjKIKc1xSRMpg/view?usp=drive_link) |
| [UniM<sup>2</sup>AE](Finetune/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevfusion_sst.yaml) | C+L | 68.4 | 71.9 | [Link](https://drive.google.com/file/d/1_woCQ0ZC-DqIs50rTtlqiTUwGc-h_CM2/view?usp=drive_link) |
| [UniM<sup>2</sup>AE w/MMIM](Finetune/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/unim2ae_MMIM.yaml) | C+L | 69.7 | 72.7 | [Link](https://drive.google.com/file/d/1H2MEyvH7wJmA1p_wYJFr90sEU2DamqCp/view?usp=drive_link) |

### 3D Object Detection (on nuScenes test)

|   Model   | Modality | mAP  | NDS  |
| :-------: | :------: | :--: | :--: |
| UniM<sup>2</sup>AE-L | L | 67.9 | 72.2 |
| UniM<sup>2</sup>AE | C+L | 70.3 | 73.3 |

Here, we train the UniM<sup>2</sup>AE-L and the UniM<sup>2</sup>AE on the trainval split of the nuScenes dataset and test them without any test time augmentation.

### BEV Map Segmentation (on nuScenes validation)

|        Model         | Modality | mIoU | Checkpoint  |
| :------------------: | :------: | :--: | :---------: |
| [BEVFusion](Finetune/bevfusion/configs/nuscenes/seg/camera-bev256d2.yaml) | C | 51.2 | [Link](https://drive.google.com/file/d/1pF7Yp9JMnFKhLLCBSPsTBbJpnM5KrowQ/view?usp=drive_link) |
| [UniM<sup>2</sup>AE](Finetune/bevfusion/configs/nuscenes/seg/camera-bev256d2.yaml) | C | 52.9 | [Link](https://drive.google.com/file/d/1kLbMYTn3Q-z6K1PdWCEWwD8EToTvsvj1/view?usp=drive_link) |
| [BEVFusion-SST](Finetune/bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml) | C+L | 61.3 | [Link](https://drive.google.com/file/d/1myGNgEl19CoSBKaeR5xTSzNpO9KZhbQY/view?usp=drive_link) |
| [UniM<sup>2</sup>AE](Finetune/bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml) | C+L | 61.4 | [Link](https://drive.google.com/file/d/1g7P7YdaCPyWPSm-c4RTM1A170c3L_xS4/view?usp=drive_link) |
| [UniM<sup>2</sup>AE w/MMIM](Finetune/bevfusion/configs/nuscenes/seg/unim2ae_MMIM.yaml) | C+L | 67.8 | [Link](https://drive.google.com/file/d/1Su25lFLLgKf2zbDIeyTunmtsV7xn7lGg/view?usp=drive_link) |

## Prerequisites

### Pre-training

- Python == 3.8
- [mmcv-full](https://github.com/open-mmlab/mmcv) == 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.14.0
- [torch](https://github.com/pytorch/pytorch) == 1.9.1+cu111
- [torchvision](https://github.com/pytorch/pytorch) == 0.10.1+cu111
- numpy == 1.19.5
- matplotlib == 3.6.2
- pyquaternion == 0.9.9
- scikit-learn == 1.1.3
- setuptools == 59.5.0

After installing these dependencies, please run this command to install the codebase:

```bash
cd Pretrain
python setup.py develop
```

### Fine-tuning

The code of Fine-tuning are built with different libraries. Please refer to [BEVFusion](https://github.com/mit-han-lab/bevfusion) and [Voxel-MAE](https://github.com/georghess/voxel-mae).

## Data Preparation

We follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download the nuScenes dataset. Please remember to download both detection dataset and the map extension for BEV map segmentation. 

After downloading the nuScenes dataset, please preprocess the nuScenes dataset by:

```bash
cd Finetune/bevfusion/
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

and create the soft link in `Pretrain/data`, `Finetune/sst/data` with `ln -s`.

After data preparation, the directory structure is as follows:

```
UniM2AE
├──Finetune
│   ├──bevfusion
│   │   ├──tools
│   │   ├──configs
│   │   ├──data
│   │   │   ├── can_bus
│   │   │   │   ├── ...
│   │   │   ├──nuscenes
│   │   │   │   ├── maps
│   │   │   │   ├── samples
│   │   │   │   ├── sweeps
│   │   │   │   ├── v1.0-test
│   │   |   |   ├── v1.0-trainval
│   │   │   │   ├── nuscenes_database
│   │   │   │   ├── nuscenes_infos_train.pkl
│   │   │   │   ├── nuscenes_infos_val.pkl
│   │   │   │   ├── nuscenes_infos_test.pkl
│   │   │   │   ├── nuscenes_dbinfos_train.pkl
│   ├──sst
│   │   ├──data
│   │   │   ├──nuscenes
│   │   │   │   ├── ...
├──Pretrain
│   ├──mmdet3d
│   ├──tools
│   ├──configs
│   ├──data
│   │   ├── can_bus
│   │   │   ├── ...
│   │   ├──nuscenes
│   │   │   ├── ...
```

## Pre-training

### Training

Please run:

```bash
cd Pretrain
bash tools/dist_train.sh configs/unim2ae_mmim.py 8 
```

and run the script for fine-tuning:

```bash
cd Pretrain
python tools/convert.py --source work_dirs/unim2ae_mmim/epoch_200.pth --target ../Finetune/bevfusion/pretrained/unim2ae-pre.pth
```

### Visualization

To get the reconstruction results of the images and the LiDAR point cloud, please run:

```bash
cd Pretrain
python tools/test.py configs/unim2ae_mmim.py --checkpoint [pretrain checkpoint path] --show-pretrain --show-dir viz
```


## Fine-tuning

We provide instructions to finetune [BEVFusion](https://github.com/mit-han-lab/bevfusion) and [Voxel-MAE](https://github.com/georghess/voxel-mae).

### BEVFusion

#### Training

If you want to train the LiDAR-only UniM<sup>2</sup>AE-L for object detection, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/sstv2.yaml --load_from pretrained/unim2ae-lidar-only-pre.pth
```

---

For UniM<sup>2</sup>AE w/MMIM detection model, please run:

```bash
cd Finetune/bevfusion

python tools/convert.py --source [lidar-only UniM2AE-L checkpoint file path] --fuser pretrained/unim2ae-pre.pth --target pretrained/unim2ae-stage1.pth --stage2

torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/unim2ae_MMIM.yaml --load_from pretrained/unim2ae-stage1.pth
```

If you want to init the camera backbone with weight pretrained on nuImages, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/unim2ae_MMIM.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/unim2ae-stage1-L.pth
```

---

For UniM<sup>2</sup>AE detection model, please run:

```bash
cd Finetune/bevfusion

torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevfusion_sst.yaml --load_from pretrained/unim2ae-stage1.pth
```

If you want to init the camera backbone with weight pretrained on nuImages, please run:

```bash
cd Finetune/bevfusion

torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevfusion_sst.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/unim2ae-L-det.pth
```

*Note*: The `unim2ae-L.pth` is the training results of the LiDAR-only UniM<sup>2</sup>AE-L for object detection.

---

For camera-only UniM<sup>2</sup>AE segmentation model, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/camera-bev256d2.yaml --load_from pretrained/unim2ae-seg-c-pre.pth
```

---

For UniM<sup>2</sup>AE segmentation model, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/fusion-sst.yaml --load_from pretrained/unim2ae-pre.pth
```

If you want to init the camera backbone with weight pretrained on nuImages, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/fusion-sst.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/unim2ae-seg-pre.pth
```

---

For UniM<sup>2</sup>AE w/MMIM segmentation model, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/unim2ae_MMIM.yaml --load_from pretrained/unim2ae-pre.pth
```

If you want to init the camera backbone with weight pretrained on nuImages, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/unim2ae_MMIM.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/unim2ae-seg-pre.pth
```

#### Evaluation

Please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval [evaluation type]
```

For example, if you want to evaluate the detection model, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/unim2ae_MMIM.yaml pretrained/unim2ae-mmim-det.pth --eval bbox
```

If you want to evaluate the segmentation model, please run:

```bash
cd Finetune/bevfusion
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/seg/unim2ae_MMIM.yaml pretrained/unim2ae-mmim-seg.pth --eval map
```

### SST

#### Training

To train the LiDAR-only anchor-based detector, please run

```bash
cd Finetune/sst
bash tools/dist_train.sh configs/sst_refactor/sst_10sweeps_VS0.5_WS16_ED8_epochs288_intensity.py 8 --cfg-options 'load_from=pretrained/unim2ae-sst-pre.pth'
```

#### Evaluation

To evaluate the LiDAR-only anchor-based detector, please run

```bash
cd Finetune/sst
bash tools/dist_train.sh configs/sst_refactor/sst_10sweeps_VS0.5_WS16_ED8_epochs288_intensity.py [checkpoint file path] 8
```

## Acknowledgement
UniM<sup>2</sup>AE is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). This repository is also inspired by the following outstanding contributions to the open-source community: [3DETR](https://github.com/facebookresearch/3detr), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [DETR](https://github.com/facebookresearch/detr), [BEVFusion](https://github.com/mit-han-lab/bevfusion), [MAE](https://github.com/facebookresearch/mae), [Voxel-MAE](https://github.com/georghess/voxel-mae), [GreenMIM](https://github.com/LayneH/GreenMIM), [SST](https://github.com/tusen-ai/SST), [TransFusion](https://github.com/XuyangBai/TransFusion).

## Citation

If you find UniM<sup>2</sup>AE is helpful to your research, please consider citing our work:

```
@article{zou2023unim,
  title={UniM$^2$AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving},
  author={Zou, Jian and Huang, Tianyu and Yang, Guanglei and Guo, Zhenhua and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2308.10421},
  year={2023}
}
```