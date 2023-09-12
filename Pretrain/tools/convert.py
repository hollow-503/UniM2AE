import torch
import collections
import argparse

def convert(source_path, target_path):
    source = torch.load(source_path)
    state_dict = source['state_dict']
    convert = collections.OrderedDict()
    convert_lidar = collections.OrderedDict()
    convert_seg = collections.OrderedDict()
    convert_seg_c = collections.OrderedDict()
    convert_sst = collections.OrderedDict()
    for item in state_dict.keys():
        if item.startswith('backbone.block_list'):
            convert['encoders.lidar.backbone'+item[8:]] = state_dict[item]
            convert_lidar['encoders.lidar.backbone'+item[8:]] = state_dict[item]
            convert_seg['encoders.lidar.backbone'+item[8:]] = state_dict[item]
            convert_sst[item] = state_dict[item]
        elif item.startswith('voxel_encoder.'):
            convert['encoders.lidar.'+item] = state_dict[item]
            convert_lidar['encoders.lidar.'+item] = state_dict[item]
            convert_seg['encoders.lidar.'+item] = state_dict[item]
            convert_sst[item] = state_dict[item]
        elif item.startswith('camera_encoder.backbone.'):
            key = item[24:]
            if not key.startswith('encoder'):
                continue
            key = key[8:]
            if key.startswith('layers.'):
                key = 'stages.' + key[7:]
                if key.find('.attn.'):
                    key = key.replace('.attn.', '.attn.w_msa.')
            if key.startswith('patch_embed'):
                if key.endswith('proj.weight'):
                    key = key[:-11] + 'projection.weight'
                elif key.endswith('proj.bias'):
                    key = key[:-9] + 'projection.bias'
            if key[:-7].endswith('mlp.fc1'):
                key = key[:-14] + 'ffn.layers.0.0' + key[-7:]
            if key[:-7].endswith('mlp.fc2'):
                key = key[:-14] + 'ffn.layers.1' + key[-7:]
            if key[:-5].endswith('.mlp.fc1'):
                key = key[:-13] + '.ffn.layers.0.0' + key[-5:]
            if key[:-5].endswith('.mlp.fc2'):
                key = key[:-13] + '.ffn.layers.1' + key[-5:]
            convert['encoders.camera.backbone.'+key] = state_dict[item]
            convert_seg_c['encoders.camera.backbone.'+key] = state_dict[item]
        elif item.startswith('fusion_module.fusion_encoder.'):
            key = item[29:]
            convert['fuser.'+key] = state_dict[item]
            convert_seg['fuser.'+key] = state_dict[item]
            
    state_dict_convert = dict()
    state_dict_convert['state_dict'] = convert
    torch.save(state_dict_convert, target_path)
    
    state_dict_convert['state_dict'] = convert_lidar
    torch.save(state_dict_convert, target_path[:-8]+'-lidar-only-pre.pth')
    
    state_dict_convert['state_dict'] = convert_seg
    torch.save(state_dict_convert, target_path[:-8]+'-seg-pre.pth')
    
    state_dict_convert['state_dict'] = convert_seg_c
    torch.save(state_dict_convert, target_path[:-8]+'-seg-c-pre.pth')
    
    state_dict_convert['state_dict'] = convert_sst
    torch.save(state_dict_convert, target_path[:-8]+'-sst-pre.pth')
    
    return convert


def convert_stage2(stage1_path, fuser_path, target_path):
    stage1 = torch.load(stage1_path)
    fuser = torch.load(fuser_path)

    stage1_state = stage1['state_dict']
    stage1_state_L = stage1['state_dict'].copy()
    fuser_state = fuser['state_dict']

    for item in fuser_state.keys():
        if item.startswith('encoders.lidar.'):
            continue
        stage1_state[item] = fuser_state[item]
        if not item.startswith('encoders.camera.'):
            stage1_state_L[item] = fuser_state[item]
    
    stage2_pre_L = dict()
    stage2_pre_L['state_dict'] = stage1_state_L
    torch.save(stage2_pre_L, target_path[:-4]+'-L.pth')
    
    stage2_pre = dict()
    stage2_pre['state_dict'] = stage1_state
    torch.save(stage2_pre, target_path)
    
    return stage1_state


def parse_args():
    parser = argparse.ArgumentParser(description='Transform the weight for fine-tuning')
    parser.add_argument('--source', type=str, help='source pth path')
    parser.add_argument('--target', type=str, help='target pth path')
    parser.add_argument('--fuser', type=str, help='fusion module pth path')
    parser.add_argument('--stage2', action='store_true',  help='whether to fuse pre-trained mmim and stage1 LiDAR-only detector')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    
    source_path = args.source
    target_path = args.target
    fuser_path = args.fuser
    stage2 = args.stage2
    
    if stage2:
        converted_pth = convert_stage2(source_path, fuser_path, target_path)
    else:
        converted_pth = convert(source_path, target_path)
    
    for item in converted_pth.keys():
        print(item)
