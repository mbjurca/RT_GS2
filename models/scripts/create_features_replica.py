import torch
import numpy as np
from plyfile import PlyData, PlyElement
import sys
import os
from tqdm import tqdm

print(sys.path)
sys.path.append(f'{sys.path[0]}/../')
from PointTransformerV3 import PointTransformerV3

# sys.path.append('/home/mihnea/mihnea/gaussian-splatting')
# from models.gaussian_splatting.scene.cameras import Camera

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = torch.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = torch.tensor(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = torch.tensor(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = torch.tensor(plydata.elements[0]["f_dc_2"]) 

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    return xyz, features_dc, features_extra, scales, rots, opacities

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    for i in range(32):
        l.append('f_pc_{}'.format(i))
    return l

def save_ply(path, xyz, f_dc, f_rest, opacities, scale, rotation, feature_geom):

    normals = np.zeros_like(xyz)

    list_of_attributes = construct_list_of_attributes()

    dtype_full = [(attribute, 'f4') for attribute in list_of_attributes]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    f_dc = f_dc.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = np.ascontiguousarray(f_rest.transpose(0, 2, 1).reshape(-1, 45))

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, feature_geom), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def add_features_geom_and_save_ply(ply_file, features_geom, output_path):
    # Ensure features_geom is a numpy array (N, D) where N is number of points
    # D is the number of dimensions/features per point. For simplicity, assuming D=1 here.
    features_geom_np = features_geom.cpu().numpy()

    xyz, features_dc, features_extra, scales, rots, opacities = load_ply(ply_file)

    save_ply(output_path, xyz, features_dc, features_extra, opacities, scales, rots, features_geom_np)


def convert_sh2rgb(features_dc):
        features_dc = features_dc * 0.28209479177387814 + 0.5
        features_min = features_dc.min()
        features_max = features_dc.max()
        colors = (features_dc - features_min) / (features_max - features_min)

        return colors

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_features(model, gaussians_path, output_path):
     
    xyz, features_dc, features_extra, scales, rots, opacities = load_ply(gaussians_path)
    xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
    color = convert_sh2rgb(torch.tensor(features_dc, dtype=torch.float32)).squeeze(-1).cuda()
    scales = torch.tensor(scales, dtype=torch.float32).cuda()
    rots = torch.tensor(rots, dtype=torch.float32).cuda()
    opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float32)).cuda()
    input = torch.cat([xyz, color, opacities, scales], dim=1).type(torch.float32).cuda()


    with torch.no_grad():
        input_dict = {
                'feat' : input, 
                'coord' : input[..., :3],
                'grid_size' : 0.07, 
                'offset': torch.tensor([input.shape[0]]).cuda()
            }
        output = model(input_dict).feat
        add_features_geom_and_save_ply(gaussians_path, output, output_path)
        
    
    return 1
    
    
if __name__ == '__main__':

    weights_path = '/media/mihnea/0C722848722838BA/Replica2/experiments/contrastive_ptv3_clip1_adam_voxelized/model_weights/best_model_epoch_14_loss_2.5098.pth'
    scene_list = [
        # '/media/mihnea/0C722848722838BA/Replica3/office_3', 
        # '/media/mihnea/0C722848722838BA/Replica3/room_2'
        # '/media/mihnea/0C722848722838BA/Replica2/office_0',
        # '/media/mihnea/0C722848722838BA/Replica2/office_1', 
        # '/media/mihnea/0C722848722838BA/Replica2/office_2', 
        # '/media/mihnea/0C722848722838BA/Replica2/office_4',
        '/media/mihnea/0C722848722838BA/Replica2/room_0',
        # '/media/mihnea/0C722848722838BA/Replica2/room_1'
        ]

    model = PointTransformerV3(in_channels=10)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    model.cuda()

    for scene in tqdm(scene_list, desc='Processing scenes'):
        if not os.path.exists(os.path.join(scene, 'point_cloud/iteration_1')):
            os.mkdir(os.path.join(scene, 'point_cloud/iteration_1'))
        gs_model = os.path.join(scene, 'point_cloud/iteration_30000/point_cloud.ply')
        output_path = os.path.join(scene, 'point_cloud/iteration_1/point_cloud.ply')        
        status = create_features(model, gs_model, output_path)
