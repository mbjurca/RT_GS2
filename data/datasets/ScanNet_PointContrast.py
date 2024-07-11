import torch 
import os
from torch.utils import data
import numpy as np
import os
from plyfile import PlyData, PlyElement
import random
from scipy.linalg import expm, norm
import re


def construct_list_of_attributes(features_dc, features_rest, scaling, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(xyz, f_dc, f_rest, scale, rotation, opacities, path):
        

    normals = np.zeros_like(xyz)

    f_dc = f_dc.reshape(xyz.shape[0], -1)
    f_rest = f_rest.reshape(xyz.shape[0], -1)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc=f_dc, features_rest=f_rest, scaling=scale, rotation=rotation)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys
                    
def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):

    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count

class ScanNet_PointContrast(data.Dataset):

    def __init__(self, 
            root_dir,
            scene_list,
            pair_dir,
            device,
            stage
            ):
        
        super().__init__()
        
        self.scene_list = scene_list
        self.model_list = []
        for scene in scene_list:
            stage_dir = 'train' if stage == 'train' else 'test'
            model_path = os.path.join(root_dir, stage_dir, scene, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
            self.model_list.append(model_path)
        self.device = device
        self.pair_dir = pair_dir
        self.pairs_list = os.listdir(pair_dir)
        self.randg = np.random.RandomState()
        self.voxel_size = 0.07


    def __len__(self):
        return len(self.pairs_list)
    
    def __getitem__(self, index): 
 
        scene_idx = int(re.search(r'scene_(\d+)_view', self.pairs_list[index]).group(1))
        xyz, features_dc, features_extra, scales, rots, opacities = self.load_ply(self.model_list[scene_idx])

        file = self.pairs_list[index]

        mask_1 = np.load(os.path.join(self.pair_dir, file))['mask_1']
        mask_2 = np.load(os.path.join(self.pair_dir, file))['mask_2']

        positions_1 = np.full_like(mask_1, -1, dtype=int)  # Ensure dtype is int for positions_1
        indices_1 = np.arange(mask_1.shape[0])[mask_1]
        positions_1[indices_1] = np.arange(len(indices_1))

        positions_2 = np.full_like(mask_2, -1, dtype=int)
        indeces_2 = np.arange(mask_2.shape[0])[mask_2]
        positions_2[indeces_2]= np.arange(len(indeces_2))

        mask_voxelize = np.zeros_like(mask_1)
        mask_voxelize[voxelize(xyz)] = 1

        mask_1 = mask_1 & mask_voxelize
        mask_2 = mask_2 & mask_voxelize

        new_positions_1 = np.full_like(mask_1, -1,dtype=int)
        indeces_1 = np.arange(mask_1.shape[0])[mask_1==1]
        new_positions_1[indeces_1]= np.arange(len(indeces_1))

        new_positions_2 = np.full_like(mask_2, -1, dtype=int)  
        indeces_2 = np.arange(mask_2.shape[0])[mask_2==1]
        new_positions_2[indeces_2]= np.arange(len(indeces_2))

        correspondences = np.load(os.path.join(self.pair_dir, file))['correspondences']
        
        # compute new correspondences after voxelization 
        mask_idx_1 = np.isin(positions_1, correspondences[:, 0])
        mask_idx_2 = np.isin(positions_2, correspondences[:, 1])
        new_idx1_mapped = np.where(mask_idx_1, new_positions_1, -1)
        new_idx2_mapped = np.where(mask_idx_2, new_positions_2, -1)
        valid_mask = (new_idx1_mapped >= 0) & (new_idx2_mapped >= 0)
        new_correspondences = np.vstack([new_idx1_mapped[valid_mask], new_idx2_mapped[valid_mask]]).T

        xyz1, scales1 = self.scale_point_cloud(xyz[mask_1], scales[mask_1])
        xyz2, scales2 = self.scale_point_cloud(xyz[mask_2], scales[mask_2])        

        T0 = self.sample_random_trans(xyz1, self.randg)
        T1 = self.sample_random_trans(xyz2, self.randg)

        xyz1 = self.apply_transform(xyz1, T0)
        xyz2 = self.apply_transform(xyz2, T1)

        opacities1 = self.sigmoid(opacities[mask_1])
        opacities2 = self.sigmoid(opacities[mask_2])

        opacities1 = self.feature_jitter(opacities1)
        opacities2 = self.feature_jitter(opacities2)

        features_dc1 = features_dc[mask_1]
        features_dc2 = features_dc[mask_2]

        colors1 = self.convert_sh2rgb(features_dc1).squeeze(-1)
        colors2 = self.convert_sh2rgb(features_dc2).squeeze(-1)

        colors1 = self.feature_jitter(colors1)
        colors2 = self.feature_jitter(colors2)

        features1 = np.hstack([colors1, opacities1, scales1])
        features2 = np.hstack([colors2, opacities2, scales2])

        return xyz1, xyz2, features1, features2, new_correspondences
    
    def scale_point_cloud(self, pts, scales, min_scale=0.6, max_scale=1.2):
        scale = min_scale + (max_scale - min_scale) * random.random()
        pts = scale * pts
        scales = scale * scales
        
        return pts, scales

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def sample_random_trans(self, pcd, randg, rotation_range=360):
        T = np.eye(4)
        R = self.M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
        T[:3, :3] = R
        T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
        return T
    
    def convert_sh2rgb(self, features_dc):
        features_dc = features_dc * 0.28209479177387814 + 0.5
        features_min = features_dc.min()
        features_max = features_dc.max()
        colors = (features_dc - features_min) / (features_max - features_min)

        return colors
    
    def feature_jitter(self, colors, mu=0, sigma=0.02):
        noise = np.random.normal(mu, sigma, colors.shape)
        colors += noise
        colors = np.clip(colors, 0, 1)
        return colors
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def load_ply(self, path):
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
    