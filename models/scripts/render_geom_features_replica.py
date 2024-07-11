import torch
import numpy as np
from plyfile import PlyData
import sys
import cv2
sys.path.append(f'{sys.path[0]}/../../')
from models.gaussian_splatting.gaussian_renderer import render
from models.gaussian_splatting.gaussian_renderer import GaussianModel
from models.gaussian_splatting.scene import Scene
from sklearn.decomposition import PCA
import os
from tqdm import tqdm 

class ModelParams():

    def __init__(self, scene_dir):

        self.sh_degree = 3
        self.source_path = scene_dir
        self.model_path = scene_dir
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = True

class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def features_to_rgb(features_list):
    # Input features shape: (16, H, W)

    features = torch.cat(features_list, dim=1)
    # Reshape features for PCA
    N, H, W = features.shape[1], features.shape[2], features.shape[3]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())
    
    print(pca_result.shape)
    # # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(N, H, W, 3)

    # # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')


    cv2.imwrite("features_399.png", rgb_array[0])
    cv2.imwrite("features_396.png", rgb_array[1])
    cv2.imwrite("features_400.png", rgb_array[2])

    # return rgb_array


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    torch.cuda.set_device('cuda:0')
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    scene_list = [
        # '/media/mihnea/0C722848722838BA/Replica3/office_3', 
        #'/media/mihnea/0C722848722838BA/Replica2/room_2'
        # '/media/mihnea/0C722848722838BA/Replica2/office_0',
        # '/media/mihnea/0C722848722838BA/Replica2/office_1', 
        # '/media/mihnea/0C722848722838BA/Replica2/office_2', 
        # '/media/mihnea/0C722848722838BA/Replica2/office_4',
        '/media/mihnea/0C722848722838BA/Replica3/room_0',
        # '/media/mihnea/0C722848722838BA/Replica2/room_1'
        ]
    
    for scene_dir in tqdm(scene_list, desc='Processing scenes'):

        dataset = ModelParams(scene_dir)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=1, shuffle=False)
        pipeline = PipelineParams()
        
        views = scene.getTrainCameras() + scene.getTestCameras()

        os.makedirs(os.path.join(scene_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(scene_dir, "render"), exist_ok=True)

        for view in views:

            rendering = render(view, gaussians, pipeline, background, pc_features=gaussians.get_features_geom)
            features = rendering["render_object"].detach().cpu()[:, np.newaxis, :, :]
            np.save(os.path.join(scene_dir, "features", view.image_name + ".npy"), features)

            rendering_color = rendering["render"].permute(1, 2, 0).detach().cpu().numpy()
            rendering_color_uint8 = (rendering_color * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(scene_dir, "render", view.image_name + ".png"), rendering_color_uint8[:, :, [2, 1, 0]])