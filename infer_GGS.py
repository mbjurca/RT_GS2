import torch 
from models.model import create_model_ggs, create_model_ssl
import argparse
import os
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import os
from models.gaussian_splatting.gaussian_renderer import GaussianModel
from models.gaussian_splatting.scene import Scene
from models.gaussian_splatting.gaussian_renderer import render


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


# inference script for a scene
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Path to the scene', default='')
    parser.add_argument('--ssl_model_path', type=str, help='Path to the SSL model', default='')
    parser.add_argument('--semantic_model_path', type=str, help='Path to the semantic model', default='')
    args = parser.parse_args()
    # set device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scene_dir = args.scene

    dataset = ModelParams(scene_dir)
    gaussians = GaussianModel(3)
    scene = Scene(dataset, gaussians, load_iteration=1, shuffle=False)
    pipeline = PipelineParams()
    views_train = scene.getTrainCameras()
    views_test = scene.getTestCameras()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

    num_classes = 46
    model_pc = create_model_ssl(None, 'ptv3', no_out_features=32)
    model_pc.to(device)
    model_sem = create_model_ggs(num_classes=num_classes)
    model_sem.to(device)

    model_state_dict = torch.load(args.semantic_model_path, map_location=device)
    model_sem.load_state_dict(model_state_dict['model_state_dict'])
    model_sem.eval()

    model_pc.load_state_dict(torch.load(args.ssl_model_path, map_location=device))
    model_pc.eval()
    
    for idx, view in enumerate(views_train):
        rendering = render(view, gaussians, pipeline, background, pc_features=gaussians.get_features_geom)
        rendered_features = rendering["render_object"].to(device)
        rendering = rendering["render"].to(device)
        pred, x_lin = model_sem(rendering.unsqueeze(0), rendered_features.unsqueeze(0))


if __name__ == '__main__':
    main()




