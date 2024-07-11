import torch 
import os
import sys
from tqdm import tqdm 
from torch.utils import data
import numpy as np
import cv2
import os
import torchvision.transforms as T
from matplotlib import pyplot as plt
import sys
from utils.dataset_utils import ToTensor, ColorJitter, RandomHorizontalFlip, RandomZoomIn, RandomPerspective, RandomCutOut, Normalize
import yaml 
import cv2
import os
import numpy as np


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
        self.debug = True

class ScanNetpp_Semantic(data.Dataset):
    def __init__(self, root_dir, scene_names, id2name, color_mapping, class_weights=None, transforms=None, stage='train', split='train', task='semantic', finetune=False, compute_semantic_weights=False):
        
        scene_split_dir = os.path.join(root_dir, split)
        if finetune:
            scene_list = scene_names
        else:
            scene_list = [os.path.join(scene_split_dir, scene) for scene in os.listdir(scene_split_dir) if os.path.isdir(os.path.join(scene_split_dir, scene))]

        self.scene_list = scene_list
        self.stage = stage
        self.task = task
        self.finetune = finetune

        self.list_instances = []
        self.load_instances(scene_list)

        with open(id2name, 'r') as file:
            self.id2name = yaml.safe_load(file)  

        # Load color_mapping from YAML
        with open(color_mapping, 'r') as file:
            self.color_mapping = yaml.safe_load(file)

        self.class_mapping = {}
        self.inv_class_mapping = {}

        idx = 0
        for class_id in self.color_mapping.keys():
            if class_id != 13 and class_id != 0:
                self.class_mapping[class_id] = idx
                self.inv_class_mapping[idx] = class_id
                idx += 1

        self.no_classses = len(self.color_mapping.keys()) - 2 # map office chair and chair to the same class and unlabled ignore

        if task == 'depth' or 'multi':
            p_perspective = 0
        else:
            p_perspective = 0.3

        self.transforms_test = T.Compose([ToTensor(), Normalize()])

        if finetune:
            self.transforms_train = T.Compose([ToTensor(), 
                                            RandomZoomIn(p=0.3),
                                            Normalize()])
        else:
            self.transforms_train = T.Compose([ToTensor(), 
                                            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                                            RandomHorizontalFlip(p=0.3),
                                            RandomCutOut(p=0.3),
                                            RandomZoomIn(p=0.3),
                                            RandomPerspective(p=p_perspective), 
                                            Normalize()])
            
        if compute_semantic_weights:
            class_weights = self.compute_semantic_weights(scene_list[0])
            self.semantic_weights = torch.zeros(self.no_classses)
            for class_id in class_weights.keys():
                if class_id in self.class_mapping.keys():
                    self.semantic_weights[self.class_mapping[class_id]] = class_weights[class_id]
        
    def compute_semantic_weights(self, scene):
        class_counts = {}
        total_pixels = 0

        # Iterate through each scene directory
        images_names = os.listdir(os.path.join(scene, 'semantic_2'))
        
        # Iterate through each image within the scene's 'semantic' subdirectory
        for image_name in images_names:
            img_path = os.path.join(scene, 'semantic_2', image_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                continue
            
            # Count occurrences of each class (unique label) in the image
            unique_classes, counts = np.unique(img, return_counts=True)
            total_pixels += np.sum(counts)
            
            # Accumulate counts for each class across all images
            for class_id, count in zip(unique_classes, counts):
                if int(class_id) in class_counts:
                    class_counts[class_id] += count
                else:
                    class_counts[class_id] = count

        # Compute weights for each class based on the inverse frequency
        class_weights = {}
        for class_id, count in class_counts.items():
            class_weight = total_pixels / (count * len(np.unique(list(class_counts.keys()))))
            class_weights[class_id] = class_weight

        # Optionally, normalize weights such that the smallest weight is 1.0
        min_weight = min(class_weights.values())
        class_weights = {class_id: weight / min_weight for class_id, weight in class_weights.items()}   

        return class_weights


    def map_id(self, id):
        return self.class_mapping.get(id, 0)
    
    def map_inv_id(self, id):
        return self.inv_class_mapping.get(id)
    
    def map_id2color(self, id):
        return self.color_mapping.get(id)

    def load_instances(self, scene_list):

        for scene in scene_list:
            names = os.listdir(os.path.join(scene, 'render'))
            names = [name.replace('.png', '') for name in names]
            if self.stage == 'train':
                self.list_instances.extend([(scene, name) for idx, name in enumerate(names) if idx % 8 != 0])
            elif self.stage == 'test':
                self.list_instances.extend([(scene, name) for idx, name in enumerate(names) if idx % 8 == 0])
            else:
                self.list_instances.extend([(scene, name) for name in names])

    def compute_mean_std_rendered_features(self):
        sum_ = np.zeros(32)
        sum_of_squares = np.zeros(32)
        total_elements = 0  # To keep track of the total number of elements per feature
        progress_bar = tqdm(self.list_instances, desc="Computing mean and std for rendered features", leave=False)
        for scene, name in progress_bar:
            feature_path = os.path.join(scene, 'features', name + '.npy')
            rendered_features = np.load(feature_path)

            # Update total count
            total_elements += rendered_features.shape[1] * rendered_features.shape[2]
            
            # Update sums and sum of squares
            sum_ += np.sum(rendered_features, axis=(1, 2))
            sum_of_squares += np.sum(rendered_features ** 2, axis=(1, 2))

        # Compute mean and variance (hence standard deviation)
        mean = sum_ / total_elements
        # Variance is E[X^2] - (E[X])^2
        variance = (sum_of_squares / total_elements) - (mean ** 2)
        std = np.sqrt(variance)

        return mean, std


    def __getitem__(self, index):

        name = self.list_instances[index]

        original_image_path = os.path.join(name[0], 'images', name[1] + '.JPG')
        render_path = os.path.join(name[0], 'render', name[1] + '.png')
        feature_path = os.path.join(name[0], 'features', name[1] + '.npy')
        semantic_path = os.path.join(name[0], 'semantic_2', name[1] + '.png')

        rendering = cv2.imread(render_path)
        rendering = cv2.resize(rendering, (640, 480))

        original_image = cv2.imread(original_image_path)
        original_image = cv2.resize(original_image, (640, 480))

        rendered_features = np.load(feature_path)

        semantic_mask = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
        semantic_mask = cv2.resize(semantic_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
        semantic_mask[semantic_mask == 13] = 11 # map chair office to chair
        semantic_mask[semantic_mask == 0] = -1 # unlabeled pixels to ignore
        semantic_mask = np.vectorize(self.map_id)(semantic_mask)

        sample = {'rendering': rendering, 'rendered_features': rendered_features, 'depth': [], 'semantic_mask': semantic_mask}

        if self.stage == 'train':
            sample = self.transforms_train(sample)
        else:
            sample = self.transforms_test(sample)
        
        return sample['rendering'], sample['rendered_features'], sample['semantic_mask'], sample['depth'], original_image, name[0], name[1]
        
    
    def display_sample(self, sample):
        # Assuming 'rendering' is a torch tensor containing values from 0 to 1
        rendering = sample['rendering'].permute(1, 2, 0)
        semantic_mask = torch.clip(sample['semantic_mask'], 0)

        # Convert the torch tensor to a numpy array
        rendering_np = rendering.numpy()[..., [2, 1, 0]]

        H, W, _ = rendering_np.shape
        colored_labels = np.zeros((H, W, 3), dtype=np.uint8)

        # Iterate over unique labels in pred
        for label in np.unique(semantic_mask):
            # Find the color corresponding to the current label
            color = self.map_id2color(label)
            # Apply this color to all positions of this label in the original image
            colored_labels[semantic_mask == label] = color

        colored_labels = cv2.cvtColor(colored_labels, cv2.COLOR_BGR2RGB)

        # Create a figure and axis
        fig, ax = plt.subplots(1, 2)

        # Display the rendering
        ax[0].imshow(rendering_np)
        ax[0].set_title('Rendering')

        # Display the colored labels
        ax[1].imshow(colored_labels)
        ax[1].set_title('Colored Labels')

        # Set the title and labels
        fig.suptitle('Rendering with Colored Labels')
        for a in ax:
            a.set_xlabel('X')
            a.set_ylabel('Y')


            
    def __len__(self):

        return len(self.list_instances)



if __name__ == '__main__':
    
    

    id2name = '/media/mihnea/0C722848722838BA/ScanNet++2/metadata/scannetpp_id2label.yaml'
    color_mapping = '/media/mihnea/0C722848722838BA/ScanNet++2/metadata/scannetpp_id2color.yaml'

    scene_names = ['1366d5ae89', '1a130d092a', '1a8e0d78c0', '281bc17764', '28a9ee4557', '2a496183e1', '39f36da05b', '4ba22fa7e4', '6d89a7320d', 'acd69a1746']

    test = ScanNetpp_Semantic(root_dir='/media/mihnea/0C722848722838BA/ScanNet++2/final_split', 
                            scene_names=scene_names,
                            id2name=id2name, 
                            color_mapping=color_mapping, 
                            stage='test', 
                            split='test',
                            task='semantic', 
                            finetune=False)

    rendering, rendered_features, semantic_mask, depth, name = test.__getitem__(30)
    color_mapping = test.color_mapping
    class_mapping = test.class_mapping
    def display_rendering_and_mask(rendering, semantic_mask, color_mapping, class_mapping):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display rendering
        rendering_np = rendering.permute(1, 2, 0).numpy()
        axes[0].imshow(rendering_np)
        axes[0].set_title('Rendering')
        axes[0].axis('off')
        
        # Display semantic mask
        semantic_mask_rgb = np.zeros_like(rendering_np)
        for class_id, color in color_mapping.items():
            if class_id != 13:
                print(np.unique(semantic_mask))
                class_mask = semantic_mask == class_mapping[class_id]
                color_scaled = [c / 255. for c in color]  # Scale color values to 0-255 range
                semantic_mask_rgb[class_mask] = color_scaled
        axes[1].imshow(semantic_mask_rgb)
        axes[1].set_title('Semantic Mask (Color)')
        axes[1].axis('off')
        
        # Display grayscale semantic mask
        axes[2].imshow(semantic_mask, cmap='gray')
        axes[2].set_title('Semantic Mask (Grayscale)')
        axes[2].axis('off')
        
        plt.show()

    # Call the function to display the rendering and semantic mask
    display_rendering_and_mask(rendering, semantic_mask, color_mapping, class_mapping)
