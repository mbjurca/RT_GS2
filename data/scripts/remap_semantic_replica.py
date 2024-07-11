import os
import numpy as np
import cv2
from tqdm import tqdm
import yaml

scene_list = ['/media/mihnea/0C722848722838BA/Replica2/office_0',
        '/media/mihnea/0C722848722838BA/Replica2/office_1', 
        '/media/mihnea/0C722848722838BA/Replica2/office_2', 
        '/media/mihnea/0C722848722838BA/Replica2/office_4',
        '/media/mihnea/0C722848722838BA/Replica2/room_0',
        '/media/mihnea/0C722848722838BA/Replica2/room_1', 
        '/media/mihnea/0C722848722838BA/Replica2/office_3', 
        '/media/mihnea/0C722848722838BA/Replica2/room_2']

color_mapping_path = '/media/mihnea/0C722848722838BA/Replica2/color_mapping.yaml'

common_classes = np.array([0, 3, 7, 8, 10, 11, 12, 13, 14, 17, 18, 19, 20, 22, 23, 26, 29, 31, 34, 35, 37, 40, 44, 47, 52, 54, 56, 59, 60, 61, 63, 64, 65, 76, 78, 79, 80, 82, 83, 87, 91, 92, 93, 95, 97, 98])
# You can also use a set for faster membership testing: common_classes_set = set(common_classes)

def remap_semantic(scene):
    os.makedirs(scene + '/semantic_remap', exist_ok=True)
    for file in tqdm(os.listdir(scene + '/semantic'), desc='images'):
        semantic_path = os.path.join(scene, 'semantic', file)
        semantic = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
        if semantic is None:
            continue
        
        # Create a mask where each pixel is True if it's in common_classes
        mask = np.isin(semantic, common_classes)
        
        # Use the mask to create the remapped semantic image
        semantic_remap = np.where(mask, semantic, 0)
        
        # Save the remapped image
        remapped_path = os.path.join(scene, 'semantic_remap', file)
        cv2.imwrite(remapped_path, semantic_remap)

def remap_color_mapping(color_mapping_path):
    # Load color_mapping from YAML
    with open(color_mapping_path, 'r') as file:
        color_mapping = yaml.safe_load(file)

    # Remove keys not in common_classes values
    color_mapping_updated = {key: value for key, value in color_mapping.items() if int(key) in common_classes}

    # Save the updated color_mapping to a new file
    updated_color_mapping_path = color_mapping_path.replace('.yaml', '_remaped.yaml')
    with open(updated_color_mapping_path, 'w') as file:
        yaml.dump(color_mapping_updated, file)

    print(len(color_mapping_updated.keys()))

def main(scene_list):
    for scene in tqdm(scene_list, desc='scenes'):
        remap_semantic(scene)


def main():

    remap_color_mapping(color_mapping_path)

    # for scene in tqdm(scene_list, desc='scenes'):

    #     remap_semantic(scene)


if __name__ == '__main__':
    main()