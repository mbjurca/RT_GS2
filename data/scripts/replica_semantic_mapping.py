import cv2
import numpy as np
import os
from tqdm import tqdm
import json
import yaml
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt


scene_list = ['/media/mihnea/0C722848722838BA/Replica2/office_0',
        '/media/mihnea/0C722848722838BA/Replica2/office_1', 
        '/media/mihnea/0C722848722838BA/Replica2/office_2', 
        '/media/mihnea/0C722848722838BA/Replica2/office_4',
        '/media/mihnea/0C722848722838BA/Replica2/room_0',
        '/media/mihnea/0C722848722838BA/Replica2/room_1', 
        '/media/mihnea/0C722848722838BA/Replica2/office_3', 
        '/media/mihnea/0C722848722838BA/Replica2/room_2'
        ]

label_file = '/media/mihnea/0C722848722838BA/Replica2/semantic_info/office_0/info_semantic.json'

def visualize_semantic_mapping(scene_path='/media/mihnea/0C722848722838BA/Replica2/office_0'):
    # Load id2name from YAML
    with open('id2name.yaml', 'r') as file:
        id2name = yaml.safe_load(file)

    # Load color_mapping from YAML
    with open('color_mapping.yaml', 'r') as file:
        color_mapping = yaml.safe_load(file)

    # Read semantic and semantic_color images
    semantic_image_list = os.listdir(os.path.join(scene_path, 'semantic'))
    semantic_color_list = os.listdir(os.path.join(scene_path, 'semantic_color'))

    for idx, semantic_image in enumerate(semantic_image_list):
        semantic = cv2.imread(os.path.join(scene_path, 'semantic', semantic_image), cv2.IMREAD_UNCHANGED)
        color = cv2.imread(os.path.join(scene_path, 'semantic_color', semantic_color_list[idx]), cv2.IMREAD_UNCHANGED)

        # Create colored semantic image using color_mapping
        colored_semantic = np.zeros_like(color)
        unique_labels = np.unique(semantic)  # Find unique labels in the semantic image
        for label in unique_labels:
            if label in color_mapping:  # Check if label is in color_mapping
                rgb = color_mapping[label]
                colored_semantic[np.where(semantic == label)] = rgb

        # Setup figure and subplots
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax_legend = fig.add_subplot(1, 3, 3)  # Dedicated axis for the legend

        # Display images
        ax1.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        ax1.set_title('Semantic Color')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(colored_semantic, cv2.COLOR_BGR2RGB))
        ax2.set_title('Colored Semantic')
        ax2.axis('off')

        # Create legend with class names for labels present in the image
        present_labels = [label for label in unique_labels if label in color_mapping]
        legend_labels = [id2name[label] for label in present_labels]
        legend_colors = [color_mapping[label] for label in present_labels]
        legend_colors_normalized = [(b/255, g/255, r/255) for r, g, b in legend_colors]

        # Generate legend patches
        legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in legend_colors_normalized]
        
        # Use the dedicated axis for the legend
        for patch, label in zip(legend_patches, legend_labels):
            ax_legend.add_patch(patch)
            patch.set_clip_on(False)  # Ensure patches are visible
        ax_legend.legend(legend_patches, legend_labels, loc='upper left')
        ax_legend.axis('off')

        plt.tight_layout()
        plt.show()


if __name__=='__main__':
    # labels = []
    # color_mapping = {}
    # id2name = {0: 'unlabled'}

    # # read json file
    # with open(label_file, 'r') as file:
    #     data = json.load(file)

    # for cls in data['classes']:
    #     id2name[cls['id']] = cls['name']

    # for scene in tqdm(scene_list, desc='Scenes', total=len(scene_list)):

    #     semantic_image_list = os.listdir(os.path.join(scene, 'semantic'))
    #     semantic_color_list = os.listdir(os.path.join(scene, 'semantic_color'))

    #     for idx, semantic_image in tqdm(enumerate(semantic_image_list), desc='Images', total=len(semantic_image_list)):

    #         semantic = cv2.imread(os.path.join(scene, 'semantic', semantic_image), cv2.IMREAD_UNCHANGED)
    #         color = cv2.imread(os.path.join(scene, 'semantic_color', semantic_color_list[idx]), cv2.IMREAD_UNCHANGED)
    #         unique_labels = np.unique(semantic)
    #         labels.extend(label for label in unique_labels if label not in labels)

    #         # Get corresponding color for each label
    #         for label in unique_labels:
    #             if label in id2name:
    #                 x, y = np.where(semantic == label)
    #                 color_label = color[x[0], y[0]]
    #                 color_mapping[int(label)] = [int(color_label[0]), int(color_label[1]), int(color_label[2])]

    # # Save id2name as YAML
    # with open('id2name.yaml', 'w') as file:
    #     yaml.dump(id2name, file)

    # Save color_mapping as YAML
    # with open('color_mapping.yaml', 'w') as file:
    #     yaml.dump(color_mapping, file)
    

    visualize_semantic_mapping()


            
