import json
import numpy as np
import os

def generate_unique_colors(num_classes):
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 256, size=(num_classes, 3))  # Generate random colors
    return colors

def add_special_colors(colors):
    colors[-2] = [0, 0, 0]  # Black
    colors[-1] = [255, 255, 255]  # White
    return colors

def read_classes_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return len(data)

def create_color_mapping(json_file_path):
    num_classes = read_classes_from_json(json_file_path) + 2  # +2 for black and white
    colors = generate_unique_colors(num_classes)
    colors = add_special_colors(colors)
    # Convert NumPy integers to Python integers in the mapping
    class_color_mapping = {int(i): [int(color[0]), int(color[1]), int(color[2])] for i, color in enumerate(colors[:-2])}
    class_color_mapping[-1] = [int(colors[-2][0]), int(colors[-2][1]), int(colors[-2][2])]  # Black
    class_color_mapping[-100] = [int(colors[-1][0]), int(colors[-1][1]), int(colors[-1][2])]  # White
    return class_color_mapping


def save_color_mapping_to_json(color_mapping, input_json_file_path):
    output_file_path = os.path.join(os.path.dirname(input_json_file_path), 'semantic_color_mapping.json')
    with open(output_file_path, 'w') as file:
        json.dump(color_mapping, file, indent=4)

# Example usage
json_file_path = '/media/mihnea/0C722848722838BA/ScanNet++2/metadata/semantic_map.json'
color_mapping = create_color_mapping(json_file_path)
save_color_mapping_to_json(color_mapping, json_file_path)
