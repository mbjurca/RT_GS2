import os
import shutil
import glob

# Define the base path to your dataset folder
base_path = "/media/mihnea/0C722848722838BA/Replica2"\

scene_list = ['/media/mihnea/0C722848722838BA/Replica2/office_0',
        '/media/mihnea/0C722848722838BA/Replica2/office_1', 
        '/media/mihnea/0C722848722838BA/Replica2/office_2', 
        '/media/mihnea/0C722848722838BA/Replica2/office_4',
        '/media/mihnea/0C722848722838BA/Replica2/room_0',
        '/media/mihnea/0C722848722838BA/Replica2/room_1', 
        '/media/mihnea/0C722848722838BA/Replica2/office_3', 
        '/media/mihnea/0C722848722838BA/Replica2/room_2']

# Iterate over each scene folder
for scene_path in scene_list:

    # Create target folders for images, semantic, and depth
    # target_folders = {'rgb': 'images', 'depth': 'depth', 'semantic_class': 'semantic'}
    target_folders = {'semantic_class': 'semantic'}
    for target_folder in target_folders.values():

        if os.path.exists(os.path.join(scene_path, target_folder)):
            shutil.rmtree(os.path.join(scene_path, target_folder))
        
        os.makedirs(os.path.join(scene_path, target_folder), exist_ok=True)

        if os.path.exists(os.path.join(scene_path, 'semantic_color')):
            shutil.rmtree(os.path.join(scene_path, 'semantic_color')) 
                    
        os.makedirs(os.path.join(scene_path, 'semantic_color'), exist_ok=True)
    
    for sequence in ['Sequence_1', 'Sequence_2']:
        for data_type in ['semantic_class']:
            source_folder = os.path.join(scene_path, sequence, data_type)
            
            # Filter to select every third image
            images = sorted(glob.glob(os.path.join(source_folder, "*.png")))
            selected_images = images[::3]  # Select every third image
            
            for image_path in selected_images:
                
                image_name = os.path.basename(image_path)
                if image_name.startswith('vis'):
                    target_folder = 'semantic_color'
                    sequence_number = '1' if sequence == 'Sequence_1' else '2'
                    
                    # Construct new image name based on the sequence and original image name
                    image_index = image_name.split('_')[-1]
                    new_image_name = f"{int(image_index.split('.')[0])}_{2 if sequence_number == '2' else 1}.png"
                    
                    # Define target path based on the type of data
                    target_path = os.path.join(scene_path, target_folder, new_image_name)
                    
                    # Copy the selected image to the target folder
                    shutil.copyfile(image_path, target_path)
                    print(f"Copied {image_path} to {target_path}")
                else:
                    sequence_number = '1' if sequence == 'Sequence_1' else '2'
                    
                    # Construct new image name based on the sequence and original image name
                    image_index = image_name.split('_')[-1]
                    new_image_name = f"{int(image_index.split('.')[0])}_{2 if sequence_number == '2' else 1}.png"
                    
                    # Define target path based on the type of data
                    target_folder = target_folders[data_type]
                    target_path = os.path.join(scene_path, target_folder, new_image_name)
                    
                    # Copy the selected image to the target folder
                    shutil.copyfile(image_path, target_path)
                    print(f"Copied {image_path} to {target_path}")

print("Dataset organization complete.")
