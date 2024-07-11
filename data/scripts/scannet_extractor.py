import os
import shutil
import numpy as np
from PIL import Image

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def process_scene(scene_folder):
    # Paths for the subfolders
    color_folder = os.path.join(scene_folder, 'color')
    pose_folder = os.path.join(scene_folder, 'pose')
    intrinsic_folder = os.path.join(scene_folder, 'intrinsic')

    # Create GS and input folders
    gs_folder = os.path.join(scene_folder, 'GS')
    input_folder = os.path.join(gs_folder, 'images')
    os.makedirs(input_folder, exist_ok=True)

    # Copy images that are multiples of 10 and get image dimensions
    width, height = None, None
    for filename in os.listdir(color_folder):
        if filename.endswith('.jpg') and int(filename.split('.')[0]) % 7 == 0:
            shutil.copy(os.path.join(color_folder, filename), input_folder)
            if width is None or height is None:
                with Image.open(os.path.join(color_folder, filename)) as img:
                    width, height = img.size

    # Read intrinsic camera parameters
    with open(os.path.join(intrinsic_folder, 'intrinsic_color.txt'), 'r') as f:
        intrinsic_params = [float(x) for x in f.read().split()]
        fx = intrinsic_params[0]
        fy = intrinsic_params[5]
        cx = intrinsic_params[2]
        cy = intrinsic_params[6]

    # Prepare to write camera.txt, images.txt, and points3D.txt
    camera_txt = os.path.join(gs_folder, 'cameras.txt')
    images_txt = os.path.join(gs_folder, 'images.txt')
    points3d_txt = os.path.join(gs_folder, 'points3D.txt')

    with open(camera_txt, 'w') as camera_file, open(images_txt, 'w') as images_file:
        # Write camera parameters
        camera_file.write(f"# Camera list with one line of data per camera:\n")
        camera_file.write(f"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS\n")
        camera_file.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

        # Write image parameters
        images_file.write(f"# Image list with two lines of data per image:\n")
        images_file.write(f"#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write(f"#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i in range(0, len(os.listdir(color_folder)), 7):
            pose_file = os.path.join(pose_folder, f"{i}.txt")
            if os.path.exists(pose_file):
                # Read extrinsic parameters
                extrinsic = np.linalg.inv(np.loadtxt(pose_file).reshape((4, 4)))
                r = extrinsic[:3, :3]
                t = extrinsic[:3, 3]

                # Convert rotation matrix to quaternion
                q = rotmat2qvec(r)

                # Write to images.txt
                images_file.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {i}.jpg\n")
                images_file.write(f"\n")

    # Create an empty points3D.txt
    open(points3d_txt, 'w').close()

if __name__ == "__main__":
    scene_folder = "/home/mihnea/data/ScanNet/scans/scene0000_00"  # Replace with your scene folder path
    process_scene(scene_folder)
