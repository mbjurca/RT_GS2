# RT_GS2

This repository contains the implementation of RT-GS2.

## Installation

Instructions for installation will be provided here.

## Data

Our experiments were conducted on three different datasets: Replica, ScanNet, and ScanNet++. Please download the data from their respective repositories and cite the original works.

### Replica

Train a Gaussian Splatting model according to the Gaussian Splatting documentation.

The directory structure should be organized as follows:
```
root/
├── scene1/
│   ├── images/
│   ├── features/
│   ├── point_cloud/
│   ├── render/
│   ├── semantic/
│   ├── input.ply
│   ├── cameras.json
│   ├── sparse/
│   └── cfg_args  
├── scene2/
│   ├── images/
│   ├── features/
│   ├── point_cloud/
│   ├── render/
│   ├── semantic/
│   ├── input.ply
│   ├
├── color_mapping_remaped.yaml
├── color_mapping.yaml
├── id2name.yaml
```

- `images/`: Contains the original sparse images of the scenes.
- `render/`: Contains the synthesized views of the scene.
- `semantic/`: Contains the semantic masks for each view.
- `point_cloud/`: Contains the point clouds from the 30,000th and 7,000th iterations returned by the Gaussian Splatting model. Iteration 1 contains the model with additional features to facilitate the training procedure.
- `input.ply`, `cameras.json`, `sparse/`, `cfg_args`: Files used for visualization and class mappings in the root directory.
- `color_mapping_remapped.yaml`, `color_mapping.yaml`, `id2name.yaml`: Files used for visualization and class mappings. Those files are provided in the dataset directory.
### ScanNet++
```
final_split/
├── train/
├──── scene1/
│    ├── images/
│    ├── features/
│    ├── point_cloud/
│    ├── render/
│    ├── semantic/
│    ├── input.ply
│    ├── cameras.json
│    ├── sparse/
│    └── cfg_args
├──── .....
├── test/
├──── scene1/
│    ├── images/
│    ├── features/
│    ├── point_cloud/
│    ├── render/
│    ├── semantic/
│    ├── input.ply
│    ├── cameras.json
│    ├── sparse/
│    └── cfg_args
├──── .....
metadata/
├── scannetpp_id2color.yaml/
├── scannetpp_id2label.yaml/
```
In order to get the data visit ScanNetpp[https://kaldir.vc.in.tum.de/scannetpp/documentation]. All the files of the matadata are available from the original repo we added our setup in the ```dataset folder```. Make sure you convert the images in the right format in order to train the gaussian splatting models. 

### ScanNet
root/
├── train/
├──── scene1/
│    ├── images/
│    ├── features/
│    ├── point_cloud/
│    ├── render/
│    ├── semantic/
│    ├── input.ply
│    ├── cameras.json
│    ├── sparse/
│    └── cfg_args
├──── .....
├── test/
├──── scene1/
│    ├── images/
│    ├── features/
│    ├── point_cloud/
│    ├── render/
│    ├── semantic/
│    ├── input.ply
│    ├── cameras.json
│    ├── sparse/
│    └── cfg_args
├──── .....
├── metadata/
├── id2color.yaml/
├── id2label.yaml/
├── labelids.txt/

```

## Run Method

### Train Gaussian Splatting models

The first step of our pipeline involves training the gaussian models separetely for this make sure you thech the original repo [https://github.com/graphdeco-inria/gaussian-splatting]. Each dataset has to be transformed in order to acomodae the requiremetns of the model. 

### Self-Supervised Constructive Learning

1. To extract pairs of views (point-cloud space), run the following command (e.g., for the Replica dataset): 
   `python data/scripts/extract_contrastive_data_replica.py`
   The code has hardcoded paths to the scenes and the output directory, so change them according to your specific paths.

2. Train the Point Transformer V3 model by running:
   `python train_SSL.py --config configs/train_ssl_replica.yaml`

### Semantic-Segmentation

1. Create a temporary Gaussian Splatting model with the additional features:
   `python models/scripts/create_features_replica.py`

2. Render the views and features to make the training pipeline faster:
   `python models/scripts/render_geom_features_replica.py`

3. To train the final model, run:
   `python train_GGS.py --config configs/generalized_semantic_replica.yaml`

### Acknolegments

This repo was based on multiple works such as: [https://github.com/graphdeco-inria/gaussian-splatting], [https://github.com/lkeab/gaussian-grouping], https://github.com/Fourier7754/AsymFormer, https://github.com/Pointcept/PointTransformerV3
