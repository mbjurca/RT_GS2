# RT-GS2: Real-Time Generalizable Semantic Segmentation for 3D Gaussian Representations of Radiance Fields

This repository contains the official implementation of [RT-GS2](https://arxiv.org/abs/2405.18033).

[[ArXiv]](https://arxiv.org/abs/2405.18033) [[Project Page]](https://mbjurca.github.io/rt-gs2/)

![overview](https://github.com/mbjurca/RT_GS2/blob/main/arch_image/arch_overview.png)

## Installation

In order to create a working enviroment we provided an enviroment.yaml file

```
conda create -f enviroment.yaml
conda activate rtgs2
```

The enviroment also has to have the dependences of the [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) model that we included. Mainly the diff-gaussian-rasterization and simple-knn packages need to be build. Also addition some dependances for [AsymFormer](https://github.com/Fourier7754/AsymFormer), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) might be needed. For information on how to install those, we refer to those repositories.

## Data

Our experiments were conducted on three different datasets: [Replica](https://github.com/facebookresearch/Replica-Dataset) (an already processed version of the Replica dataset can also be found [here](https://www.dropbox.com/scl/fo/puh6djua6ewgs0afsswmz/AGudMbll0n0v_iADmqrrRds?rlkey=ep5495umv628y2sk8hvnh8msc&e=1&dl=0) ), [ScanNet](http://www.scan-net.org/), and [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/documentation). The data can be downloaded using the aforementioned links. We have also added some additional metadata for each dataset in `configs/dataset`.

Make sure you follow the Gaussian Splatting preprocessing to make sure the the images are in the right format to train the gaussian splatting models.

The directories for each dataset have to be organized in the following way (please make sure that also the included metadata is correctly organized as indicated below):

### Replica

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

- `root/`: The location of the data root directory. Make sure this corresponds with the hard-coded data path in the config files.
- `images/`: Contains the original sparse images of the scenes.
- `render/`: Contains the synthesized views of the scene.
- `semantic/`: Contains the semantic masks for each view.
- `point_cloud/`: Contains the point clouds from the 30,000th and 7,000th iterations returned by the Gaussian Splatting model. Iteration 1 contains the model with additional features to facilitate the training procedure.
- `input.ply`, `cameras.json`, `sparse/`, `cfg_args`: Files used for visualization and class mappings in the root directory.
- `color_mapping_remapped.yaml`, `color_mapping.yaml`, `id2name.yaml`: Files used for visualization and class mappings. Those files are provided in the dataset directory.

### ScanNet
```
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
├──── id2color.yaml/
├──── id2label.yaml/
├──── labelids.txt/
```

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
├──── scannetpp_id2color.yaml/
├──── scannetpp_id2label.yaml/
```

## Train RT-GS2

### Train Gaussian Splatting models

The first step of our pipeline involves the training of the gaussian models. This is done using the instructions in the original [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting), for each scene seperately. Each dataset has to be transformed in order to accomodate the requirements of the model. The resulting gaussian models have to be stored with respect to the mentioned exemplified organizations structure.

### Train View-independent 3D Gaussian feature learning (Self-Supervised Constrastive Learning)

1. As a first step for the 3D Gaussian feature learning, pairs of different views of the same scene have to be collected (in point cloud space). To do so, run the following command (e.g., for the Replica dataset, similar for the other datasets): 
   `python data/scripts/extract_contrastive_data_replica.py`
   The code in this script has hardcoded paths to the scenes and the output directory, so please change them according to your specific paths.

2. Once these pairs have been collected, train the Point Transformer V3 model by running:
   `python train_SSL.py --config configs/train_ssl_replica.yaml`

### Train View-Dependent / View-Independent (VDVI) feature fusion (Semantic Segmentation)

1. Create a temporary Gaussian Splatting model with the additional features:
   `python models/scripts/create_features_replica.py`

2. Render the views and features to make the training pipeline faster:
   `python models/scripts/render_geom_features_replica.py`

3. To train the final model, run:
   `python train_GGS.py --config configs/generalized_semantic_replica.yaml`

## Inference

After training both models, one can run the inference to obtain the real-time performance of the full model by running:

```
python infer_GGS.py --scene path/to/gaussian_scene/ --ssl_model_path path/to/point_cloud_model --semantic_model_path path/to/semantic_model
```

## Citation
If you find this work useful in your research, please cite:
```
@article{jurca2024rt,
  title={RT-GS2: Real-Time Generalizable Semantic Segmentation for 3D Gaussian Representations of Radiance Fields},
  author={Jurca, Mihnea-Bogdan and Royen, Remco and Giosan, Ion and Munteanu, Adrian},
  journal={arXiv preprint arXiv:2405.18033},
  year={2024}
}
```

## Acknowledgments

This repo was based on multiple works such as: [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping), [AsymFormer](https://github.com/Fourier7754/AsymFormer), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)
