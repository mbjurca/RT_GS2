from yacs.config import CfgNode as CN
import os

_C = CN()

_C.NAME = ''
_C.TASK = ''

_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.ROOT_DIR = ""
_C.DATASET.ROOT = ""
_C.DATASET.SCENE = ""
_C.DATASET.NUM_CLASSES = 0
_C.DATASET.SCENE_LIST_TRAIN = []
_C.DATASET.SCENE_LIST_TEST = []
_C.DATASET.PAIR_DIR_TRAIN = ""
_C.DATASET.PAIR_DIR_TEST = ""
_C.DATASET.ID2NAME = ""
_C.DATASET.COLOR_MAPPING = ""
_C.DATASET.CLASS_WEIGHTS = ""

_C.MODEL = CN()
_C.MODEL.SEMANTIC_HEAD=''

_C.BACKBONE = CN()
_C.BACKBONE.USE_XYZ = 'true'
_C.BACKBONE.NUM_FEATURES = 3

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.OUTPUT_DIR = ''
_C.TRAIN.SAVE_PREDICTIONS = True
_C.TRAIN.PICK_MODEL = 'last'
_C.TRAIN.LR = 0.001
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.GRAD_CLIP_NORM = 2.
_C.TRAIN.IMG_SIZE = (420, 630)
_C.TRAIN.GPU = 1
_C.TRAIN.FINETUNE = False
_C.TRAIN.MODEL_PATH = ''
_C.TRAIN.SCENE_TO_FINETUNE = ''
_C.TRAIN.LOSS = ''  
_C.TRAIN.COMPUTE_SEMANTIC_WEIGHTS = False
_C.TRAIN.LOAD_PRETRAIN = False

_C.DATA=CN()
_C.DATA.IGNORE_INDEX=-1


_C.VAL = CN()
_C.VAL.ANNOTATIONS_FILE = ""
_C.VAL.QUESTIONS_FILE = ""
_C.VAL.BATCH_SIZE = 128

_C.POINTNETPP = CN()
_C.POINTNETPP.USE_XYZ = True

# def update_configs(cfg, model_file_path, model_backbone_file_path, data_file_path, dataset_file_path, training_file_path):
#     cfg.defrost()
#     cfg.merge_from_file(model_file_path)
#     cfg.merge_from_file(model_backbone_file_path)
#     cfg.merge_from_file(data_file_path)
#     cfg.merge_from_file(dataset_file_path)
#     cfg.merge_from_file(training_file_path)
#     cfg.freeze()

def update_configs(cfg, config_file_path):
    cfg.defrost()
    cfg.merge_from_file(config_file_path)
    cfg.freeze()

def update_configs_gen(cfg, model_file_path, data_file_path, dataset_file_path, training_file_path):
    cfg.defrost()
    cfg.merge_from_file(model_file_path)
    cfg.merge_from_file(data_file_path)
    cfg.merge_from_file(dataset_file_path)
    cfg.merge_from_file(training_file_path)
    cfg.freeze()

def get_configs():

    return _C.clone()