import torch 
from torch import nn
from torch.utils.data import DataLoader
from models.model import create_model_ggs
from data.datasets.Replica_Semantic import Replica_Semantic
from data.datasets.ScanNetpp_Semantic import ScanNetpp_Semantic
from data.datasets.ScanNet_Semantic import ScanNet_Semantic
import argparse
import os
import argparse
from configs.configs import update_configs, get_configs
from core.functions import train_gen_semantic, test_gen_semantic, train_gen_depth, test_gen_depth
from core.criterion import LossCR
from utils.utils import create_lr_scheduler
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')
    args = parser.parse_args()

    # set device 
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # create the config
    cfg = get_configs()
    update_configs(cfg, args.config)

    # Directory for saving model weights
    output_dir = os.path.join(cfg.TRAIN.OUTPUT_DIR, 'experiments', cfg.NAME)
    weights_dir = os.path.join(output_dir, 'model_weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    scene_list = [cfg.TRAIN.SCENE_TO_FINETUNE] if cfg.TRAIN.FINETUNE else cfg.DATASET.SCENE_LIST_TRAIN

    if cfg.DATASET.NAME == 'replica':
        train_dataset = Replica_Semantic(scene_list=scene_list, 
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING,
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS, 
                                        stage='train', 
                                        task=cfg.TASK, 
                                        finetune=cfg.TRAIN.FINETUNE, 
                                        compute_semantic_weights=cfg.TRAIN.COMPUTE_SEMANTIC_WEIGHTS) 
        test_dataset_nove_view = Replica_Semantic(scene_list=scene_list, 
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING, 
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS,
                                        stage='test', 
                                        task=cfg.TASK, 
                                        finetune=cfg.TRAIN.FINETUNE)
    elif cfg.DATASET.NAME == 'scannetpp':
        train_dataset = ScanNetpp_Semantic(root_dir=cfg.DATASET.ROOT_DIR, 
                                        scene_names=scene_list,
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING,
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS, 
                                        stage='train', 
                                        task=cfg.TASK, 
                                        finetune=cfg.TRAIN.FINETUNE, 
                                        compute_semantic_weights=cfg.TRAIN.COMPUTE_SEMANTIC_WEIGHTS) 
        test_dataset_nove_view = ScanNetpp_Semantic(root_dir=cfg.DATASET.ROOT_DIR, 
                                        scene_names=scene_list,
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING, 
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS,
                                        stage='test', 
                                        task=cfg.TASK, 
                                        finetune=cfg.TRAIN.FINETUNE)
    elif cfg.DATASET.NAME == 'scannet':
        train_dataset = ScanNet_Semantic(root_dir=cfg.DATASET.ROOT_DIR, 
                                        scene_names=scene_list,
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING,
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS, 
                                        stage='train', 
                                        task=cfg.TASK, 
                                        finetune=cfg.TRAIN.FINETUNE, 
                                        compute_semantic_weights=cfg.TRAIN.COMPUTE_SEMANTIC_WEIGHTS) 
        test_dataset_nove_view = ScanNet_Semantic(root_dir=cfg.DATASET.ROOT_DIR, 
                                        scene_names=scene_list,
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING, 
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS,
                                        stage='test', 
                                        task=cfg.TASK, 
                                        finetune=cfg.TRAIN.FINETUNE)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    test_loader_nove_view = DataLoader(test_dataset_nove_view, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    if not cfg.TRAIN.FINETUNE:
        if cfg.DATASET.NAME == 'replica':
            test_dataset = Replica_Semantic(scene_list=cfg.DATASET.SCENE_LIST_TEST, 
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING,
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS,  
                                        stage='test',
                                        task=cfg.TASK)
        elif cfg.DATASET.NAME == 'scannetpp':
            test_dataset = ScanNetpp_Semantic(root_dir=cfg.DATASET.ROOT_DIR, 
                                        scene_names=cfg.DATASET.SCENE_LIST_TEST,
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING,
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS, 
                                        stage='test', 
                                        split='test',
                                        task=cfg.TASK)
        elif cfg.DATASET.NAME == 'scannet':
            test_dataset = ScanNet_Semantic(root_dir=cfg.DATASET.ROOT_DIR, 
                                        scene_names=cfg.DATASET.SCENE_LIST_TEST,
                                        id2name=cfg.DATASET.ID2NAME, 
                                        color_mapping=cfg.DATASET.COLOR_MAPPING,
                                        class_weights=cfg.DATASET.CLASS_WEIGHTS, 
                                        stage='test', 
                                        split='test',
                                        task=cfg.TASK)
            
        
        test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    if cfg.TASK == 'semantic':
        num_classes = train_dataset.no_classses
    elif cfg.TASK == 'depth':
        num_classes = 1


    load_model_path = cfg.TRAIN.MODEL_PATH if cfg.TRAIN.FINETUNE or cfg.TRAIN.LOAD_PRETRAIN else None
    model = create_model_ggs(num_classes=num_classes, load_model=load_model_path).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, weight_decay=1e-4)
    if load_model_path:
        model_state_dict = torch.load(load_model_path, map_location='cpu') 
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
    
    if cfg.TRAIN.FINETUNE:
        warmup = False
        label_smoothing = 0.
    else:
        warmup = True
        label_smoothing = 0.15

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), cfg.TRAIN.EPOCHS, warmup=warmup)

    if cfg.TASK == 'semantic':
        if cfg.TRAIN.FINETUNE:
            criterion = LossCR(num_classes=num_classes, feat_dim=256, alpha=1, label_smoothing=label_smoothing)
        else:
            # criterion = LossCR(num_classes=num_classes, feat_dim=256, alpha=1, label_smoothing=label_smoothing)
            criterion = nn.CrossEntropyLoss(ignore_index=cfg.DATA.IGNORE_INDEX, label_smoothing=label_smoothing)

    elif cfg.TASK == 'depth':
        criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=os.path.join('runs', cfg.NAME))

    best_miou = 0.
    miou_test = 0.
    best_rmse = 1000.
    for epoch in tqdm(range(1, cfg.TRAIN.EPOCHS), desc="Epochs"):

        if cfg.TASK == 'semantic':
            loss_train = train_gen_semantic(model=model,
                                        optimizer=optimizer, 
                                        lr_scheduler=lr_scheduler,
                                        loader=train_loader, 
                                        criterion=criterion, 
                                        gard_clip_norm=cfg.TRAIN.GRAD_CLIP_NORM,
                                        finetune=cfg.TRAIN.FINETUNE,
                                        device=device
                                        )
            writer.add_scalar('Loss/avg_train_loss', loss_train, epoch)

            loss_test_novel_view, miou_valid_novel_view, acc_novel_view= test_gen_semantic(model=model,
                                                                                        loader=test_loader_nove_view, 
                                                                                        criterion=criterion, 
                                                                                        num_classes=num_classes,
                                                                                        finetune=cfg.TRAIN.FINETUNE,
                                                                                        device=device)
            writer.add_scalar('Loss/avg_test_loss_novel_view', loss_test_novel_view, epoch)
            writer.add_scalar('mIoU_valid/test_novel_view', miou_valid_novel_view, epoch)
            writer.add_scalar('Accuracy/test_novel_view', acc_novel_view, epoch)
            miou_test = miou_valid_novel_view

            if not cfg.TRAIN.FINETUNE:
                loss_test, miou_valid_test, acc_novel_test= test_gen_semantic(model=model,
                                                                                loader=test_loader, 
                                                                                criterion=criterion, 
                                                                                num_classes=num_classes, 
                                                                                finetune=cfg.TRAIN.FINETUNE,
                                                                                device=device)
                writer.add_scalar('Loss/avg_test_loss', loss_test, epoch)
                writer.add_scalar('mIoU_valid/test', miou_valid_test, epoch)
                writer.add_scalar('Accuracy/test', acc_novel_test, epoch)
                miou_test = miou_valid_test
        elif cfg.TASK == 'depth':
            mse_train, rmse_train = train_gen_depth(model=model,
                            optimizer=optimizer, 
                            lr_scheduler=lr_scheduler,
                            loader=train_loader, 
                            criterion=criterion, 
                            gard_clip_norm=cfg.TRAIN.GRAD_CLIP_NORM,
                            device=device
                            )
            writer.add_scalar('RMSE/train', rmse_train, epoch)
            writer.add_scalar('MSE/train', mse_train, epoch)

            mse_train_novel_view, rmse_train_novel_view = test_gen_depth(model=model,
                                                    loader=test_loader_nove_view, 
                                                    criterion=criterion, 
                                                    device=device)
            writer.add_scalar('RMSE/test_novel_view', rmse_train_novel_view, epoch)
            writer.add_scalar('MSE/test_novel_view', mse_train_novel_view, epoch)

            if not cfg.TRAIN.FINETUNE:
                mse_test, rmse_test = test_gen_depth(model=model,
                                            loader=test_loader, 
                                            criterion=criterion, 
                                            device=device)
                writer.add_scalar('RMSE/test', rmse_test, epoch)
                writer.add_scalar('MSE/test', mse_test, epoch)


        lr = lr_scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', lr, epoch)
                            
        # Check if this is the best model based on test mIoU
        if cfg.TASK == 'semantic':
            if miou_test > best_miou:
                best_miou = miou_test
                # Save the model and optimizer parameters
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                model_save_path = os.path.join(weights_dir, f'best_model_epoch_{epoch}_miou_{best_miou:.4f}.pth')
                torch.save(checkpoint, model_save_path)
                print(f"Saved new best model with mIoU: {best_miou:.4f} at epoch {epoch} in {model_save_path}")
            elif epoch == cfg.TRAIN.EPOCHS - 1:
                # Save the model and optimizer parameters
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                model_save_path = os.path.join(weights_dir, f'last_model_epoch_{epoch}_miou_{miou_test:.4f}.pth')
                torch.save(checkpoint, model_save_path)

            # Always save the last model of the current epoch
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            model_save_path = os.path.join(weights_dir, f'model_epoch_{epoch}.pth')
            torch.save(checkpoint, model_save_path)
        elif cfg.TASK == 'depth':
            if rmse_test < best_rmse:
                best_rmse = rmse_test
                # Save the model and optimizer parameters
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                model_save_path = os.path.join(weights_dir, f'best_model_epoch_{epoch}_rmse_{best_rmse:.4f}.pth')
                torch.save(checkpoint, model_save_path)
                print(f"Saved new best model with RMSE: {best_rmse:.4f} at epoch {epoch} in {model_save_path}")
            elif epoch == cfg.TRAIN.EPOCHS - 1:
                # Save the model and optimizer parameters
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                model_save_path = os.path.join(weights_dir, f'last_model_epoch_{epoch}_rmse_{rmse_test:.4f}.pth')
                torch.save(checkpoint, model_save_path)

            # Always save the last model of the current epoch
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            model_save_path = os.path.join(weights_dir, f'model_epoch_{epoch}.pth')
            torch.save(checkpoint, model_save_path)


    writer.close()

if __name__=='__main__':
    main()