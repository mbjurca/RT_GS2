import torch 
from torch.utils.data import DataLoader
from models.model import create_model_ssl
from data.datasets.Replica_PointContrast import Replica_PointContrast
from data.datasets.ScanNetpp_PointContrast import ScanNetpp_PointContrast
from data.datasets.ScanNet_PointContrast import ScanNet_PointContrast
from core.criterion import NCESoftmaxLoss
import argparse
import os
import argparse
from configs.configs import update_configs, get_configs
from core.functions import train_SSL_contrastive, test_SSL_contrastive
import gc
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')
    args = parser.parse_args()

    # create the config
    cfg = get_configs()
    update_configs(cfg, args.config)

    # set device 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.TRAIN.GPU)
    torch.cuda.set_device(cfg.TRAIN.GPU)
    device = f'cuda:{cfg.TRAIN.GPU}' if torch.cuda.is_available() else 'cpu'


    # Directory for saving model weights
    output_dir = os.path.join(cfg.TRAIN.OUTPUT_DIR, 'experiments', cfg.NAME)
    weights_dir = os.path.join(output_dir, 'model_weights')
    os.makedirs(weights_dir, exist_ok=True)

    if cfg.DATASET.NAME == 'replica':
        train_dataset = Replica_PointContrast(
                                scene_list=cfg.DATASET.SCENE_LIST_TRAIN, 
                                pair_dir=cfg.DATASET.PAIR_DIR_TRAIN,            
                                device=device)
        
        test_dataset = Replica_PointContrast(
                                scene_list=cfg.DATASET.SCENE_LIST_TEST, 
                                pair_dir=cfg.DATASET.PAIR_DIR_TEST,
                                device=device)
    elif cfg.DATASET.NAME == 'scannetpp':
        train_dataset = ScanNetpp_PointContrast(
                                root_dir=cfg.DATASET.ROOT_DIR,
                                scene_list=cfg.DATASET.SCENE_LIST_TRAIN, 
                                pair_dir=cfg.DATASET.PAIR_DIR_TRAIN,            
                                device=device, 
                                stage='train')
        
        test_dataset = ScanNetpp_PointContrast(
                                root_dir=cfg.DATASET.ROOT_DIR,
                                scene_list=cfg.DATASET.SCENE_LIST_TEST, 
                                pair_dir=cfg.DATASET.PAIR_DIR_TEST,
                                device=device, 
                                stage='test')
    elif cfg.DATASET.NAME == 'scannet':
        train_dataset = ScanNet_PointContrast(
                                root_dir=cfg.DATASET.ROOT_DIR,
                                scene_list=cfg.DATASET.SCENE_LIST_TRAIN, 
                                pair_dir=cfg.DATASET.PAIR_DIR_TRAIN,            
                                device=device, 
                                stage='train')
        
        test_dataset = ScanNet_PointContrast(
                                root_dir=cfg.DATASET.ROOT_DIR,
                                scene_list=cfg.DATASET.SCENE_LIST_TEST, 
                                pair_dir=cfg.DATASET.PAIR_DIR_TEST,
                                device=device, 
                                stage='test')

        
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)


    model = create_model_ssl(cfg, 'ptv3', no_out_features=32)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-5)

    criterion = NCESoftmaxLoss()

    writer = SummaryWriter(log_dir=os.path.join('runs', cfg.NAME))

    best_acc = 0.
    best_loss = 1000. 
    best_model_path = None                     
    for epoch in tqdm(range(1, cfg.TRAIN.EPOCHS), desc="Epochs"):

        loss_train, acc_train, acc_1_train = train_SSL_contrastive(model=model,
                                model_name='ptv3',
                                optimizer=optimizer, 
                                loader=train_loader, 
                                criterion=criterion, 
                                grad_clip_norm=cfg.TRAIN.GRAD_CLIP_NORM, 
                                device=device)
        
        writer.add_scalar('Loss/avg_train_loss', loss_train, epoch)
        writer.add_scalar('Accuracy/train', acc_train, epoch)
        writer.add_scalar('Accuracy_1/train', acc_1_train, epoch)

        loss_test, acc_test, acc_1_test = test_SSL_contrastive(model=model,
                            model_name='ptv3',
                            loader=test_loader, 
                            criterion=criterion, 
                            device=device)
        
        writer.add_scalar('Loss/avg_test_loss', loss_test, epoch)
        writer.add_scalar('Accuracy/test', acc_test, epoch)
        writer.add_scalar('Accuracy_1/test', acc_1_test, epoch)

        if loss_test < best_loss:
            # Delete the previously saved best model
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_loss = loss_test
            # Save the new best model
            checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
            model_save_path = os.path.join(weights_dir, f'best_model_epoch_{epoch}_loss_{best_loss:.4f}.pth')
            torch.save(model.state_dict(), model_save_path)
            best_model_path = model_save_path  # Update the best model path
            print(f"Saved new best model with loss: {best_loss:.4f} at epoch {epoch} in {model_save_path}")

        # Always save the last model of the current epoch
        model_save_path = os.path.join(weights_dir, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, model_save_path)

        torch.cuda.empty_cache()
        gc.collect()

    writer.close()

if __name__=='__main__':
    main()