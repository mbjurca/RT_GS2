import torch
import torch.nn.functional as F
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from utils.utils import calculate_miou_and_log_confusion_matrix, miou, calculate_segmentation_metrics
from tqdm import tqdm
import numpy as np

def train_gen_semantic(model, optimizer, lr_scheduler, loader, criterion, gard_clip_norm, finetune, device):
    model.train()
    total_loss = 0

    # Adding tqdm progress bar for training
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for rendering, rendered_features, labels, _, _, _, _ in progress_bar:

        rendering = rendering.to(device)
        rendered_features = rendered_features.to(device)
        labels = labels.to(device)

        logits, z = model(rendering, rendered_features)
        if finetune:
            loss = criterion(logits, labels, None, z)
        else:
            # loss = criterion(logits, labels, None, z)
            loss = criterion(logits, labels)

        optimizer.zero_grad()  # Clear gradients.
        loss.backward()  # Backward pass.
        if gard_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gard_clip_norm)  # Gradient clipping
        optimizer.step()  # Update model parameters.
        lr_scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(batch_loss=loss.item())

    return total_loss / len(loader)

def test_gen_semantic(model, loader, criterion, num_classes, finetune, device):
    model.eval()
    total_loss = 0  
    
    preds_list = []
    labels_list = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Test", leave=False)
        for rendering, rendered_features, labels, _, _, _, _ in progress_bar:

            rendering = rendering.to(device)
            rendered_features = rendered_features.to(device)
            labels = labels.to(device)

            logits, z = model(rendering, rendered_features)

            if finetune:
                loss = criterion(logits, labels, None, z)
            else:
                # loss = criterion(logits, labels, None, z)
                loss = criterion(logits, labels)

            pred = logits.argmax(1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            preds_list.append(pred)
            labels_list.append(labels)

            total_loss += loss.item()

            progress_bar.set_postfix(batch_loss=loss.item())
    
    labels = np.concatenate(labels_list, axis=0)
    pred = np.concatenate(preds_list, axis=0)

    miou, miou_valid_class, accuracy, _, _ = calculate_segmentation_metrics(labels, pred, num_classes)

    return total_loss/len(loader), miou_valid_class, accuracy


def train_gen_depth(model, optimizer, lr_scheduler, loader, criterion, gard_clip_norm, device):
    model.train()
    total_loss = 0
    total_rmse = 0

    # Adding tqdm progress bar for training
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for rendering, rendered_features, _, gt_depth, _, _ in progress_bar:

        rendering = rendering.to(device)
        rendered_features = rendered_features.to(device)
        gt_depth = gt_depth.to(device)

        logits, z = model(rendering, rendered_features)
        loss = criterion(logits.squeeze(1), gt_depth)  # Loss computation.

        optimizer.zero_grad()  # Clear gradients.
        loss.backward()  # Backward pass.
        if gard_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gard_clip_norm)  # Gradient clipping
        optimizer.step()  # Update model parameters.
        lr_scheduler.step()

        total_loss += loss.item()
        total_rmse += torch.sqrt(loss).item()

        progress_bar.set_postfix(batch_loss=loss.item(), MRSE=loss.item()**0.5)

    return total_loss / len(loader), total_rmse / len(loader)  

def test_gen_depth(model, loader, criterion, device):
    model.eval()
    total_loss = 0  
    total_rmse = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Test", leave=False)
        for rendering, rendered_features, _,gt_depth, _, _ in progress_bar:

            rendering = rendering.to(device)
            rendered_features = rendered_features.to(device)
            gt_depth = gt_depth.to(device)

            logits, z = model(rendering, rendered_features)
            loss = criterion(logits.squeeze(1), gt_depth)  # Loss computation.

            total_loss += loss.item()
            total_rmse += torch.sqrt(loss).item()

            progress_bar.set_postfix(batch_loss=loss.item(), MRSE=loss.item()**0.5)

    return total_loss / len(loader), total_rmse / len(loader)

def train_SSL_contrastive(model, model_name, optimizer, loader, criterion, grad_clip_norm, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_accuracy_1 = 0

    progress_bar = tqdm(loader, desc="Train", leave=False)
    for xyz1, xyz2, features1, features2, correspondences in progress_bar:
        
        xyz1 = xyz1
        xyz2 = xyz2
        features1 = features1
        features2 = features2

        input1 = torch.cat([xyz1, features1], dim=2).type(torch.float32).to(device)
        input2 = torch.cat([xyz2, features2], dim=2).type(torch.float32).to(device)

        correspondences = correspondences.squeeze().to(device)
        if model_name == 'ptv3':
            input = torch.cat([input1, input2], dim=1).type(torch.float32).to(device)
            input_dict = {
                'feat' : input.squeeze(0), 
                'coord' : input[..., :3].squeeze(0),
                'grid_size' : 0.07, 
                'offset': torch.tensor([input1.shape[1], input1.shape[1]+input2.shape[1]]).to(device)
            }
            pred = model(input_dict).feat
            pred_1 = pred[:input1.shape[1], :]
            pred_2 = pred[input1.shape[1]:, :]
        elif model_name == 'pointnetpp':
            pred_1 = model(input1).permute(0, 2, 1).squeeze()
            pred_2 = model(input2).permute(0, 2, 1).squeeze()

        q_idx = correspondences[:, 0]
        k_pos_idx = correspondences[:, 1]

        q = pred_1[q_idx]
        k_pos = pred_2[k_pos_idx]

        if 4096 < q.shape[0]:
            sampled_inds = np.random.choice(q.shape[0], 4096, replace=False)
            q = q[sampled_inds]
            k_pos = k_pos[sampled_inds]
        
        npos = q.shape[0] 

        # pos logit
        logits = torch.mm(q, k_pos.transpose(1, 0)) # npos by npos
        labels = torch.arange(npos).cuda().long()
        out = torch.div(logits, 0.07)

        loss = criterion(out, labels)

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        acc = (torch.argmax(logits, dim=1) == labels).sum().item() / npos
        total_accuracy += acc

        top_1_percent_threshold = torch.quantile(out, 0.998, dim=1)
        is_top_1_percent = out[labels, labels] >= top_1_percent_threshold
        acc_1 = is_top_1_percent.sum().item() / npos
        total_accuracy_1 += acc_1

        progress_bar.set_postfix(batch_loss=loss.item(), batch_accuracy=acc, batch_accuracy_1=acc_1)

    return total_loss / len(loader), total_accuracy / len(loader), total_accuracy_1 / len(loader)

def test_SSL_contrastive(model, model_name, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_accuracy_1 = 0

    with torch.no_grad():
        
        progress_bar = tqdm(loader, desc="Test", leave=False)
        for xyz1, xyz2, features1, features2, correspondences in progress_bar:
            
            xyz1 = xyz1.to(device)
            xyz2 = xyz2.to(device)
            features1 = features1.to(device)
            features2 = features2.to(device)

            input1 = torch.cat([xyz1, features1], dim=2).type(torch.float32)
            input2 = torch.cat([xyz2, features2], dim=2).type(torch.float32)

            correspondences = correspondences.squeeze().to(device)
            if model_name == 'ptv3':
                input = torch.cat([input1, input2], dim=1).type(torch.float32).to(device)
                input_dict = {
                    'feat' : input.squeeze(0), 
                    'coord' : input[..., :3].squeeze(0),
                    'grid_size' : 0.07, 
                    'offset': torch.tensor([input1.shape[1], input1.shape[1]+input2.shape[1]]).to(device)
                }
                pred = model(input_dict).feat
                pred_1 = pred[:input1.shape[1], :]
                pred_2 = pred[input1.shape[1]:, :]
            elif model_name == 'pointnetpp':
                pred_1 = model(input1).permute(0, 2, 1).squeeze()
                pred_2 = model(input2).permute(0, 2, 1).squeeze()

            q_idx = correspondences[:, 0]
            k_pos_idx = correspondences[:, 1]

            q = pred_1[q_idx]
            k_pos = pred_2[k_pos_idx]

            if 4096 < q.shape[0]:
                sampled_inds = np.random.choice(q.shape[0], 4096, replace=False)
                q = q[sampled_inds]
                k_pos = k_pos[sampled_inds]
            
            npos = q.shape[0] 

            # pos logit
            logits = torch.mm(q, k_pos.transpose(1, 0)) # npos by npos
            labels = torch.arange(npos).cuda().long()
            out = torch.div(logits, 0.07)

            loss = criterion(out, labels)
            total_loss += loss.item()

            acc = (torch.argmax(logits, dim=1) == labels).sum().item() / npos
            total_accuracy += acc

            top_1_percent_threshold = torch.quantile(out, 0.998, dim=1)
            is_top_1_percent = out[labels, labels] >= top_1_percent_threshold
            acc_1 = is_top_1_percent.sum().item() / npos
            total_accuracy_1 += acc_1

            progress_bar.set_postfix(batch_loss=loss.item(), batch_accuracy=acc, batch_accuracy_1=acc_1)

    return total_loss / len(loader), total_accuracy / len(loader), total_accuracy_1 / len(loader)