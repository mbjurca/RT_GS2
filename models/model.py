import torch 
from torch import nn
from models.PointTransformerV3 import PointTransformerV3
import torch.nn.functional as F

from models.gaussian_splatting.gaussian_renderer import render
from models.AsymFormer.src.AsymFormer import B0_T as AsymFormer

class GS3DGS(nn.Module):

    def __init__(self, num_classes, pc_feature_extractor, pc_feature_extractor_name, img_size, device): 

        super(GS3DGS, self).__init__()
        self.device = device        
        self.pc_feature_extractor = pc_feature_extractor
        self.pc_feature_extractor_name = pc_feature_extractor_name
        self.img_size = img_size
        self.head = AsymFormer(num_classes=num_classes)

    def forward(self, features, rendering, view, gaussians, pipeline, background, mask):


        if self.pc_feature_extractor_name == 'ptv3':
            batch = torch.full((features.shape[1],), 0, device=self.device)
            xyz = features[:, :, :3]
            input_dict = {
                'feat' : xyz.squeeze(0), 
                'coord' : xyz.squeeze(0), 
                'grid_size' : 0.06, 
                'batch' : batch
            }
            model_features = self.pc_feature_extractor(input_dict).feat.unsqueeze(0).permute(1, 2, 0)
        elif self.pc_feature_extractor_name == 'pointnetpp':
            model_features = self.pc_feature_extractor(features).permute(2, 1, 0)
        elif self.pc_feature_extractor_name == 'randlanet':
            model_features = self.pc_feature_extractor(features).permute(2, 1, 0)

        if mask != None:
            mask = mask.unsqueeze(0)

        rendered_features = render(view[0], gaussians[0], pipeline[0], background[0], pc_features=model_features, mask=mask)['render_object'].unsqueeze(0)
        rendered_features = F.interpolate(rendered_features, size=(self.img_size[1], self.img_size[0]), mode='bilinear', align_corners=False)
        
        logits = self.head(rendering, rendered_features)

        return logits
    

def create_model(cfg, pc_feature_extractor_name, img_size, device):

    if pc_feature_extractor_name=='ptv3':
        pc_feature_extractor = PointTransformerV3(in_channels=3)

    model = GS3DGS(cfg.DATASET.NUM_CLASSES, 
                    pc_feature_extractor=pc_feature_extractor, 
                    pc_feature_extractor_name = pc_feature_extractor_name, 
                    img_size=img_size,
                    device=device).to(device)
        
    return model

def create_model_ssl(cfg, backbone, no_out_features):

    if backbone == 'ptv3':
        model_1 = PointTransformerV3(in_channels=10)
        return model_1

def create_model_ggs(num_classes, load_model=None):
    if load_model:
        model = AsymFormer(num_classes=num_classes)
        model_state_dict = torch.load(load_model, map_location='cpu')
        model.load_state_dict(model_state_dict['model_state_dict'])
        return model
    
    return AsymFormer(num_classes=num_classes)

    


