import torch
import torch.nn as nn
import torchvision.models as models


def build_model(cfg):
    backbone = cfg['model'].get('backbone','resnet50')
    pretrained = cfg['model'].get('pretrained',True)
    output_dim = cfg['model'].get('output_dim',28)
    if backbone == 'resnet50':
        m = models.resnet50(pretrained=pretrained)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, output_dim)
    else:
        raise NotImplementedError(backbone)
    return m