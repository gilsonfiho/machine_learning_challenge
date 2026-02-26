"""
src/models/builder.py - Construção e carregamento de modelos
"""

import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES


def build_resnet50(fine_tuning=False, num_classes=NUM_CLASSES):
    """ResNet50 pré-treinado"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if fine_tuning:
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def build_efficientnet(fine_tuning=False, num_classes=NUM_CLASSES):
    """EfficientNetV2-S pré-treinado"""
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if fine_tuning:
        for param in model.features[-2:].parameters():
            param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def build_convnext(fine_tuning=False, num_classes=NUM_CLASSES):
    """ConvNeXt-Tiny pré-treinado"""
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if fine_tuning:
        for param in model.features[5].parameters():
            param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def get_model(model_name, fine_tuning=False, num_classes=NUM_CLASSES):
    """Obter modelo por nome"""
    models_map = {
        'resnet50': build_resnet50,
        'efficientnet': build_efficientnet,
        'convnext': build_convnext,
    }
    
    if model_name not in models_map:
        raise ValueError(f"Modelo '{model_name}' não encontrado")
    
    return models_map[model_name](fine_tuning=fine_tuning, num_classes=num_classes)
