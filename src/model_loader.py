import torch
from torchvision import models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def load_pretrained_model():
    """
    Load a pre-trained Faster R-CNN model.
    """
    # Load the model with updated weights parameter
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detection_model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    detection_model.eval()  # Set model to evaluation mode
    return detection_model
