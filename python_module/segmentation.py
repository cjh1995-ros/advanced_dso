from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

PASCAL_VOC_SEGMENT_LABEL = {
    1: 'Aeroplane',
    2: 'Bicycle',
    3: 'Bird',
    4: 'Boat',
    5: 'Bottle',
    6: 'Bus',
    7: 'Car',
    8: 'Dog',
    9: 'Chair',
    10: 'Cow',
    11: 'Diningtable',
    12: 'Cat',
    13: 'Horse',
    14: 'Motorbike',
    15: 'Person',
    16: 'Pottedplant',
    17: 'Sheep',
    18: 'Sofa',
    19: 'Train',
    20: 'Tvmonitor',
}

DEL_LIST = [9, 11, 16, 18, 20]

PASCAL_VOC_SEGMENT_LABEL = {k: v for k, v in PASCAL_VOC_SEGMENT_LABEL.items() if k not in DEL_LIST}

kitty_path = Path.cwd() / 'data' / 'data_odometry_color' / 'dataset' / 'sequences' / '00' / 'image_2' / '000000.png'

def segment(image_path: Path):
    # Load the pre-trained segmentation model
    model = models.segmentation.fcn_resnet50(pretrained=True)

    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the input image
    image_path = str(image_path)
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # Convert the output to a segmentation mask
    _, predicted_class = torch.max(output, dim=0)
    segmentation_mask = predicted_class.byte().cpu().numpy()

    segmentation_mask[segmentation_mask != 0] = 1

    return segmentation_mask