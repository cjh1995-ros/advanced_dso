from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

kitty_path = Path.cwd() / 'data' / 'data_odometry_color' / 'dataset' / 'sequences' / '00' / 'image_2' / '000000.png'

# Load the pre-trained segmentation model
model = models.segmentation.fcn_resnet50(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load and preprocess the input image
image_path = str(kitty_path)
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

# Visualize the segmentation mask
import matplotlib.pyplot as plt
plt.imshow(segmentation_mask, cmap='gray')
plt.axis('off')
plt.show()
