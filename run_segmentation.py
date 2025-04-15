import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from segmentation.common import get_model, jaccard
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pre-trained model
model_path = 'segmentation/best_model/unet'
model = get_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Function to predict segmentation
def predict_segmentation(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        mask = output.round()
    
    return mask.cpu().numpy()[0, 0]

# Test images from the dataset
test_dirs = ['TB_Chest_Radiography_Database/Normal', 'TB_Chest_Radiography_Database/Tuberculosis']
results = []

# Process 3 images from each directory
for dir_path in test_dirs:
    image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:3]
    for img_file in image_files:
        img_path = os.path.join(dir_path, img_file)
        segmentation_mask = predict_segmentation(img_path)
        
        results.append({
            'image': img_path,
            'true_class': os.path.basename(dir_path),
            'segmentation_mask': segmentation_mask
        })
        print(f"Processed segmentation for: {img_path}")

# Visualize results
fig, axes = plt.subplots(len(results), 2, figsize=(10, 4*len(results)))

for i, result in enumerate(results):
    # Original image
    img = Image.open(result['image']).convert('L')
    img = img.resize((1024, 1024))
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(f"Original: {os.path.basename(result['image'])}")
    axes[i, 0].axis('off')
    
    # Segmentation mask
    axes[i, 1].imshow(result['segmentation_mask'], cmap='viridis')
    axes[i, 1].set_title("Lung Segmentation")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('segmentation_results.png')
print("Results saved to segmentation_results.png") 