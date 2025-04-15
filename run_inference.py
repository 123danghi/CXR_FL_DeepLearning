import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from classification.utils import get_model
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pre-trained model
model_path = 'classification/best_models/ResNet50_full'
model = get_model('ResNet50', classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classes
class_names = ['Normal', 'Tuberculosis', 'Other']

# Function to predict
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return class_names[predicted.item()], probabilities.cpu().numpy()[0]

# Test images from the dataset
test_dirs = ['TB_Chest_Radiography_Database/Normal', 'TB_Chest_Radiography_Database/Tuberculosis']
results = []

# Process 5 images from each directory
for dir_path in test_dirs:
    image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:5]
    for img_file in image_files:
        img_path = os.path.join(dir_path, img_file)
        prediction, probabilities = predict_image(img_path)
        results.append({
            'image': img_path,
            'true_class': os.path.basename(dir_path),
            'predicted_class': prediction,
            'probabilities': probabilities
        })
        print(f"Image: {img_path}")
        print(f"True class: {os.path.basename(dir_path)}")
        print(f"Predicted class: {prediction}")
        print(f"Probabilities: {probabilities}")
        print("-----------------------------------")

# Visualize some results
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()

for i, result in enumerate(results[:10]):
    img = Image.open(result['image'])
    axes[i].imshow(img, cmap='gray')
    color = 'green' if result['true_class'] == result['predicted_class'] else 'red'
    axes[i].set_title(f"True: {result['true_class']}\nPred: {result['predicted_class']}", color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('classification_results.png')
print("Results saved to classification_results.png") 