import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import glob
from classification.utils import get_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define transforms for inference (same as in app.py)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
def load_model(model_path='improved_models/improved_tb_model.pth'):
    # Create ResNet50 model with 2 output classes
    model = get_model('ResNet50', classes=3)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()  # 0: Normal, 1: Tuberculosis

def create_confusion_matrix():
    # Load the model
    model = load_model()
    
    # Define test directories
    normal_dir = 'test_data_small/Normal'
    tb_dir = 'test_data_small/Tuberculosis'
    
    # Check if the test directories exist
    if not os.path.exists(normal_dir) or not os.path.exists(tb_dir):
        print("Test directories not found. Creating sample structure...")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        print(f"Please place Normal X-ray images in {normal_dir} and Tuberculosis X-ray images in {tb_dir}")
        return
    
    # Get file lists
    normal_files = glob.glob(os.path.join(normal_dir, '*.png')) + glob.glob(os.path.join(normal_dir, '*.jpg'))
    tb_files = glob.glob(os.path.join(tb_dir, '*.png')) + glob.glob(os.path.join(tb_dir, '*.jpg'))
    
    if len(normal_files) == 0 or len(tb_files) == 0:
        print(f"No images found in test directories.")
        print(f"Please place Normal X-ray images in {normal_dir} and Tuberculosis X-ray images in {tb_dir}")
        return
    
    print(f"Found {len(normal_files)} Normal images and {len(tb_files)} Tuberculosis images")
    
    # Predict on test images
    y_true = []
    y_pred = []
    
    # Normal images (class 0)
    print("Processing Normal images...")
    for i, img_path in enumerate(normal_files):
        y_true.append(0)  # True label is Normal (0)
        prediction = predict_image(model, img_path)
        y_pred.append(prediction)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(normal_files)} Normal images")
    
    # TB images (class 1)
    print("Processing Tuberculosis images...")
    for i, img_path in enumerate(tb_files):
        y_true.append(1)  # True label is TB (1)
        prediction = predict_image(model, img_path)
        y_pred.append(prediction)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(tb_files)} TB images")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall for TB cases
    specificity = tn / (tn + fp)  # Recall for Normal cases
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Plot confusion matrix
    classes = ['Normal', 'Tuberculosis']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix for Tuberculosis Detection')
    
    # Add metrics to the plot
    metrics_text = f"""
    Accuracy: {accuracy:.4f}
    Sensitivity (TB Recall): {sensitivity:.4f}
    Specificity (Normal Recall): {specificity:.4f}
    Precision (TB): {precision:.4f}
    F1 Score (TB): {f1_score:.4f}
    """
    plt.figtext(0.02, 0.02, metrics_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print(f"Confusion matrix saved as confusion_matrix.png")
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (TB Recall): {sensitivity:.4f}")
    print(f"Specificity (Normal Recall): {specificity:.4f}")
    print(f"Precision (TB): {precision:.4f}")
    print(f"F1 Score (TB): {f1_score:.4f}")
    
    return cm, accuracy, sensitivity, specificity, precision, f1_score

if __name__ == "__main__":
    create_confusion_matrix() 