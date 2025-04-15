import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from classification.utils import get_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from segmentation.common import get_model as get_seg_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === CLASSIFICATION ANALYSIS ===

# Load the classification model
clf_model_path = 'classification/best_models/ResNet50_full'
clf_model = get_model('ResNet50', classes=3)
clf_model.load_state_dict(torch.load(clf_model_path, map_location=device))
clf_model.to(device)
clf_model.eval()

# Define transforms for classification
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classes
class_names = ['Normal', 'Tuberculosis', 'Other']

# Function to predict for classification
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = clf_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = clf_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return predicted.item(), probabilities.cpu().numpy()[0]

# === SEGMENTATION ANALYSIS ===

# Load the segmentation model
seg_model_path = 'segmentation/best_model/unet'
seg_model = get_seg_model()
seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
seg_model.to(device)
seg_model.eval()

# Define transforms for segmentation
seg_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Function to predict segmentation
def predict_segmentation(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_tensor = seg_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = seg_model(image_tensor)
        mask = output.round()
    
    return mask.mean().item() * 100  # Return percentage of lung area

# Analyze a sample of the dataset
def analyze_sample():
    print("Analyzing a sample of the dataset...")
    
    # Directories
    data_dirs = ['TB_Chest_Radiography_Database/Normal', 'TB_Chest_Radiography_Database/Tuberculosis']
    class_mapping = {'Normal': 0, 'Tuberculosis': 1}
    
    # Prepare for metrics
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    seg_percentages = {'Normal': [], 'Tuberculosis': []}
    
    # Process 20 images from each class
    sample_size = 20
    
    for dir_path in data_dirs:
        class_name = os.path.basename(dir_path)
        class_idx = class_mapping[class_name]
        
        # Get image files
        image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:sample_size]
        
        print(f"Processing {len(image_files)} images from {class_name} class...")
        
        for img_file in image_files:
            img_path = os.path.join(dir_path, img_file)
            
            # Classification
            pred_idx, probs = predict_image(img_path)
            true_labels.append(class_idx)
            predicted_labels.append(pred_idx)
            confidence_scores.append(probs[pred_idx])
            
            # Segmentation (for a smaller subset)
            if len(seg_percentages[class_name]) < 10:
                seg_percent = predict_segmentation(img_path)
                seg_percentages[class_name].append(seg_percent)
                print(f"Image: {img_path}, Class: {class_name}, Predicted: {class_names[pred_idx]}, Lung area: {seg_percent:.2f}%")
    
    # Calculate classification metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    
    # For the classification report, filter to only include 0 and 1 classes
    report_true = []
    report_pred = []
    for i in range(len(true_labels)):
        if predicted_labels[i] < 2:  # Only include Normal and Tuberculosis predictions
            report_true.append(true_labels[i])
            report_pred.append(predicted_labels[i])
    
    class_report = classification_report(
        report_true, 
        report_pred, 
        labels=[0, 1], 
        target_names=list(class_mapping.keys())
    )
    
    # Count how many were classified as "Other"
    other_count = predicted_labels.count(2)
    
    # Print classification results
    print("\n===== CLASSIFICATION RESULTS =====")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Images classified as 'Other': {other_count} out of {len(true_labels)}")
    
    print("\nConfusion Matrix (excluding 'Other' class):")
    print(conf_matrix)
    
    print("\nClassification Report (excluding 'Other' class):")
    print(class_report)
    
    # Calculate class-specific accuracies
    normal_acc = sum([1 for i in range(len(true_labels)) if true_labels[i] == 0 and predicted_labels[i] == 0]) / sample_size
    tb_acc = sum([1 for i in range(len(true_labels)) if true_labels[i] == 1 and predicted_labels[i] == 1]) / sample_size
    
    print(f"\nNormal class accuracy: {normal_acc*100:.2f}%")
    print(f"Tuberculosis class accuracy: {tb_acc*100:.2f}%")
    
    # Print segmentation results
    print("\n===== SEGMENTATION RESULTS =====")
    print(f"Average lung area in Normal X-rays: {np.mean(seg_percentages['Normal']):.2f}%")
    print(f"Average lung area in Tuberculosis X-rays: {np.mean(seg_percentages['Tuberculosis']):.2f}%")
    
    # Create a simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    im = ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix (excluding Other class)')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(list(class_mapping.keys()))
    ax1.set_yticklabels(list(class_mapping.keys()))
    
    # Add text annotations to confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax1.text(j, i, conf_matrix[i, j], ha="center", va="center", 
                    color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
    
    # Segmentation results boxplot
    ax2.boxplot([seg_percentages['Normal'], seg_percentages['Tuberculosis']], 
                labels=['Normal', 'Tuberculosis'])
    ax2.set_title('Lung Area Segmentation Comparison')
    ax2.set_ylabel('Percentage of Image Segmented as Lung')
    
    plt.tight_layout()
    plt.savefig('sample_analysis.png')
    print("Visualization saved to sample_analysis.png")

if __name__ == "__main__":
    analyze_sample() 