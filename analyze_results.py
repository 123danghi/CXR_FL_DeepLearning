import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from classification.utils import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from segmentation.common import get_model as get_seg_model, jaccard

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
    
    return mask

# Load and analyze the dataset
def analyze_dataset():
    print("Analyzing dataset...")
    
    # Directories
    data_dirs = ['TB_Chest_Radiography_Database/Normal', 'TB_Chest_Radiography_Database/Tuberculosis']
    class_mapping = {'Normal': 0, 'Tuberculosis': 1}
    
    # Prepare for metrics
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    segmentation_scores = []
    
    max_images_per_class = 100  # Limit to analyze in reasonable time
    
    # Process images
    for dir_path in data_dirs:
        class_name = os.path.basename(dir_path)
        class_idx = class_mapping[class_name]
        
        # Get image files
        image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if max_images_per_class > 0:
            image_files = image_files[:max_images_per_class]
        
        print(f"Processing {len(image_files)} images from {class_name} class...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(dir_path, img_file)
            
            # Classification
            pred_idx, probs = predict_image(img_path)
            true_labels.append(class_idx)
            predicted_labels.append(pred_idx)
            confidence_scores.append(probs[pred_idx])
            
            # Segmentation (for a sample of images)
            if len(segmentation_scores) < 20 and len(segmentation_scores) % 2 == (0 if class_name == 'Normal' else 1):
                seg_mask = predict_segmentation(img_path)
                # Calculate percentage of image that is segmented as lung
                seg_percentage = seg_mask.mean().item() * 100
                segmentation_scores.append({
                    'image': img_path,
                    'class': class_name,
                    'seg_percentage': seg_percentage
                })
    
    # Calculate classification metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, target_names=list(class_mapping.keys()), output_dict=True)
    
    # Print classification results
    print("\n===== CLASSIFICATION RESULTS =====")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    for cls in class_mapping.keys():
        print(f"{cls}: Precision: {class_report[cls]['precision']:.2f}, "
              f"Recall: {class_report[cls]['recall']:.2f}, "
              f"F1-Score: {class_report[cls]['f1-score']:.2f}")
    
    # Print segmentation results
    print("\n===== SEGMENTATION RESULTS =====")
    normal_segs = [item['seg_percentage'] for item in segmentation_scores if item['class'] == 'Normal']
    tb_segs = [item['seg_percentage'] for item in segmentation_scores if item['class'] == 'Tuberculosis']
    
    print(f"Average lung area in Normal X-rays: {np.mean(normal_segs):.2f}%")
    print(f"Average lung area in Tuberculosis X-rays: {np.mean(tb_segs):.2f}%")
    
    # Visualize results
    visualize_results(true_labels, predicted_labels, confidence_scores, segmentation_scores, class_mapping)
    
    return accuracy, class_report, segmentation_scores

def visualize_results(true_labels, predicted_labels, confidence_scores, segmentation_scores, class_mapping):
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Confusion Matrix Heatmap
    ax1 = fig.add_subplot(221)
    cm = confusion_matrix(true_labels, predicted_labels)
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix')
    
    tick_marks = np.arange(len(class_mapping))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(list(class_mapping.keys()))
    ax1.set_yticklabels(list(class_mapping.keys()))
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    
    # 2. Classification Accuracy by Class
    ax2 = fig.add_subplot(222)
    class_report = classification_report(true_labels, predicted_labels, target_names=list(class_mapping.keys()), output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    normal_scores = [class_report['Normal'][m] for m in metrics]
    tb_scores = [class_report['Tuberculosis'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, normal_scores, width, label='Normal')
    ax2.bar(x + width/2, tb_scores, width, label='Tuberculosis')
    
    ax2.set_title('Classification Metrics by Class')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # 3. Confidence Distribution
    ax3 = fig.add_subplot(223)
    normal_conf = [confidence_scores[i] for i in range(len(true_labels)) if true_labels[i] == 0]
    tb_conf = [confidence_scores[i] for i in range(len(true_labels)) if true_labels[i] == 1]
    
    ax3.hist(normal_conf, alpha=0.5, label='Normal')
    ax3.hist(tb_conf, alpha=0.5, label='Tuberculosis')
    ax3.set_title('Confidence Score Distribution')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # 4. Segmentation Area Comparison
    ax4 = fig.add_subplot(224)
    normal_segs = [item['seg_percentage'] for item in segmentation_scores if item['class'] == 'Normal']
    tb_segs = [item['seg_percentage'] for item in segmentation_scores if item['class'] == 'Tuberculosis']
    
    ax4.boxplot([normal_segs, tb_segs], labels=['Normal', 'Tuberculosis'])
    ax4.set_title('Lung Area Segmentation Comparison')
    ax4.set_ylabel('Percentage of Image Segmented as Lung')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    print("Visualization saved to analysis_results.png")

if __name__ == "__main__":
    accuracy, class_report, seg_scores = analyze_dataset() 