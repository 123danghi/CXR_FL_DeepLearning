import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from classification.utils import get_model
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define transforms for inference
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path='improved_models/improved_tb_model.pth'):
    """Load the fine-tuned model"""
    # Check if improved model exists, otherwise use the original model
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        # Create ResNet50 model with 2 output classes
        model = get_model('ResNet50', classes=3)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Fine-tuned model not found. Loading pre-trained model and modifying output layer.")
        # Load pre-trained model and modify
        original_model_path = 'classification/best_models/ResNet50_full'
        model = get_model('ResNet50', classes=3)
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        
        # Modify last layer for binary classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)
    model.eval()
    return model

def predict_single_image(model, image_path):
    """Make prediction for a single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    pred_class = predicted.item()  # 0: Normal, 1: Tuberculosis
    prob = probabilities[0][pred_class].item()
    
    return pred_class, prob

def evaluate_dataset():
    """Evaluate the model on the entire dataset"""
    model = load_model()
    
    # Directories
    normal_dir = 'TB_Chest_Radiography_Database/Normal'
    tb_dir = 'TB_Chest_Radiography_Database/Tuberculosis'
    
    # Get all image files
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    tb_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) 
               if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    all_files = normal_files + tb_files
    # True labels (0: Normal, 1: Tuberculosis)
    all_labels = [0] * len(normal_files) + [1] * len(tb_files)
    
    print(f"Evaluating {len(all_files)} images ({len(normal_files)} Normal, {len(tb_files)} Tuberculosis)")
    
    # Make predictions
    predictions = []
    confidences = []
    
    for img_path in tqdm(all_files):
        pred_class, prob = predict_single_image(model, img_path)
        predictions.append(pred_class)
        confidences.append(prob)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    conf_mat = confusion_matrix(all_labels, predictions)
    class_report = classification_report(all_labels, predictions, 
                                        target_names=['Normal', 'Tuberculosis'])
    
    # Calculate class-specific accuracies
    normal_acc = accuracy_score(all_labels[:len(normal_files)], 
                               predictions[:len(normal_files)])
    tb_acc = accuracy_score(all_labels[len(normal_files):], 
                           predictions[len(normal_files):])
    
    # Print results
    print("\n===== MODEL EVALUATION ON FULL DATASET =====")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Normal class accuracy: {normal_acc*100:.2f}%")
    print(f"Tuberculosis class accuracy: {tb_acc*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    print("\nClassification Report:")
    print(class_report)
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion matrix
    im = ax1.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix')
    
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Normal', 'Tuberculosis'])
    ax1.set_yticklabels(['Normal', 'Tuberculosis'])
    
    # Add text annotations
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax1.text(j, i, conf_mat[i, j], ha="center", va="center", 
                    color="white" if conf_mat[i, j] > conf_mat.max()/2 else "black")
    
    # Confidence distribution
    normal_conf = confidences[:len(normal_files)]
    tb_conf = confidences[len(normal_files):]
    
    ax2.hist(normal_conf, alpha=0.5, label='Normal')
    ax2.hist(tb_conf, alpha=0.5, label='Tuberculosis')
    ax2.set_title('Prediction Confidence Distribution')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('full_dataset_evaluation.png')
    print("Evaluation visualizations saved to full_dataset_evaluation.png")
    
    return accuracy, normal_acc, tb_acc

def visualize_predictions():
    """Visualize some example predictions"""
    model = load_model()
    
    # Directories
    normal_dir = 'TB_Chest_Radiography_Database/Normal'
    tb_dir = 'TB_Chest_Radiography_Database/Tuberculosis'
    
    # Get random samples from each class
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    tb_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) 
               if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select a few random samples from each class
    np.random.seed(42)
    sample_normal = np.random.choice(normal_files, 5, replace=False)
    sample_tb = np.random.choice(tb_files, 5, replace=False)
    
    samples = list(sample_normal) + list(sample_tb)
    true_labels = [0] * len(sample_normal) + [1] * len(sample_tb)
    
    # Create a figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    # Make predictions and visualize
    for i, (img_path, true_label) in enumerate(zip(samples, true_labels)):
        # Make prediction
        pred_class, prob = predict_single_image(model, img_path)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Plot
        axes[i].imshow(img)
        
        # Set title color based on correctness
        color = 'green' if pred_class == true_label else 'red'
        class_names = ['Normal', 'Tuberculosis']
        
        axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_class]}\nConf: {prob:.2f}", 
                         color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    print("Example predictions saved to prediction_examples.png")

if __name__ == "__main__":
    # Evaluate on the full dataset
    accuracy, normal_acc, tb_acc = evaluate_dataset()
    
    # Show some example predictions
    visualize_predictions() 