import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from classification.utils import get_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Custom dataset class
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data augmentation transforms for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Basic transforms for validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_data():
    print("Loading dataset...")
    
    # Directories
    normal_dir = 'TB_Chest_Radiography_Database/Normal'
    tb_dir = 'TB_Chest_Radiography_Database/Tuberculosis'
    
    # Get all image files
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    tb_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) 
               if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(normal_files)} Normal images and {len(tb_files)} Tuberculosis images")
    
    # Balance the dataset by oversampling the minority class (TB)
    # We only oversample in the training set
    if len(normal_files) > len(tb_files):
        ratio = len(normal_files) // len(tb_files)
        tb_files_oversample = tb_files * ratio
        tb_files_oversample += tb_files[:len(normal_files) - len(tb_files_oversample)]
    else:
        tb_files_oversample = tb_files
    
    # Create labels (0: Normal, 1: TB)
    normal_labels = [0] * len(normal_files)
    tb_labels = [1] * len(tb_files)
    tb_labels_oversample = [1] * len(tb_files_oversample)
    
    # Combine files and labels
    all_files = normal_files + tb_files
    all_labels = normal_labels + tb_labels
    
    # For training (oversampled)
    train_files = normal_files + tb_files_oversample
    train_labels = normal_labels + tb_labels_oversample
    
    # Shuffle
    train_data = list(zip(train_files, train_labels))
    random.shuffle(train_data)
    train_files, train_labels = zip(*train_data)
    
    # Split into train/validation
    split = int(0.8 * len(all_files))
    val_files = all_files[split:]
    val_labels = all_labels[split:]
    
    print(f"Training set size: {len(train_files)} images")
    print(f"Validation set size: {len(val_files)} images")
    
    # Create datasets
    train_dataset = ChestXRayDataset(train_files, train_labels, transform=train_transform)
    val_dataset = ChestXRayDataset(val_files, val_labels, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, val_files, val_labels

def fine_tune_model():
    # Load pre-trained model
    model_path = 'classification/best_models/ResNet50_full'
    model = get_model('ResNet50', classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # For fine-tuning, we'll modify the model to focus on just Normal vs TB classification
    # Get the number of features in the last layer
    num_ftrs = model.fc.in_features
    
    # Replace the final fully connected layer with a new one (2 classes)
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))  # Higher weight for TB class
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Load data
    train_loader, val_loader, val_files, val_labels = load_data()
    
    # Training settings
    num_epochs = 10
    best_val_acc = 0.0
    best_model_wts = None
    
    # Statistics
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc.item())
        
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Update scheduler based on validation loss
        scheduler.step(epoch_loss)
        
        # Save the best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_model_wts = model.state_dict().copy()
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save the fine-tuned model
    os.makedirs('improved_models', exist_ok=True)
    torch.save(model.state_dict(), 'improved_models/improved_tb_model.pth')
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    print("Training progress saved to training_progress.png")
    
    return model, val_files, val_labels

def evaluate_model(model, val_files, val_labels):
    # Set model to eval mode
    model.eval()
    
    # Predictions
    predictions = []
    class_mapping = {0: 'Normal', 1: 'Tuberculosis'}
    
    # Process all validation files
    print("Evaluating model on validation set...")
    
    for img_path, true_label in tqdm(zip(val_files, val_labels), total=len(val_files)):
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, pred = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        pred_label = pred.item()
        predictions.append(pred_label)
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, predictions)
    conf_mat = confusion_matrix(val_labels, predictions)
    class_report = classification_report(val_labels, predictions, target_names=list(class_mapping.values()))
    
    # Calculate class-specific accuracies
    normal_idx = [i for i, label in enumerate(val_labels) if label == 0]
    tb_idx = [i for i, label in enumerate(val_labels) if label == 1]
    
    normal_preds = [predictions[i] for i in normal_idx]
    tb_preds = [predictions[i] for i in tb_idx]
    
    normal_acc = sum([1 for p, t in zip(normal_preds, [0]*len(normal_preds)) if p == t]) / len(normal_preds)
    tb_acc = sum([1 for p, t in zip(tb_preds, [1]*len(tb_preds)) if p == t]) / len(tb_preds)
    
    # Print results
    print("\n===== IMPROVED MODEL EVALUATION =====")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Normal class accuracy: {normal_acc*100:.2f}%")
    print(f"Tuberculosis class accuracy: {tb_acc*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    print("\nClassification Report:")
    print(class_report)
    
    # Visualize results
    plt.figure(figsize=(8, 6))
    
    im = plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    
    tick_marks = np.arange(len(class_mapping))
    plt.xticks(tick_marks, list(class_mapping.values()))
    plt.yticks(tick_marks, list(class_mapping.values()))
    
    # Add text annotations
    thresh = conf_mat.max() / 2
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j], 
                    ha="center", va="center", 
                    color="white" if conf_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('improved_model_results.png')
    print("Evaluation results saved to improved_model_results.png")

def main():
    # Fine-tune the model
    model, val_files, val_labels = fine_tune_model()
    
    # Evaluate the model
    evaluate_model(model, val_files, val_labels)

if __name__ == "__main__":
    main() 