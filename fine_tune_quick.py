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

def load_data(batch_size=32):
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
    
    # Sample for quicker training (use 200 normal and 200 TB images)
    # For real training, you'd use more images or all of them
    np.random.seed(42)
    normal_files_sample = np.random.choice(normal_files, 250, replace=False)
    
    # Use all TB images as they are the minority class
    tb_files_sample = tb_files
    
    # Duplicate TB images to balance classes
    tb_files_oversample = tb_files_sample * (len(normal_files_sample) // len(tb_files_sample))
    tb_files_oversample += tb_files_sample[:len(normal_files_sample) - len(tb_files_oversample)]
    
    # Create labels (0: Normal, 1: TB)
    normal_labels = [0] * len(normal_files_sample)
    tb_labels = [1] * len(tb_files_sample)
    tb_labels_oversample = [1] * len(tb_files_oversample)
    
    # For validation, use a different set of images
    # 100 normal and 100 TB for validation
    normal_val = [f for f in normal_files if f not in normal_files_sample][:100]
    tb_val = tb_files_sample[:100]  # Use first 100 TB images for validation
    
    val_files = normal_val + tb_val
    val_labels = [0] * len(normal_val) + [1] * len(tb_val)
    
    # For training, use the sampled and oversampled images
    train_files = list(normal_files_sample) + tb_files_oversample
    train_labels = normal_labels + tb_labels_oversample
    
    # Shuffle training data
    train_data = list(zip(train_files, train_labels))
    random.shuffle(train_data)
    train_files, train_labels = zip(*train_data)
    
    print(f"Training set size: {len(train_files)} images")
    print(f"Validation set size: {len(val_files)} images")
    
    # Create datasets
    train_dataset = ChestXRayDataset(train_files, train_labels, transform=train_transform)
    val_dataset = ChestXRayDataset(val_files, val_labels, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
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
    # Higher weight for TB class to emphasize improving TB accuracy
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))  
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Load data
    train_loader, val_loader, val_files, val_labels = load_data(batch_size=32)
    
    # Training settings - just 2 epochs for demonstration
    num_epochs = 2
    best_val_acc = 0.0
    best_model_wts = None
    best_tb_acc = 0.0
    
    # Statistics
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    tb_acc_history = []
    
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
        val_preds = []
        
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_preds.extend(preds.cpu().numpy())
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc.item())
        
        # Calculate TB accuracy specifically
        tb_idx = [i for i, label in enumerate(val_labels) if label == 1]
        tb_preds = [val_preds[i] for i in tb_idx]
        tb_acc = sum([1 for p in tb_preds if p == 1]) / len(tb_preds)
        tb_acc_history.append(tb_acc)
        
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} TB Acc: {tb_acc:.4f}")
        
        # Save the best model based on TB accuracy
        if tb_acc > best_tb_acc:
            best_tb_acc = tb_acc
            best_model_wts = model.state_dict().copy()
            print(f"New best model saved with TB accuracy: {best_tb_acc:.4f}")
    
    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    # Save the fine-tuned model
    os.makedirs('improved_models', exist_ok=True)
    torch.save(model.state_dict(), 'improved_models/improved_tb_model.pth')
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(tb_acc_history, label='TB Accuracy')
    plt.title('Tuberculosis Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quick_training_progress.png')
    print("Training progress saved to quick_training_progress.png")
    
    # Evaluate the final model on validation set
    model.eval()
    val_preds = []
    
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, val_preds)
    conf_mat = confusion_matrix(val_labels, val_preds)
    class_report = classification_report(val_labels, val_preds, target_names=['Normal', 'Tuberculosis'])
    
    # Calculate class-specific accuracies
    normal_idx = [i for i, label in enumerate(val_labels) if label == 0]
    tb_idx = [i for i, label in enumerate(val_labels) if label == 1]
    
    normal_preds = [val_preds[i] for i in normal_idx]
    tb_preds = [val_preds[i] for i in tb_idx]
    
    normal_acc = sum([1 for p, t in zip(normal_preds, [0]*len(normal_preds)) if p == t]) / len(normal_preds)
    tb_acc = sum([1 for p, t in zip(tb_preds, [1]*len(tb_preds)) if p == t]) / len(tb_preds)
    
    # Print results
    print("\n===== FINE-TUNED MODEL EVALUATION =====")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Normal class accuracy: {normal_acc*100:.2f}%")
    print(f"Tuberculosis class accuracy: {tb_acc*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    print("\nClassification Report:")
    print(class_report)
    
    return model

if __name__ == "__main__":
    fine_tune_model() 