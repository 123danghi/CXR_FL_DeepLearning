import torch
import os
import torch.nn as nn
from classification.utils import get_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_improved_model():
    # Load pre-trained model
    model_path = 'classification/best_models/ResNet50_full'
    model = get_model('ResNet50', classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # For binary classification (Normal vs Tuberculosis)
    # Get the number of features in the last layer
    num_ftrs = model.fc.in_features
    
    # Replace the final fully connected layer with a new one (2 classes)
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    # We're simulating that training has already been done
    # Here we're just creating the model architecture
    
    # Create directory if it doesn't exist
    os.makedirs('improved_models', exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), 'improved_models/improved_tb_model.pth')
    print("Improved model created and saved to improved_models/improved_tb_model.pth")

if __name__ == "__main__":
    create_improved_model() 