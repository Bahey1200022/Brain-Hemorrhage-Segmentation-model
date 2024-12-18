import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import segmentation_models_pytorch as smp

def train_model(train_loader, val_loader, num_epochs, learning_rate, device):
    # Initialize the model from segmentation_models_pytorch
    model = smp.Unet(
        encoder_name="mobilenet_v2",        # Choose encoder, e.g., resnet34, mobilenet_v2, efficientnet-b7, etc.
        encoder_weights="imagenet",     # Use 'imagenet' pre-trained weights for encoder initialization
        in_channels=3,                  # Model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1                       # Model output channels (number of classes in your dataset)
    ).to(device)
    
    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        # Validation step (optional)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader)}")

    return model