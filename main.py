import torch
from data_loader import get_dataloaders
from model import UNet
from train import train_model
import config
from torchvision import transforms
import numpy as np
import segmentation_models_pytorch as smp

def evaluate_model(model, test_loader, device):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            # Compute Dice coefficient
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            total_dice += dice.mean().item()

            # Compute IoU
            intersection = (preds * masks).sum((1, 2, 3))
            union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            total_iou += iou.mean().item()

    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

def main():
    # Data augmentation and transformation for images
    train_image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
        transforms.RandomRotation(30),      # Randomly rotate the image by up to 30 degrees
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Randomly crop and resize the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, etc.
        transforms.ToTensor(),              # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    val_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data augmentation and transformation for masks
    train_mask_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the mask horizontally
        transforms.RandomVerticalFlip(),    # Randomly flip the mask vertically
        transforms.RandomRotation(30),      # Randomly rotate the mask by up to 30 degrees
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Randomly crop and resize the mask
        transforms.ToTensor(),              # Convert the mask to a tensor
        transforms.Resize((256, 256))       # Resize the mask
    ])

    val_mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        config.train_dir, 
        batch_size=16, 
        train_image_transform=train_image_transform, 
        val_image_transform=val_image_transform, 
        train_mask_transform=train_mask_transform, 
        val_mask_transform=val_mask_transform
    )

    # Train model
    trained_model = train_model(train_loader, val_loader, config.num_epochs, config.learning_rate, config.device)

    # Save the model
    torch.save(trained_model.state_dict(), 'trial_TL2_augmented2.pth')

    # Load the model architecture
    model = smp.Unet(
        encoder_name="resnet34",        # Choose encoder, e.g., resnet34, mobilenet_v2, efficientnet-b7, etc.
        encoder_weights="imagenet",     # Use 'imagenet' pre-trained weights for encoder initialization
        in_channels=3,                  # Model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1                       # Model output channels (number of classes in your dataset)
    ).to(config.device)

    # Load the saved state dictionary
    model.load_state_dict(torch.load('trial_TL2_augmented2.pth', map_location=config.device))

    # Evaluate the model
    evaluate_model(model, test_loader, config.device)

if __name__ == "__main__":
    main()