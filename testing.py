import os
import torch
import numpy as np
import cv2
import config
from model import UNet
from torchvision import transforms
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

def load_model(model_path, device):
    model = smp.Unet(
        encoder_name="mobilenet_v2",        # Choose encoder, e.g., resnet34, mobilenet_v2, efficientnet-b7, etc.
        encoder_weights="imagenet",     # Use 'imagenet' pre-trained weights for encoder initialization
        in_channels=3,                  # Model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1                       # Model output channels (number of classes in your dataset)
    ).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(image, (256, 256))  # Resize for display purposes
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image, original_image

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    original_mask = cv2.resize(mask, (256, 256))  # Resize for display purposes
    return original_mask

def segment_image(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()
        output = (output > 0.5).astype(np.uint8)
    return output

def save_segmentation(output_mask, output_path):
    plt.imsave(output_path, output_mask, cmap='gray')

def display_images(original_image, original_mask, output_mask):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(original_mask, cmap='gray')
    axes[1].set_title('Original Mask')
    axes[1].axis('off')
    
    axes[2].imshow(output_mask, cmap='gray')
    axes[2].set_title('Segmented Image')
    axes[2].axis('off')
    
    plt.show()

def process_folder(model, input_folder, output_folder, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Preprocess the input image
            image_tensor, original_image = preprocess_image(image_path)
            
            # Perform segmentation
            output_mask = segment_image(model, image_tensor, device)
            
            # Save the segmentation result
            save_segmentation(output_mask, output_path)
            
            # Optionally display the images
            # display_images(original_image, None, output_mask)

if __name__ == "__main__":
    model_path = 'trial_TL3_mobilenet.pth'
    input_folder = r'C:\Users\moham\OneDrive\Desktop\test\dataset\png_volumes\ID_0c3eef60_ID_6994ad7df0.nii'
    output_folder = r'C:\Users\moham\OneDrive\Desktop\test\dataset\predicted_masks'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = load_model(model_path, device)

    # Process the folder of images
    process_folder(model, input_folder, output_folder, device)