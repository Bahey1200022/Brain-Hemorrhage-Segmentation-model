# Configuration settings for the project

import torch
batch_size = 8
learning_rate = 1e-3
num_epochs = 100
train_dir = r'C:\Users\moham\OneDrive\Desktop\test\dataset'
val_dir = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
