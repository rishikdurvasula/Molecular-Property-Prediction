import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

# Load dataset
dataset = QM9(root='./data')

# Split dataset into train, validation, and test sets
train_dataset = dataset[:100000]  # 100k for training
val_dataset = dataset[100000:110000]  # 10k for validation
test_dataset = dataset[110000:]  # Remaining for testing

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
