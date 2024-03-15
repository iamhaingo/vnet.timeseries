from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

# Generate fake data
num_samples = 1000
window_size = 80
num_features = 14
num_classes = 7

# Generate random input data (acceleration in our case)
X_train = torch.randn(num_samples, window_size, num_features)
print(f"X_train shape: {X_train.shape}")

# Generate random labels
y_train = torch.randint(0, num_classes, (num_samples,))
print(f"y_train shape: {y_train.shape}")

# Convert to TensorDataset
dataset = TensorDataset(X_train, y_train)

# Define the size of the validation set (20% des données)
val_size = int(0.2 * num_samples)
train_size = num_samples - val_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
