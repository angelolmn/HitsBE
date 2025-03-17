import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from aeon.datasets import load_classification
from torch.utils.data import DataLoader, TensorDataset

from hitsbe.models.hitsBERT import HitsBERTClassifier


# Hyperparameters
batch_size = 8
sequence_length = 128  
num_classes = 3  # Based on dataset labels
hidden_size = 768

# Load the dataset
X_train, y_train = load_classification("ArrowHead", split="train")
X_test, y_test = load_classification("ArrowHead", split="test")

print(f"Number of training instances: {len(X_train)}")
print(f"Number of classes: {set(y_train)}")

# Remove unnecessary dimensions 
X_train_shaped = np.squeeze(X_train, axis=1)
X_test_shaped  = np.squeeze(X_test, axis=1)

# Convert class labels to integers
y_train = np.array([int(x) for x in y_train])
y_test = np.array([int(x) for x in y_test])

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_shaped, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test_shaped, dtype=torch.float)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

# Define device with cuda
if not torch.cuda.is_available():
    sys.exit("ERROR: No GPU detected.")

device = torch.device("cuda")

# Initialize the model and move it to the device
model = HitsBERTClassifier(batch_size=batch_size, sequence_length=sequence_length, num_classes=num_classes).to(device)

# Freeze all layers of the BERT model
for param in model.bert.bert.parameters():
    param.requires_grad = False

# Unfreeze the first four layers of the BERT encoder for partial training
for layer in model.bert.bert.encoder.layer[:4]:
    for param in layer.parameters():
        param.requires_grad = True

# Create DataLoader objects for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------ Training ------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
num_epochs = 20

# Training mode
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()  # Reset gradients
        logits = model(batch_X)  # Forward pass, PyTorch creates a computational graph
        loss = criterion(logits, batch_y)  # Compute loss
        loss.backward()  # Backpropagation. Loss is connected with the model through the computational graph
        optimizer.step()  # Update model parameters. The optimizer knows the parameters by its definition

        total_loss += loss.item() * batch_X.size(0)  # Accumulate batch loss
    
    avg_loss = total_loss / len(train_dataset)  # Compute average loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ------ Evaluation ------
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():  # Disable gradient computation for inference
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)  # Forward pass
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)  # Get predicted class labels

        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())

# Compute overall accuracy
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
accuracy = np.mean(all_preds == all_labels)
print(f"Test Accuracy: {accuracy:.4f}")
