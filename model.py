import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict

# ========================
# Path Setup
# ========================
base_dir = r"C:\\Users\\sanyu\\PycharmProjects\\PythonProject\\Dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# ========================
# Hyperparameters
# ========================
IMG_SIZE = 240  # for EfficientNetB3
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Data Transformations
# ========================
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================
# Data Loaders
# ========================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

# Limit to 100 samples per class
max_images_per_class = 100
class_counts = defaultdict(int)
selected_indices = []

for idx, (_, label) in enumerate(train_dataset.samples):
    if class_counts[label] < max_images_per_class:
        selected_indices.append(idx)
        class_counts[label] += 1

train_dataset = Subset(train_dataset, selected_indices)

val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# Model Setup (EfficientNetB3)
# ========================
model = models.efficientnet_b3(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(datasets.ImageFolder(train_dir).classes))
)

# Fine-tune last few layers
for param in model.features[6:].parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ========================
# Loss, Optimizer, Scheduler
# ========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ========================
# Training & Validation Loop
# ========================
train_acc, val_acc, train_loss, val_loss = [], [], [], []
best_val_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct.double() / len(train_loader.dataset)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc.item())

    model.eval()
    val_running_loss = 0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct.double() / len(val_loader.dataset)
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc.item())

    print(f"Epoch {epoch+1}: Train Acc={epoch_acc:.4f}, Val Acc={val_epoch_acc:.4f}")

    scheduler.step(val_epoch_acc)

    if val_epoch_acc > best_val_acc:
        torch.save(model.state_dict(), "best_model.pth")
        best_val_acc = val_epoch_acc
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 5:
        print("Early stopping triggered.")
        break

# ========================
# Plot Training History
# ========================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# ========================
# Evaluate on Test Set
# ========================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)

test_accuracy = test_correct.double() / len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")

# ========================
# Save Final Model
# ========================
torch.save(model.state_dict(), "efficientnet_dog_classifier_final.pth")
print("Model saved as efficientnet_dog_classifier_final.pth")

