import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
import torchvision.models as models
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# dataset structure
class FireClassificationDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = os.listdir(img_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_filename = self.image_filenames[index]
        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path).convert('RGB')

        label_path = os.path.join(self.label_dir, os.path.splitext(img_filename)[0] + '.txt')
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            # Check if there's any fire object (class_id = 1) in the image
            label = 1 if any(int(line.split()[0]) == 1 for line in lines) else 0
        except FileNotFoundError:
            print(f"Warning: Label file not found for image {img_filename}. Using default label 0.")
            label = 0

        if self.transform:
            img = self.transform(img)

        return img, label #img_filename (returning it causes some problems in to_device())

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
TENSOR_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 0.001

dataset_path = './D-Fire'

transform = Compose([
    Resize((TENSOR_SIZE, TENSOR_SIZE)),
    ToTensor(),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_images_path = dataset_path + "/train/images"
train_labels_path = dataset_path + "/train/labels"

test_images_path = dataset_path + "/test/images"
test_labels_path = dataset_path + "/test/labels"

# load and split the datasets
train_dataset = FireClassificationDataset(train_images_path, train_labels_path, transform=transform)
test_dataset = FireClassificationDataset(test_images_path, test_labels_path, transform=transform)

validation_size = round(0.5 * len(test_dataset))
test_size = len(test_dataset) - validation_size

test_dataset, validation_dataset = random_split(test_dataset, [test_size, validation_size])

# setup the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = BATCH_SIZE * 2, pin_memory = True)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE * 2, pin_memory = True)

# model init
model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

# freeze all the layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two layers
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

num_classes = 2 # fire or no fire
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

losses = []
accuracies = []
# Training
for epoch in range(10):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    total_samples = 0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        running_accuracy += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

        optimizer.step()
    
    average_loss = running_loss / len(train_dataloader)
    average_accuracy = running_accuracy / total_samples
    losses.append(average_loss)
    accuracies.append(average_accuracy)

    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_dataloader)))


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracies) + 1), accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()

torch.save(model, "pretrained_model.pth")

# Validation
model.eval()
val_predictions = []
val_targets = []
with torch.no_grad():
    for inputs, labels in validation_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        val_predictions.extend(predicted.cpu().numpy())
        val_targets.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_targets, val_predictions) * 100
    print(f'Validation accuracy: {val_acc:.3f}%')


# testing
model.eval()
test_predictions = []
test_targets = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())
    test_acc = accuracy_score(test_targets, test_predictions) * 100
    print(f'Test accuracy: {test_acc:.3f}%')

    # Calculate F1 score, precision and recall
    f1 = f1_score(test_targets, test_predictions)
    precision = precision_score(test_targets, test_predictions)
    recall = recall_score(test_targets, test_predictions)

    print(f'F1 Score: {f1:.3f}%')
    print(f'Precision: {precision:.3f}%')
    print(f'Recall: {recall:.3f}%')

    # Compute confusion matrix for test set
    cm = confusion_matrix(test_targets, test_predictions)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(2, 2))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=np.arange(2),
           yticklabels=np.arange(2),
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.show()
