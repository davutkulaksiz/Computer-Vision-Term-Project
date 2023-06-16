import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

class FIREDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.fire_dir = os.path.join(root_dir, "fire_images")
        self.nonfire_dir = os.path.join(root_dir, "non_fire_images")
        self.fire_images = os.listdir(self.fire_dir)
        self.nonfire_images = os.listdir(self.nonfire_dir)
        self.transform = transform

    def __len__(self):
        return len(self.fire_images) + len(self.nonfire_images)

    def __getitem__(self, index):
        if index < len(self.fire_images):
            image_name = self.fire_images[index]
            image_path = os.path.join(self.fire_dir, image_name)
            label = 1  # Assign label 1 for fire images
        else:
            index -= len(self.fire_images)
            image_name = self.nonfire_images[index]
            image_path = os.path.join(self.nonfire_dir, image_name)
            label = 0  # Assign label 0 for non-fire images

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
TENSOR_SIZE = 128
BATCH_SIZE = 32

# Load the model
model = torch.load("pretrained_model.pth")
model.eval()

# define transform
transform = Compose([
    Resize((TENSOR_SIZE, TENSOR_SIZE)),
    ToTensor(),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load dataset
dataset_path = './fire_dataset'

dataset = FIREDataset(dataset_path, transform=transform)

dataset_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True)


predictions = []
targets = []
with torch.no_grad():
    for inputs, labels in dataset_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    test_acc = accuracy_score(targets, predictions) * 100
    print(f'Test accuracy: {test_acc:.3f}%')

    # Calculate F1 score, precision and recall
    f1 = f1_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)

    print(f'F1 Score: {f1:.3f}%')
    print(f'Precision: {precision:.3f}%')
    print(f'Recall: {recall:.3f}%')

    # Compute confusion matrix for test set
    cm = confusion_matrix(targets, predictions)

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

