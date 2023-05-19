import torch.nn as nn
import torch.nn.functional as F


TENSOR_SIZE = 128


class ImageClassificationModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        # Generate predictions
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        # Calculate accuracy
        acc = accuracy(out, labels)
        return {'validation_loss': loss.detach(), 'validation_accuracy': acc} 

    def validation_epoch_end(self, outputs):
        batch_losses = [x['validation_loss'] for x in outputs]
        # Combine losses
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['validation_accuracy'] for x in outputs]
        # Combine accuracies
        epoch_acc = torch.stack(batch_accs).mean()
        return {'validation_loss': epoch_loss.item(), 'validation_accuracy': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        train_loss, val_loss, val_acc = result['train_loss'], result['validation_loss'], result['validation_accuracy']
        print(f"Epoch [{epoch}/{EPOCH_NUMBER}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class CNNModel(ImageClassificationModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * int(TENSOR_SIZE / 2) * int(TENSOR_SIZE / 2), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
        
    def forward(self, xb):
        return self.network(xb)