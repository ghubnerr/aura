from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class EmotionModel(nn.Module):
    def __init__(self, pretrained = False):
        super(EmotionModel, self).__init__()

        self.backbone = models.resnet18(pretrained = pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 8)
        self.fc = self.backbone.fc

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def embed(self, image: np.ndarray):
        image = Image.fromarray(image)
        image = self.transform(image)
        image = image.unsqueeze(0) # add batch dim
        image.to(self.device)
        return self(image)

    
    def save(self, path: str):
        torch.save(self.state_dict(), f'{path}/aura_emotion_classifier.pth')
        return True
    
    def load(self, path: str):
        self.load_state_dict(path)
        return True
    
    def train_loop(self, provider, epochs, batch_size):
        self.to(self.device)

        true_labels = []
        pred_labels = []

        self.train()
        for epoch in range(epochs):

            running_loss = 0.0
            correct = 0
            total = 0

            batch_iterator = tqdm(provider.get_next_image_batch(batch_size, False), desc=f'Epoch {epoch+1}/{epochs}', unit=' # batch')
            for batch in batch_iterator:
                images = []
                labels = []
                for image, label, _ in batch:
                    image = Image.fromarray(image)
                    image = self.transform(image)
                    images.append(image)
                    labels.append(label)

                images = torch.stack(images)
                labels = torch.tensor(labels)
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                batch_iterator.set_postfix(loss=loss.item(), accuracy=100. * correct / total)


            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            self.scheduler.step()
        return true_labels, pred_labels

    def test(self, provider):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            running_loss = 0.0
            correct_per_class = [0] * 8
            total_per_class = [0] * 8

            true_labels = []
            pred_labels = []

            for batch in provider.get_next_image_batch(len(provider.test), test=True):
                images = []
                labels = []
                for image, label, _ in batch:
                    image = Image.fromarray(image)
                    image = self.transform(image)
                    images.append(image)
                    labels.append(label)

                images = torch.stack(images)
                labels = torch.tensor(labels)

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update per-class statistics
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    total_per_class[label] += 1
                    if pred == label:
                        correct_per_class[label] += 1

            val_loss = running_loss / total
            val_acc = 100. * correct / total

            print(f"Validation Results, Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Print per-class accuracy
        for label in range(8):
            if total_per_class[label] > 0:
                acc = 100. * correct_per_class[label] / total_per_class[label]
                print(f"Label {label}: Accuracy {acc:.2f}% ({correct_per_class[label]}/{total_per_class[label]})")
            else:
                print(f"Label {label}: No samples in validation set.")
        
        return true_labels, pred_labels
