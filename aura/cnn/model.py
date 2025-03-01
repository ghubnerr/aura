import os

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
    """Deep learning model for emotion classification using ResNet50 backbone.
    
    A CNN-based model that performs emotion classification across 8 categories using 
    a modified ResNet50 architecture. Supports both inference and feature extraction.
    
    Args:
        pretrained (bool): Whether to initialize with pretrained ResNet50 weights
    
    Attributes:
        backbone (nn.Module): ResNet50 model backbone
        fc (nn.Linear): Final fully connected layer (num_features -> 8)
        criterion (nn.Module): Loss function (CrossEntropyLoss)
        optimizer (optim.Optimizer): Adam optimizer
        scheduler (optim.lr_scheduler): Learning rate scheduler
        transform (transforms.Compose): Image preprocessing pipeline
        device (torch.device): Device to run the model on (CPU/GPU)
    """

    def __init__(self, pretrained=False):
        super(EmotionModel, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
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
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            torch.Tensor: Model predictions of shape (batch_size, 8)
        """
        return self.backbone(x)
    
    def embed(self, image, use_representation=False, skip_transform=False):
        """Extract features or predictions from an input image.
        
        Args:
            image (Union[np.ndarray, torch.Tensor]): Input image in one of these formats:
                - np.ndarray: RGB image of shape (H, W, 3)
                - torch.Tensor: RGB tensor of shape (3, H, W) or (1, 3, H, W)
            use_representation (bool): If True, return GAP features instead of predictions
            skip_transform (bool): If True, skip image preprocessing steps
        
        Returns:
            Union[np.ndarray, torch.Tensor]: One of:
                - np.ndarray: GAP features of shape (2048,) if use_representation=True
                - torch.Tensor: Class predictions of shape (1, 8) if use_representation=False
        
        Notes:
            When use_representation=True, extracts features from the global average pooling
            layer (2048-dimensional vector). Otherwise, returns model's class predictions.
        """
        if not skip_transform:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            transformed_image = self.transform(image)
            transformed_image = transformed_image.unsqueeze(0).to(self.device)
        else:
            if isinstance(image, np.ndarray):
                transformed_image = torch.from_numpy(image)
            elif isinstance(image, torch.Tensor):
                transformed_image = image
            
            if transformed_image.dim() == 3:
                transformed_image = transformed_image.unsqueeze(0)
            transformed_image = transformed_image.to(self.device)

        if use_representation:
            def hook(module, input, output):
                self.representation = output

            handle = self.backbone.avgpool.register_forward_hook(hook)
            self.eval()
            with torch.no_grad():
                self(transformed_image)
            handle.remove()

            gap_representation = self.representation.squeeze().cpu().detach().numpy()
            return gap_representation
        else:
            return self(transformed_image)
        

    def save(self, path = None):
        if path:
            torch.save(self.state_dict(), path)
            return True
        save_dir = os.path.join(os.environ.get("STORAGE_PATH"), "aura_storage", "aura_emotion_classifier.pth")
        torch.save(self.state_dict(), save_dir)
        return True
    
    def load(self, path: str):
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict)
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

            batch_iterator = tqdm(provider.get_next_image_batch(batch_size, source = "train"), desc=f'Epoch {epoch+1}/{epochs}', unit=' # batch')
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

            for batch in provider.get_next_image_batch(len(provider.test), source = "test"):
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
    
    
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model = EmotionModel(pretrained=True)
    model.to(model.device)
    model.eval()

    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    transformed_image = model.transform(test_image)
    transformed_image = transformed_image.to(model.device)

    with torch.no_grad():
        feature_vector = model.embed(transformed_image, use_representation=True, skip_transform=True)
        
        feature_tensor = torch.tensor(feature_vector, device=model.device, dtype=torch.float32)
        logits_from_embed = model.fc(feature_tensor).cpu().numpy()
        logits_direct = model(transformed_image.unsqueeze(0)).cpu().numpy()

    print("Logits from embed method and final layer:", logits_from_embed)
    print("Logits from direct model call:", logits_direct)

    assert np.allclose(logits_from_embed, logits_direct, atol=1e-6), "Outputs do not match!"
    print("Test passed: Outputs from both methods are the same.")
