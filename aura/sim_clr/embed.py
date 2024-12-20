import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimCLR:
    def __init__(self, feature_dim=128, temperature=0.5):
        """
        Initialize SimCLR model
        
        Args:
            feature_dim (int): Dimensionality of embedding space
            temperature (float): Temperature scaling for contrastive loss
        """
        self.temperature = temperature
        
        # CNN Encoder with projection head
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Flatten
            nn.Flatten(),
            
            # Projection Head
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def embed(self, x):
        """
        Generate embedding for input tensor
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Embedding vector
        """
        with torch.no_grad():
            return self.model(x.to(self.device))

    def contrastive_loss(self, z1, z2):
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        
        Args:
            z1 (torch.Tensor): First set of embeddings
            z2 (torch.Tensor): Second set of embeddings
        
        Returns:
            torch.Tensor: Contrastive loss
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature
        
        # Remove diagonal elements (self-similarity)
        sim_matrix = sim_matrix[~torch.eye(2 * batch_size, dtype=bool, device=self.device)]
        sim_matrix = sim_matrix.view(2 * batch_size, -1)
        
        # Labels for contrastive loss
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(self.device)
        
        return F.cross_entropy(sim_matrix, labels)

    def train(self, dataloader, optimizer, epochs=10):
        """
        Train SimCLR model
        
        Args:
            dataloader (torch.utils.data.DataLoader): Training data loader
            optimizer (torch.optim.Optimizer): Optimizer
            epochs (int): Number of training epochs
        """
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for original, augmentations in dataloader:
                original = original.to(self.device)
                
                # Compute embeddings for original and augmentations
                z_original = self.model(original)
                aug_embeddings = [self.model(aug.to(self.device)) for aug_list in augmentations for aug in aug_list]
                
                # Compute contrastive losses
                losses = [self.contrastive_loss(z_original, z_aug) for z_aug in aug_embeddings]
                loss = torch.mean(torch.stack(losses))
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    def save(self, path):
        """
        Save model weights
        
        Args:
            path (str): File path to save weights
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Load model weights
        
        Args:
            path (str): File path to load weights from
        """
        self.model.load_state_dict(torch.load(path))