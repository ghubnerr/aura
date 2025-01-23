import clip
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from aura.cnn import EmotionModel

class CLIPImageEncoder(nn.Module):
    def __init__(self, projection_dim=128, device='cuda'):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.device = device
        self.projection = nn.Sequential(
            nn.Linear(512, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            image_features = image_features.float()
        return self.projection(image_features)

class EmotionEncoderWithProjection(nn.Module):    
    def __init__(self, projection_dim=128, device='cuda', emotion_model_path="/a/buffalo.cs.fiu.edu./disk/jccl-002/homes/glucc002/Desktop/Projects/aura/aura/cnn/checkpoints/aura_emotion_classifier.pth"):
        super().__init__()
        self.base_encoder = EmotionModel(pretrained=False).to(device)
        self.base_encoder.load(emotion_model_path)
        self.device = device
        
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def forward(self, x):
        batch_embeddings = []
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            # If batch dimension is present, process each image
            if len(x.shape) == 4:
                for img in x:
                    embedding = self.base_encoder.embed(img, use_representation=True)
                    batch_embeddings.append(embedding)
                embeddings = np.stack(batch_embeddings)
            else:
                embeddings = self.base_encoder.embed(x, use_representation=True)
                embeddings = np.expand_dims(embeddings, 0)
        
        embeddings = torch.from_numpy(embeddings).float().to(self.device)
        return self.projection_head(embeddings)