import torch
from torch import nn
import clip


class Projector(nn.Module):
    def __init__(self, in_dim=512, reduction=4):
        super().__init__()
        self.projector_image = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction),
            nn.BatchNorm1d(in_dim // reduction),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
        )

    def forward(self, image):
        return self.projector_image(image)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.mlp = nn.Sequential(
            # nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.mlp(x)
        return x

    def forward_embedding(self, x):
        x = self.fc1(x)
        embedding = torch.nn.functional.normalize(x, p=2, dim=-1)
        return embedding


class CLIP_with_text(nn.Module):
    def __init__(self, clip_model, mlp_model, norm_dim):
        super().__init__()
        self.clip_model = clip_model
        self.norm_layer = nn.BatchNorm1d(norm_dim)
        self.mlp_model = mlp_model

    def forward(self, image, text):
        features = self.clip_model.encode_image(image)
        new_features = features @ text.T
        # new_features = self.norm_layer(new_features)
        # new_features = new_features - torch.mean(new_features)
        # new_features = new_features / torch.std(new_features, dim=1, keepdim=True)
        new_features = new_features / new_features.norm(dim=-1, keepdim=True)
        logits = self.mlp_model(new_features)
        return logits


class CLIP_no_text(nn.Module):
    def __init__(self, clip_model, mlp_model):
        super().__init__()
        self.clip_model = clip_model
        self.mlp_model = mlp_model

    def forward(self, images):
        features = self.clip_model.encode_image(images)
        logits = self.mlp_model(features)
        return logits





