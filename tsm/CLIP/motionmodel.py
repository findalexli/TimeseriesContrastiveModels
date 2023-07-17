import timm
from torch import nn

import torch
import transformers
from typing import Optional
import torch
from torch import nn
import math
from torch import optim
import itertools
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

class MotionCNNEncoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 2048):
        super().__init__()
        repo = 'OxWearables/ssl-wearables'
        self.pretrained_model = torch.hub.load(repo, 'harnet10', class_num=5, pretrained=True)

    def forward(self, x):
        reshaped = torch.reshape(x, (len(x), 3, -1))
        return torch.squeeze(self.pretrained_model.feature_extractor(reshaped), -1)

class MotionTFEncoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 2048):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.embedding = nn.Linear(3, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.target_token_idx = 0

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects inputs in the shape (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        return x[:, self.target_token_idx, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.tensor(10000.0).log() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class CLIPDualEncoderModel(LightningModule):
    def __init__(
        self,
        text_encoder_alias: str,
        motion_encoder: nn.Module,
        text_encoder_trainable: bool = True,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        head_lr: float = 1e-3,
        motion_encoder_lr: float = 1e-4,  # Add motion_encoder_lr attribute
        text_encoder_lr: float = 1e-5,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )
        self.motion_encoder = motion_encoder
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.motion_projection = ProjectionHead(
            embedding_dim=1024,  # Modify this to match the output size of the motion encoder
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.motion_encoder_lr = motion_encoder_lr  # Assign motion_encoder_lr attribute
        self.text_encoder_lr = text_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        self.save_hyperparameters()

    def _compute_losses(self, text_embeddings, motion_embeddings):
        logits = (text_embeddings @ motion_embeddings.T) / self.temperature
        similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(similarity / self.temperature, dim=-1)
        loss = (-targets * self.log_softmax(logits)).sum(1)
        return loss

    def forward(self, inputs):
        text_features = self.text_encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        motion_features = self.motion_encoder(inputs["motion"])

        text_embeddings = self.text_projection(text_features)
        motion_embeddings = self.motion_projection(motion_features)

        return text_embeddings, motion_embeddings

    def configure_optimizers(self):
        parameters = [
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {"params": self.motion_encoder.parameters(), "lr": self.motion_encoder_lr},
            {
                "params": itertools.chain(
                    self.text_projection.parameters(),
                    self.motion_projection.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
        )
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    def training_step(self, batch, *args, **kwargs):
        text_embeddings, motion_embeddings = self.forward(batch)
        loss = self._compute_losses(text_embeddings, motion_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        return loss

    def validation_step(self, batch, *args, **kwargs):
        text_embeddings, motion_embeddings = self.forward(batch)
        loss = self._compute_losses(text_embeddings, motion_embeddings).mean()
        val_loss = self.all_gather(loss)
        self.log("val/loss", val_loss.mean())
        return loss
    
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()


        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)


        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)


        x += projected


        return self.layer_norm(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()


        self.model = transformers.AutoModel.from_pretrained(model_name)


        for param in self.model.parameters():
            param.requires_grad = trainable


        self.target_token_idx = 0


    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state


        return last_hidden_state[:, self.target_token_idx, :]



# Generate dummy data
batch_size = 4
seq_len = 300
num_features = 3
dummy_data = torch.randn(batch_size, seq_len, num_features)

# # Create MotionEncoder instance
motion_encoder = MotionCNNEncoder()
motion_encoder.parameters()

print(motion_encoder(dummy_data).shape)
# # Test forward pass
# output = motion_encoder(dummy_data)
# print(output.shape)
# class ImageEncoder(nn.Module):
#     def __init__(
#         self, model_name: str, pretrained: bool = True, trainable: bool = True
#     ) -> None:
#         super().__init__()


#         self.model = timm.create_model(
#             model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
#         )


#         for param in self.model.parameters():
#             param.requires_grad = trainable


#         self.target_token_idx = 0


#     def forward(self, x):
#         return self.model(x)