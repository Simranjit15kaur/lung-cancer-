import torch
import torch.nn as nn


class ImprovedTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.layer_norm_input = nn.LayerNorm(d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.layer_norm_input(x)
        x = x.unsqueeze(1) + self.pos_encoding
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class HybridEnsemble:
    """
    Container class for the trained hybrid ensemble.
    Holds references to trained models and scalers.
    """

    def __init__(self, input_dim, num_classes=3, device="cpu"):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device

        # These attributes are populated when loading the .pkl
        self.models = {}
        self.scalers = {}
        self.meta_model = None
        self.label_mapping = None
