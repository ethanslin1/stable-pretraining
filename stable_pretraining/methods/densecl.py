import copy

import torch
import torch.nn as nn
import lightning as L

from stable_pretraining.forward import densecl


class DenseCL(L.LightningModule):
    """Dense Contrastive Learning for self-supervised representation learning."""

    def __init__(
        self,
        backbone,
        feature_dim,
        projection_dim=128,
        temperature=0.2,
        dense_loss_weight=0.5,
        momentum=0.999,
        learning_rate=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.backbone = backbone
        self.momentum_backbone = copy.deepcopy(backbone)

        # Freeze the target encoder; update it by EMA.
        for parameter in self.momentum_backbone.parameters():
            parameter.requires_grad = False

        self.global_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

        self.local_projector = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, projection_dim, kernel_size=1),
        )

        self.temperature = temperature
        self.dense_loss_weight = dense_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @torch.no_grad()
    def update_momentum_encoder(self):
        """Exponential moving average update of the target backbone."""
        for online, target in zip(
            self.backbone.parameters(),
            self.momentum_backbone.parameters(),
        ):
            target.data.mul_(self.momentum).add_(online.data, alpha=1 - self.momentum)

    def forward(self, batch, stage="train"):
        if stage == "train":
            self.update_momentum_encoder()

        return densecl(self, batch, stage)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
