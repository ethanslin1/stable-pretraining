# import torch
# import torchmetrics
# import lightning.pytorch as pl
# import stable_pretraining as spt

# from stable_pretraining.data.densecl_data import make_densecl_cifar10

# train_dataloader, val_dataloader = make_densecl_cifar10()

# datamodule = spt.data.DataModule(
#     train=train_dataloader,
#     val=val_dataloader,
# )

# linear_probe = spt.OnlineProbe(
#     datamodule,
#     name="linear_probe",
#     input="embedding",
#     target="label",
#     probe=torch.nn.Linear(2048, 10),
#     loss=torch.nn.CrossEntropyLoss(),
#     metrics={
#         "top1": torchmetrics.classification.MulticlassAccuracy(10),
#     },
# )


# trainer = pl.Trainer(
#     max_epochs=100,
#     accelerator="auto",
#     devices="auto",
#     callbacks=[linear_probe],
# )

# manager = spt.Manager(
#     trainer=trainer,
#     module=module,
#     data=datamodule,
# )

# manager()

import torch
import torch.nn as nn
import torchmetrics
import lightning.pytorch as pl
import stable_pretraining as spt

from stable_pretraining.data.densecl_data import make_densecl_cifar10
from stable_pretraining.methods.densecl import DenseCL
from stable_pretraining.manager import Manager
from lightning.pytorch.loggers import CSVLogger

SMOKE_TEST = True  # True = one tiny CPU pass to check wiring. Will set to False when running true training.
FEATURE_DIM = 512  # ResNet-18 output channels (2048 for ResNet-50)


train_dataloader, val_dataloader = make_densecl_cifar10()

datamodule = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

resnet = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc

with torch.no_grad():
    dummy = backbone(torch.zeros(2, 3, 32, 32))
assert dummy.ndim == 4 and dummy.shape[1] == FEATURE_DIM, (
    dummy.shape
)  # expect (2, 512, 4, 4)
backbone.train()

module = DenseCL(
    backbone=backbone,
    feature_dim=FEATURE_DIM,
    projection_dim=128,
    temperature=0.2,
    dense_loss_weight=0.5,
    learning_rate=1e-3,
    momentum=0.999,
    weight_decay=1e-4,
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(FEATURE_DIM, 10),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    input_dim=FEATURE_DIM,  # 512 for ResNet-18
    num_classes=10,
    k=10,
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
    },
)

logger = CSVLogger(
    save_dir="logs",
    name="densecl_cifar10",
)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    callbacks=[linear_probe, knn_probe],
    fast_dev_run=SMOKE_TEST,
    precision="16-mixed",
    logger=logger,
)

manager = Manager(
    trainer=trainer,
    module=module,
    data=datamodule,
)

print("Manager object:", type(manager))
print("Is manager object callable:", callable(manager))

manager()
