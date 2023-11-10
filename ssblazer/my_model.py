from lightning.pytorch import LightningModule
import torch
import numpy as np
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        decay = self.warmup / self.max_iters

        if epoch <= self.warmup:
            lr_factor = 1 * (epoch / self.warmup)
        else:
            lr_factor = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (
                        (epoch - (decay * self.max_iters))
                        / ((1 - decay) * self.max_iters)
                    )
                )
            )

        return lr_factor


# For residual block
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False
    )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out


class SSBlazer(LightningModule):
    def __init__(
        self,
        warmup,
        max_epochs,
        lr,
        *args,
        **kwargs,
    ) -> None:
        super(SSBlazer, self).__init__(*args, **kwargs)

        # Inception
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(),
        )
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(),
        )
        self.bidirectional = nn.LSTM(
            input_size=28,
            hidden_size=12,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.self_att = nn.TransformerEncoderLayer(
            d_model=28, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.1
        )

        # Preprocess before resnet
        self.preprocess = nn.Sequential(
            conv3x3(4, 8), nn.BatchNorm2d(8), nn.ELU(inplace=False)
        )

        # Res block
        self.resnet = nn.Sequential(
            self.make_layer(ResidualBlock, in_channels=8, out_channels=16, blocks=15),
            self.make_layer(ResidualBlock, in_channels=16, out_channels=32, blocks=15),
        )

        self.pooling2d = nn.AvgPool2d(kernel_size=(5, 1), stride=3)

        self.self_att_seq = nn.TransformerEncoderLayer(
            d_model=32, nhead=8, dim_feedforward=512, batch_first=True, dropout=0.1
        )

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(2908, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 20),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.main_output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

        self.warmup = warmup
        self.max_epochs = max_epochs
        self.lr = lr

        # Set up metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        # self.train_auroc=BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, batch):
        input = batch["seq"]

        center = input.shape[1]
        start = center // 2 - 4
        end = start + 9

        cutted = input[:, start:end, :]
        cutted = cutted.permute(0, 2, 1)
        cutted = cutted.unsqueeze(3)

        c1 = self.conv2d_1(cutted)
        c2 = self.conv2d_2(cutted)
        c3 = self.conv2d_3(cutted)
        inception_concat = torch.cat((cutted, c1, c2, c3), 1).squeeze(3)
        inception_concat = inception_concat.permute(0, 2, 1)

        center_atten = self.self_att(inception_concat)

        center_flatten = self.flatten(center_atten)

        input = input.permute(0, 2, 1)
        input = input.unsqueeze(3)

        preprocessed_seq = self.preprocess(input)

        emb_seq = self.resnet(preprocessed_seq)
        emb_seq = self.pooling2d(emb_seq).squeeze(3)
        emb_seq = emb_seq.permute(0, 2, 1)
        emb_seq = self.self_att_seq(emb_seq)
        emb_seq_flatten = self.flatten(emb_seq)

        concat = torch.cat((emb_seq_flatten, center_flatten), 1).squeeze(1)
        dense_out = self.dense(concat)
        logits = self.main_output(dense_out).squeeze(1)

        preds = self.sigmoid(logits)
        return logits, preds

    def training_step(self, batch, batch_idx=None, dataloader_idx=None):
        if batch_idx == 0:
            rank_zero_info("Train batch sample: {}".format(str(batch)))
        logits, preds = self.forward(batch)
        targets = batch["label"]
        loss = self.BCEWithLogitsLoss(logits, targets)

        self.train_acc(preds, targets)

        self.log(
            "train/BCELoss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "train/Accuracy",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return loss

    def validation_step(self, batch, batch_idx=None, dataloader_idx=None):
        logits, preds = self.forward(batch)
        targets = batch["label"]
        loss = self.BCEWithLogitsLoss(logits, targets)

        self.val_acc(preds, targets)
        self.val_auroc(preds, targets.int())
        self.val_auprc(preds, targets.int())

        self.log(
            "val/BCELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "val/Accuracy",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "val/AUROC",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "val/AUPRC",
            self.val_auprc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        rank_zero_info(str(self.trainer.callback_metrics))

    def on_before_optimizer_step(self, optimizer):
        for p in self.parameters():
            if p.grad is not None:
                valid_gradients = not (
                    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                )

        if not valid_gradients:
            rank_zero_info(
                f"detected inf or nan values in gradients. not updating model parameters"
            )
            optimizer.zero_grad()

    def configure_optimizers(self):
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), self.lr)

        # Apply learning rate scheduler per epoch.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup, max_iters=self.max_epochs
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}
