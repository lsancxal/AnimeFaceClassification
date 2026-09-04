"""Training loop for AnimeCNN."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINT_PATH,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    MIXUP_ALPHA,
    NUM_EPOCHS,
    USE_TTA_DURING_TRAINING,
    WEIGHT_DECAY,
)
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.metrics import evaluate_loader
from src.training.mixup import mixup_criterion, mixup_data
from src.training.performance import (
    autocast_context,
    create_grad_scaler,
    to_device,
)


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = -1.0


class Trainer:
    """Owns optimizer, scheduler, checkpointing, and the epoch loop."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device = DEVICE,
        num_epochs: int = NUM_EPOCHS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        label_smoothing: float = LABEL_SMOOTHING,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        checkpoint_path: str = CHECKPOINT_PATH,
        mixup_alpha: float = MIXUP_ALPHA,
        use_tta_during_training: bool = USE_TTA_DURING_TRAINING,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_path = checkpoint_path
        self.mixup_alpha = mixup_alpha
        self.use_tta_during_training = use_tta_during_training
        self.scaler = create_grad_scaler()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer_kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        if device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=MIN_LEARNING_RATE,
        )

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = to_device(inputs, self.device)
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")

            self.optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                if self.mixup_alpha > 0:
                    inputs, labels_a, labels_b, lam = mixup_data(
                        inputs, labels, alpha=self.mixup_alpha
                    )
                    outputs = self.model(inputs)
                    loss = mixup_criterion(
                        self.criterion, outputs, labels_a, labels_b, lam
                    )
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item()

        return running_loss / max(len(train_loader), 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        history = TrainingHistory()
        epochs_without_improvement = 0

        if self.checkpoint_path:
            os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)

        total_start = time.perf_counter()

        for epoch in range(self.num_epochs):
            epoch_start = time.perf_counter()

            train_loss = self.train_one_epoch(train_loader)
            val_loss, accuracy = evaluate_loader(
                self.model,
                val_loader,
                self.criterion,
                device=self.device,
                use_tta=self.use_tta_during_training,
            )

            epoch_time = time.perf_counter() - epoch_start
            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.accuracies.append(accuracy)
            history.epoch_times.append(epoch_time)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            improved = accuracy > history.best_val_accuracy
            if improved:
                history.best_val_accuracy = accuracy
                history.best_epoch = epoch + 1
                epochs_without_improvement = 0
                if self.checkpoint_path:
                    save_checkpoint(
                        self.checkpoint_path,
                        self.model,
                        epoch=history.best_epoch,
                        val_accuracy=history.best_val_accuracy,
                        val_loss=val_loss,
                    )
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {100 * accuracy:.2f}%, "
                f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s"
                + (" *" if improved else "")
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no val accuracy improvement for "
                    f"{self.early_stopping_patience} epochs)"
                )
                break

        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            checkpoint = load_checkpoint(
                self.checkpoint_path, self.model, device=self.device
            )
            print(
                f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
                f"(val accuracy: {100 * checkpoint['val_accuracy']:.2f}%)"
            )

        total_time = time.perf_counter() - total_start
        avg_epoch_time = sum(history.epoch_times) / max(len(history.epoch_times), 1)
        print(f"Training time: {total_time:.2f} seconds")
        print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        print(
            f"Best val accuracy: {100 * history.best_val_accuracy:.2f}% "
            f"at epoch {history.best_epoch}"
        )
        return history


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    **kwargs,
) -> tuple[list[float], list[float], list[float]]:
    """Backward-compatible wrapper around :class:`Trainer`."""
    history = Trainer(model, **kwargs).fit(train_loader, val_loader)
    return history.train_losses, history.val_losses, history.accuracies
