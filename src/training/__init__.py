"""Public training package API."""

from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.metrics import evaluate_loader, get_predictions, predict_logits
from src.training.mixup import mixup_criterion, mixup_data
from src.training.performance import enable_performance_features, optimize_model
from src.training.trainer import Trainer, TrainingHistory

__all__ = [
    "Trainer",
    "TrainingHistory",
    "evaluate_loader",
    "get_predictions",
    "predict_logits",
    "mixup_data",
    "mixup_criterion",
    "save_checkpoint",
    "load_checkpoint",
    "enable_performance_features",
    "optimize_model",
]
