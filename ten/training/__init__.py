"""
TEN Training Module
==================

Training utilities, data loading, and optimization.
"""

from ten.training.trainer import TENTrainer, TrainingArguments
from ten.training.data import (
    WikiTextDataset,
    OpenWebTextDataset,
    LongRangeArenaDataset,
    TextDataset,  # Alias for WikiTextDataset
    get_wikitext_dataloader,
    get_lra_dataloader,
)
from ten.training.scheduler import (
    CosineWithWarmupScheduler,
    get_optimizer,
    get_scheduler,
)

__all__ = [
    "TENTrainer",
    "TrainingArguments",
    "WikiTextDataset",
    "OpenWebTextDataset",
    "LongRangeArenaDataset",
    "TextDataset",
    "get_wikitext_dataloader",
    "get_lra_dataloader",
    "CosineWithWarmupScheduler",
    "get_optimizer",
    "get_scheduler",
]
