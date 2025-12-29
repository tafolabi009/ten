"""
Learning Rate Schedulers
========================

Implements cosine annealing with warmup (Appendix B.3).
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineWithWarmupScheduler(LambdaLR):
    """
    Cosine learning rate schedule with warmup.
    
    Reference: Appendix B.3 - "cosine annealing"
    
    lr(t) = {
        lr_max * t / warmup_steps,                           if t < warmup_steps
        lr_max * 0.5 * (1 + cos(π * (t - warmup) / (T - warmup))),  otherwise
    }
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    
    def lr_lambda(self, current_step: int) -> float:
        """Compute learning rate multiplier."""
        if current_step < self.num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.num_warmup_steps))
        
        # Cosine annealing
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        
        return max(
            self.min_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress))
        )


class LinearWithWarmupScheduler(LambdaLR):
    """
    Linear learning rate decay with warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    
    def lr_lambda(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        
        return max(
            0.0,
            float(self.num_training_steps - current_step) / 
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        )


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with proper weight decay handling.
    
    Reference: Appendix B.3
    """
    # Separate weight decay for different parameter types
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't apply weight decay to biases, layer norms, and embeddings
        if any(nd in name.lower() for nd in ['bias', 'norm', 'embedding', 'alpha', 'omega']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
    )


def get_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create learning rate scheduler.
    
    Args:
        scheduler_type: "cosine" or "linear"
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return CosineWithWarmupScheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "linear":
        return LinearWithWarmupScheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
