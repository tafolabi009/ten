"""
TEN Trainer
===========

Training loop with logging, checkpointing, and evaluation.

Reference: Appendix B.3 (Training Hyperparameters), Section 6.1 (Experimental Setup)
"""

import os
import math
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingArguments:
    """
    Training arguments following Appendix B.3.
    """
    # Output
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    
    # Training duration
    max_steps: int = 100000
    num_epochs: Optional[int] = None  # If set, overrides max_steps
    
    # Batch settings
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Optimizer (Appendix B.3)
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float = 1.0
    
    # Scheduler
    warmup_steps: int = 2000
    lr_scheduler_type: str = "cosine"  # "cosine" or "linear"
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Evaluation
    eval_steps: int = 1000
    eval_strategy: str = "steps"  # "steps" or "epoch"
    
    # Checkpointing
    save_steps: int = 5000
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 100
    logging_dir: Optional[str] = None
    report_to: str = "wandb"  # "wandb", "tensorboard", or "none"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class TENTrainer:
    """
    Trainer for TEN/HTEN models.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Cosine learning rate schedule with warmup
    - W&B logging
    - Checkpointing
    - Evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        
        # Set device
        self.device = torch.device(args.device)
        self.model.to(self.device)
        
        # Set seed
        self._set_seed(args.seed)
        
        # Create output directories
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.logging_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = args.fp16 or args.bf16
        self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
        self.scaler = GradScaler() if args.fp16 else None
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Initialize logging
        self._init_logging()
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate weight decay for different parameter types
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Don't apply weight decay to biases and layer norms
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        from ten.training.scheduler import get_scheduler
        
        total_steps = self.args.max_steps
        
        return get_scheduler(
            self.args.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps,
        )
    
    def _init_logging(self):
        """Initialize logging (W&B or TensorBoard)."""
        if self.args.report_to == "wandb" and WANDB_AVAILABLE:
            wandb.init(
                project="temporal-eigenstate-networks",
                name=self.args.run_name,
                config=vars(self.args),
                dir=self.args.logging_dir,
            )
            wandb.watch(self.model)
    
    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to configured backend."""
        if self.args.report_to == "wandb" and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training metrics
        """
        print(f"***** Running training *****")
        print(f"  Num examples = {len(self.train_dataloader.dataset)}")
        print(f"  Num steps = {self.args.max_steps}")
        print(f"  Batch size = {self.args.per_device_train_batch_size}")
        print(f"  Gradient Accumulation = {self.args.gradient_accumulation_steps}")
        print(f"  Device = {self.device}")
        
        self.model.train()
        
        train_loss = 0.0
        train_start_time = time.time()
        
        progress_bar = tqdm(total=self.args.max_steps, desc="Training")
        
        while self.global_step < self.args.max_steps:
            for batch in self.train_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch.get("labels", batch["input_ids"]),
                    )
                    loss = outputs["loss"]
                    loss = loss / self.args.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                train_loss += loss.item()
                
                # Gradient step
                if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                progress_bar.update(1)
                
                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = train_loss / self.args.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    metrics = {
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/epoch": self.epoch,
                    }
                    
                    # Add throughput
                    elapsed = time.time() - train_start_time
                    tokens_per_sec = (
                        self.args.logging_steps * 
                        self.args.per_device_train_batch_size * 
                        batch["input_ids"].shape[1]
                    ) / elapsed
                    metrics["train/tokens_per_sec"] = tokens_per_sec
                    
                    self._log_metrics(metrics, self.global_step)
                    progress_bar.set_postfix(loss=avg_loss, lr=lr)
                    
                    train_loss = 0.0
                    train_start_time = time.time()
                
                # Evaluation
                if (self.eval_dataloader is not None and 
                    self.global_step % self.args.eval_steps == 0):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, self.global_step)
                    
                    # Save best model
                    if eval_metrics["eval/loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["eval/loss"]
                        self.save_checkpoint("best")
                    
                    self.model.train()
                
                # Checkpointing
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                
                if self.global_step >= self.args.max_steps:
                    break
            
            self.epoch += 1
        
        progress_bar.close()
        
        # Final save
        self.save_checkpoint("final")
        
        return {"global_step": self.global_step, "train_loss": train_loss}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluation loop.
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch.get("labels", batch["input_ids"]),
                )
            
            loss = outputs["loss"]
            num_tokens = batch["input_ids"].numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
        
        metrics = {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }
        
        # Custom metrics
        if self.compute_metrics is not None:
            custom_metrics = self.compute_metrics(self.model, self.eval_dataloader)
            metrics.update(custom_metrics)
        
        print(f"Eval Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{name}")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save optimizer and scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save config
        torch.save(self.args, os.path.join(checkpoint_dir, "training_args.pt"))
        
        print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load model
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=self.device)
        )
        
        # Load training state
        state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"))
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_eval_loss = state["best_eval_loss"]
        
        print(f"Loaded checkpoint from {checkpoint_dir}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only save_total_limit."""
        if self.args.save_total_limit is None:
            return
        
        checkpoints = []
        for item in Path(self.args.output_dir).iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-step_"):
                step = int(item.name.split("_")[-1])
                checkpoints.append((step, item))
        
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        for step, checkpoint_dir in checkpoints[self.args.save_total_limit:]:
            import shutil
            shutil.rmtree(checkpoint_dir)
            print(f"Removed old checkpoint: {checkpoint_dir}")
