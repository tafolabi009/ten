"""
Data Loading Utilities
======================

Dataset classes for language modeling, long-range arena, and drug discovery tasks.

Reference: Section 6.1 (Datasets)
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class WikiTextDataset(Dataset):
    """
    WikiText-103 dataset for language modeling.
    
    Reference: Section 6.1, Table 1
    """
    
    def __init__(
        self,
        split: str = "train",
        seq_length: int = 2048,
        tokenizer: Optional[Any] = None,
        cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.seq_length = seq_length
        
        # Load tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Load dataset
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
        
        # Tokenize and concatenate all text
        all_tokens = []
        for example in dataset:
            text = example["text"]
            if text.strip():
                tokens = self.tokenizer.encode(text)
                all_tokens.extend(tokens)
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.num_sequences = (len(self.tokens) - 1) // self.seq_length
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        tokens = self.tokens[start_idx:end_idx]
        
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class OpenWebTextDataset(Dataset):
    """
    OpenWebText subset dataset for language modeling.
    
    Reference: Section 6.1
    """
    
    def __init__(
        self,
        split: str = "train",
        seq_length: int = 2048,
        max_examples: int = 100000,
        tokenizer: Optional[Any] = None,
        cache_dir: Optional[str] = None,
    ):
        self.seq_length = seq_length
        
        # Load tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Load dataset (streaming for large dataset)
        from datasets import load_dataset
        
        if split == "train":
            dataset = load_dataset("openwebtext", split="train", streaming=True, cache_dir=cache_dir)
        else:
            # OpenWebText doesn't have official splits, use a portion
            dataset = load_dataset("openwebtext", split="train", streaming=True, cache_dir=cache_dir)
        
        # Tokenize examples
        all_tokens = []
        count = 0
        for example in dataset:
            if count >= max_examples:
                break
            text = example["text"]
            if text.strip():
                tokens = self.tokenizer.encode(text)
                all_tokens.extend(tokens)
                count += 1
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.num_sequences = (len(self.tokens) - 1) // self.seq_length
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        tokens = self.tokens[start_idx:end_idx]
        
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class LongRangeArenaDataset(Dataset):
    """
    Long Range Arena (LRA) benchmark dataset.
    
    Tasks:
    - ListOps: Parsing expressions
    - Text: IMDb sentiment
    - Retrieval: ACL anthology similarity
    - Image: CIFAR-10 as sequences
    - Pathfinder: Path detection in images
    
    Reference: Section 6.3, Table 2
    """
    
    TASKS = ["listops", "text", "retrieval", "image", "pathfinder"]
    
    def __init__(
        self,
        task: str = "listops",
        split: str = "train",
        max_length: int = 2048,
        cache_dir: Optional[str] = None,
    ):
        assert task in self.TASKS, f"Task must be one of {self.TASKS}"
        
        self.task = task
        self.split = split
        self.max_length = max_length
        
        self._load_task(task, split, cache_dir)
    
    def _load_task(self, task: str, split: str, cache_dir: Optional[str]):
        """Load task-specific data."""
        from datasets import load_dataset
        
        if task == "listops":
            # ListOps: Parsing expressions
            # Format: expressions like "(max 1 2 (min 3 4))" -> output value
            dataset = load_dataset(
                "lhoestq/lra", 
                "listops-1000",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            self.data = [(ex["Source"], ex["Target"]) for ex in dataset]
            self.num_classes = 10
            
        elif task == "text":
            # Text: IMDb sentiment (binary classification)
            dataset = load_dataset("imdb", split=split, cache_dir=cache_dir)
            
            # Character-level tokenization for LRA
            self.data = []
            for ex in dataset:
                chars = [ord(c) for c in ex["text"][:self.max_length]]
                if len(chars) < self.max_length:
                    chars += [0] * (self.max_length - len(chars))
                self.data.append((chars, ex["label"]))
            
            self.num_classes = 2
            
        elif task == "retrieval":
            # Document retrieval (matching)
            # Using simplified version
            dataset = load_dataset(
                "lhoestq/lra",
                "aan",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            self.data = [(ex["input1_id"] + ex["input2_id"], ex["label"]) for ex in dataset]
            self.num_classes = 2
            
        elif task == "image":
            # CIFAR-10 as 1D sequence (32*32*3 = 3072)
            dataset = load_dataset("cifar10", split=split, cache_dir=cache_dir)
            
            self.data = []
            for ex in dataset:
                img = ex["img"]
                # Flatten image to sequence
                import numpy as np
                img_array = np.array(img).flatten().tolist()
                self.data.append((img_array, ex["label"]))
            
            self.num_classes = 10
            
        elif task == "pathfinder":
            # Pathfinder: Synthetic task
            dataset = load_dataset(
                "lhoestq/lra",
                "pathfinder32-curv_contour_length_14",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            self.data = [(ex["input_ids"], ex["label"]) for ex in dataset]
            self.num_classes = 2
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence, label = self.data[idx]
        
        # Convert to tensor
        if isinstance(sequence, list):
            sequence = torch.tensor(sequence, dtype=torch.long)
        else:
            sequence = torch.tensor([ord(c) for c in str(sequence)], dtype=torch.long)
        
        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            padding = torch.zeros(self.max_length - len(sequence), dtype=torch.long)
            sequence = torch.cat([sequence, padding])
        else:
            sequence = sequence[:self.max_length]
        
        return {
            "input_ids": sequence,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_wikitext_dataloader(
    split: str = "train",
    seq_length: int = 2048,
    batch_size: int = 32,
    num_workers: int = 4,
    tokenizer: Optional[Any] = None,
) -> DataLoader:
    """
    Create WikiText-103 dataloader.
    
    Reference: Table 1 (Language modeling perplexity on WikiText-103)
    """
    dataset = WikiTextDataset(
        split=split,
        seq_length=seq_length,
        tokenizer=tokenizer,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_lra_dataloader(
    task: str = "listops",
    split: str = "train",
    max_length: int = 2048,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create Long Range Arena dataloader.
    
    Reference: Table 2 (Long Range Arena accuracy)
    """
    dataset = LongRangeArenaDataset(
        task=task,
        split=split,
        max_length=max_length,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


class CollatorForLanguageModeling:
    """Collator for language modeling that handles variable length sequences."""
    
    def __init__(self, pad_token_id: int = 0, max_length: int = 2048):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_len = min(
            max(f["input_ids"].shape[0] for f in features),
            self.max_length
        )
        
        input_ids = []
        labels = []
        
        for f in features:
            ids = f["input_ids"][:max_len]
            lbls = f["labels"][:max_len]
            
            # Pad if needed
            if len(ids) < max_len:
                padding = torch.full((max_len - len(ids),), self.pad_token_id, dtype=torch.long)
                ids = torch.cat([ids, padding])
                lbls = torch.cat([lbls, torch.full((max_len - len(lbls),), -100, dtype=torch.long)])
            
            input_ids.append(ids)
            labels.append(lbls)
        
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
        }


# Alias for convenience
TextDataset = WikiTextDataset
