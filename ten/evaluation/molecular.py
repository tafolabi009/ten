"""
Molecular Property Prediction
=============================

TEN-based models for drug discovery tasks using MoleculeNet benchmarks.

Tasks:
- BBBP: Blood-brain barrier penetration (binary)
- Tox21: Toxicity prediction (multi-task)
- ESOL: Solubility prediction (regression)
- FreeSolv: Free solvation energy (regression)
- Lipophilicity: Octanol/water partition coefficient (regression)
- HIV: HIV replication inhibition (binary)
- BACE: Beta-secretase inhibition (binary)

Reference: MoleculeNet (Wu et al., 2018)
"""

import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ten.model.config import TENConfig
from ten.model.ten import TEN


# SMILES vocabulary
SMILES_CHARS = [
    '<pad>', '<unk>', '<cls>', '<sep>',
    'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'c', 'n', 'o', 's', 'p',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '(', ')', '[', ']', '=', '#', '@', '+', '-', '.',
    '/', '\\', '%', 'H', 'B', 'Si', 'Se', 'Te', 'As',
    'b', 'si', 'se', 'te', 'as',
]


class SMILESTokenizer:
    """
    Simple SMILES tokenizer.
    
    Converts SMILES strings to token sequences for TEN input.
    """
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        
        # Build vocabulary
        self.char_to_idx = {c: i for i, c in enumerate(SMILES_CHARS)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(SMILES_CHARS)
        
        self.pad_token_id = self.char_to_idx['<pad>']
        self.unk_token_id = self.char_to_idx['<unk>']
        self.cls_token_id = self.char_to_idx['<cls>']
        self.sep_token_id = self.char_to_idx['<sep>']
    
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into characters/tokens."""
        tokens = []
        i = 0
        while i < len(smiles):
            # Check for two-character tokens (Cl, Br, Si, etc.)
            if i + 1 < len(smiles):
                two_char = smiles[i:i+2]
                if two_char in self.char_to_idx:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character
            tokens.append(smiles[i])
            i += 1
        
        return tokens
    
    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """Encode SMILES string to token IDs."""
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = ['<cls>'] + tokens + ['<sep>']
        
        # Convert to IDs
        ids = [self.char_to_idx.get(t, self.unk_token_id) for t in tokens]
        
        # Truncate or pad
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [self.pad_token_id] * (self.max_length - len(ids))
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to SMILES."""
        tokens = [self.idx_to_char.get(i, '<unk>') for i in ids]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<pad>', '<cls>', '<sep>', '<unk>']]
        return ''.join(tokens)
    
    def batch_encode(
        self,
        smiles_list: List[str],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Batch encode multiple SMILES strings."""
        encoded = [self.encode(s) for s in smiles_list]
        
        if return_tensors == "pt":
            input_ids = torch.tensor(encoded, dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        
        return {"input_ids": encoded}


class MoleculeNetDataset(Dataset):
    """
    MoleculeNet dataset for molecular property prediction.
    
    Supported datasets:
    - bbbp: Blood-brain barrier penetration
    - tox21: Toxicity prediction
    - hiv: HIV replication inhibition
    - bace: Beta-secretase inhibition
    - esol: Aqueous solubility
    - freesolv: Free solvation energy
    - lipo: Lipophilicity
    """
    
    DATASETS = ['bbbp', 'tox21', 'hiv', 'bace', 'esol', 'freesolv', 'lipo']
    
    TASK_TYPES = {
        'bbbp': 'classification',
        'tox21': 'classification',
        'hiv': 'classification',
        'bace': 'classification',
        'esol': 'regression',
        'freesolv': 'regression',
        'lipo': 'regression',
    }
    
    NUM_TASKS = {
        'bbbp': 1,
        'tox21': 12,
        'hiv': 1,
        'bace': 1,
        'esol': 1,
        'freesolv': 1,
        'lipo': 1,
    }
    
    def __init__(
        self,
        dataset_name: str = 'bbbp',
        split: str = 'train',
        tokenizer: Optional[SMILESTokenizer] = None,
        cache_dir: Optional[str] = None,
    ):
        assert dataset_name in self.DATASETS, f"Dataset must be one of {self.DATASETS}"
        
        self.dataset_name = dataset_name
        self.split = split
        self.task_type = self.TASK_TYPES[dataset_name]
        self.num_tasks = self.NUM_TASKS[dataset_name]
        
        self.tokenizer = tokenizer or SMILESTokenizer()
        
        self._load_dataset(dataset_name, split, cache_dir)
    
    def _load_dataset(self, dataset_name: str, split: str, cache_dir: Optional[str]):
        """Load dataset from OGB or DeepChem."""
        try:
            # Try loading from OGB (Open Graph Benchmark)
            from ogb.graphproppred import PygGraphPropPredDataset
            
            ogb_name = f'ogbg-mol{dataset_name}'
            dataset = PygGraphPropPredDataset(name=ogb_name, root=cache_dir)
            
            # Get split indices
            split_idx = dataset.get_idx_split()
            idx = split_idx[split]
            
            self.data = []
            for i in idx:
                data = dataset[i]
                smiles = data.smiles if hasattr(data, 'smiles') else None
                label = data.y.squeeze().numpy()
                
                if smiles is not None:
                    self.data.append((smiles, label))
                    
        except (ImportError, ValueError):
            # Fallback: Load from DeepChem
            try:
                import deepchem as dc
                
                if dataset_name == 'bbbp':
                    loader = dc.molnet.load_bbbp
                elif dataset_name == 'tox21':
                    loader = dc.molnet.load_tox21
                elif dataset_name == 'hiv':
                    loader = dc.molnet.load_hiv
                elif dataset_name == 'bace':
                    loader = dc.molnet.load_bace_classification
                elif dataset_name == 'esol':
                    loader = dc.molnet.load_delaney
                elif dataset_name == 'freesolv':
                    loader = dc.molnet.load_sampl
                elif dataset_name == 'lipo':
                    loader = dc.molnet.load_lipo
                else:
                    raise ValueError(f"Unknown dataset: {dataset_name}")
                
                tasks, datasets, transformers = loader(featurizer='Raw')
                
                split_map = {'train': 0, 'valid': 1, 'test': 2}
                split_dataset = datasets[split_map.get(split, 0)]
                
                self.data = []
                for x, y, w, ids in split_dataset.itersamples():
                    smiles = ids
                    label = y
                    self.data.append((smiles, label))
                    
            except ImportError:
                # Final fallback: Create synthetic data for testing
                print(f"Warning: Could not load {dataset_name}, using synthetic data")
                self.data = self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> List[Tuple[str, float]]:
        """Create synthetic molecular data for testing."""
        # Simple synthetic SMILES
        synthetic_smiles = [
            'CC(C)C',  # Isobutane
            'CCO',  # Ethanol
            'CCCC',  # Butane
            'c1ccccc1',  # Benzene
            'CC(=O)O',  # Acetic acid
            'CN',  # Methylamine
            'CO',  # Methanol
            'CC',  # Ethane
            'C(C)O',  # Ethanol
            'C1CCCCC1',  # Cyclohexane
        ] * 100  # Repeat for more data
        
        import random
        data = []
        for smiles in synthetic_smiles:
            if self.task_type == 'classification':
                label = float(random.random() > 0.5)
            else:
                label = random.gauss(0, 1)
            data.append((smiles, label))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles, label = self.data[idx]
        
        # Tokenize SMILES
        encoded = self.tokenizer.encode(smiles)
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Convert label
        if self.task_type == 'classification':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label,
        }


@dataclass
class MolecularConfig(TENConfig):
    """Configuration for molecular property prediction."""
    num_tasks: int = 1
    task_type: str = 'classification'  # 'classification' or 'regression'
    pooling: str = 'cls'  # 'cls', 'mean', or 'max'


class TENForMolecularPrediction(nn.Module):
    """
    TEN model for molecular property prediction.
    
    Uses SMILES string encoding and predicts molecular properties.
    """
    
    def __init__(self, config: MolecularConfig):
        super().__init__()
        self.config = config
        
        # Core TEN model (use smaller vocab for SMILES)
        ten_config = TENConfig(
            vocab_size=len(SMILES_CHARS),
            hidden_dim=config.hidden_dim,
            num_eigenstates=config.num_eigenstates,
            num_layers=config.num_layers,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            max_seq_length=512,
        )
        self.ten = TEN(ten_config)
        
        # Prediction head
        self.pooler = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_tasks),
        )
        
        self.task_type = config.task_type
        self.pooling = config.pooling
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: SMILES token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target labels
        
        Returns:
            Dict with logits and optional loss
        """
        # Get hidden states
        hidden_states, _ = self.ten(input_ids)
        
        # Pooling
        if self.pooling == 'cls':
            pooled = hidden_states[:, 0]  # CLS token
        elif self.pooling == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(1) / mask.sum(1)
            else:
                pooled = hidden_states.mean(1)
        elif self.pooling == 'max':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states.masked_fill(mask == 0, float('-inf'))
            pooled = hidden_states.max(1)[0]
        
        pooled = torch.tanh(self.pooler(pooled))
        
        # Prediction
        logits = self.classifier(pooled)
        
        result = {'logits': logits}
        
        if labels is not None:
            if self.task_type == 'classification':
                if self.config.num_tasks == 1:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(-1), labels.float()
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                loss = F.mse_loss(logits.squeeze(-1), labels)
            
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def predict(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Predict properties for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            tokenizer: SMILES tokenizer
            device: Device to run on
        
        Returns:
            Predictions tensor
        """
        self.eval()
        
        encoded = tokenizer.batch_encode(smiles_list)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['logits']
        
        if self.task_type == 'classification':
            predictions = torch.sigmoid(logits)
        else:
            predictions = logits
        
        return predictions


class MolecularPropertyPrediction:
    """
    Training and evaluation pipeline for molecular property prediction.
    """
    
    def __init__(
        self,
        model: TENForMolecularPrediction,
        tokenizer: SMILESTokenizer,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def train(
        self,
        train_dataset: MoleculeNetDataset,
        val_dataset: Optional[MoleculeNetDataset] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Training history
        """
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset, batch_size)
                history['val_loss'].append(val_metrics['loss'])
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_metrics['loss']:.4f}")
            else:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        return history
    
    @torch.no_grad()
    def evaluate(
        self,
        dataset: MoleculeNetDataset,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            total_loss += outputs['loss'].item()
            
            logits = outputs['logits']
            
            if self.model.task_type == 'classification':
                preds = torch.sigmoid(logits)
            else:
                preds = logits
            
            all_preds.append(preds.cpu())
            all_labels.append(batch['labels'].cpu())
        
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        metrics = {
            'loss': total_loss / len(loader),
        }
        
        # Compute task-specific metrics
        if self.model.task_type == 'classification':
            from ten.evaluation.metrics import calculate_auroc, calculate_auprc
            metrics['auroc'] = calculate_auroc(preds.numpy(), labels.numpy())
            metrics['auprc'] = calculate_auprc(preds.numpy(), labels.numpy())
        else:
            from ten.evaluation.metrics import calculate_rmse, calculate_mae
            metrics['rmse'] = calculate_rmse(preds.numpy(), labels.numpy())
            metrics['mae'] = calculate_mae(preds.numpy(), labels.numpy())
        
        return metrics
