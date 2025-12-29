"""
Language Modeling Heads for TEN/HTEN
====================================

Wrapper models for causal language modeling tasks.
Reference: Section 6.2 (Language Modeling Results)
"""

from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ten.model.config import TENConfig, HTENConfig
from ten.model.ten import TEN
from ten.model.hten import HTEN


class TENForLanguageModeling(nn.Module):
    """
    TEN with language modeling head.
    
    Used for WikiText-103 and OpenWebText experiments (Section 6.2).
    """
    
    def __init__(self, config: TENConfig):
        super().__init__()
        self.config = config
        
        # Core TEN model
        self.ten = TEN(config)
        
        # Language modeling head (tied with token embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights with token embeddings
        self.lm_head.weight = self.ten.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        states: Optional[list] = None,
        use_parallel: bool = True,
        return_hidden: bool = False
    ) -> dict:
        """
        Language modeling forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            labels: Target token IDs for loss computation (batch, seq_len)
            states: Optional previous states for continuation
            use_parallel: Use parallel scan for training
            return_hidden: Return hidden states
        
        Returns:
            Dict with logits, loss (optional), hidden states (optional), new_states
        """
        # Get hidden states from TEN
        hidden_states, new_states = self.ten(input_ids, states, use_parallel)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        result = {
            "logits": logits,
            "states": new_states,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for causal LM (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        
        if return_hidden:
            result["hidden_states"] = hidden_states
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            input_ids: Prompt tokens (batch, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End of sequence token
        
        Returns:
            Generated token IDs (batch, prompt_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Process prompt
        output = self.forward(input_ids, use_parallel=False)
        states = output["states"]
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get last token logits
            logits = output["logits"][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # Process single token with state
            output = self.forward(next_token, states=states, use_parallel=False)
            states = output["states"]
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters (excluding LM head which is tied)."""
        return self.ten.get_num_params(non_embedding)


class HTENForLanguageModeling(nn.Module):
    """
    HTEN with language modeling head.
    
    Multi-scale processing for improved long-range modeling (Section 5, 6.2).
    """
    
    def __init__(self, config: HTENConfig):
        super().__init__()
        self.config = config
        
        # Core HTEN model
        self.hten = HTEN(config)
        
        # Language modeling head (tied with token embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights with token embeddings
        self.lm_head.weight = self.hten.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        states: Optional[List[dict]] = None,
        use_parallel: bool = True,
        return_hidden: bool = False
    ) -> dict:
        """
        Language modeling forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            labels: Target token IDs for loss computation
            states: Optional previous states
            use_parallel: Use parallel scan
            return_hidden: Return hidden states
        
        Returns:
            Dict with logits, loss (optional), hidden states (optional), new_states
        """
        # Get hidden states from HTEN
        hidden_states, new_states = self.hten(input_ids, states, use_parallel)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        result = {
            "logits": logits,
            "states": new_states,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        
        if return_hidden:
            result["hidden_states"] = hidden_states
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation (same as TEN)."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        output = self.forward(input_ids, use_parallel=False)
        states = output["states"]
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            logits = output["logits"][:, -1, :] / temperature
            
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            output = self.forward(next_token, states=states, use_parallel=False)
            states = output["states"]
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        return self.hten.get_num_params(non_embedding)
