"""
Hierarchical Temporal Eigenstate Networks (HTEN)
================================================

Implements multi-scale processing from Section 5.

Key Equation (Eq. 12):
    h_t = Σ_{s∈S} W_s · Upsample(TEN_s(Downsample_s(x)))

Advantages (Section 5.2):
1. Efficiency: Lower scales process fewer tokens
2. Long-range: Coarse scales capture global structure
3. Fine-grained: Fine scales preserve local detail  
4. Parallelism: All scales computed in parallel

Total complexity: O(TKd log|S|), still linear in T
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from ten.model.config import HTENConfig
from ten.model.ten import TENLayer, EigenstateEvolution, ResonanceCoupling


class ScaleDownsample(nn.Module):
    """
    Downsampling for multi-scale processing (Section 5.1).
    
    Downsampling uses average pooling by factor s.
    """
    
    def __init__(self, scale: int, method: str = "avg_pool"):
        super().__init__()
        self.scale = scale
        self.method = method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input by scale factor.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
        
        Returns:
            Downsampled tensor (batch, seq_len // scale, hidden_dim)
        """
        if self.scale == 1:
            return x
        
        batch, seq_len, d = x.shape
        
        # Pad to make divisible by scale
        pad_len = (self.scale - seq_len % self.scale) % self.scale
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        if self.method == "avg_pool":
            # Reshape and average pool
            x = rearrange(x, 'b (n s) d -> b n s d', s=self.scale)
            x = x.mean(dim=2)
        elif self.method == "strided":
            # Take every s-th element
            x = x[:, ::self.scale, :]
        else:
            raise ValueError(f"Unknown downsample method: {self.method}")
        
        return x


class ScaleUpsample(nn.Module):
    """
    Upsampling for multi-scale processing (Section 5.1).
    
    Upsampling uses linear interpolation to original resolution.
    """
    
    def __init__(self, scale: int, method: str = "linear_interp"):
        super().__init__()
        self.scale = scale
        self.method = method
    
    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Upsample to target sequence length.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            target_len: Target sequence length
        
        Returns:
            Upsampled tensor (batch, target_len, hidden_dim)
        """
        if self.scale == 1:
            return x[:, :target_len, :]
        
        batch, seq_len, d = x.shape
        
        if self.method == "linear_interp":
            # Transpose for interpolate (expects channel dim at position 1)
            x = rearrange(x, 'b t d -> b d t')
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=True)
            x = rearrange(x, 'b d t -> b t d')
        elif self.method == "nearest":
            x = rearrange(x, 'b t d -> b d t')
            x = F.interpolate(x, size=target_len, mode='nearest')
            x = rearrange(x, 'b d t -> b t d')
        elif self.method == "repeat":
            x = repeat(x, 'b t d -> b (t s) d', s=self.scale)
            x = x[:, :target_len, :]
        else:
            raise ValueError(f"Unknown upsample method: {self.method}")
        
        return x


class ScaleTEN(nn.Module):
    """
    TEN module for a single scale in HTEN.
    
    Each scale has K/|S| eigenstates and processes T/s tokens.
    """
    
    def __init__(self, config: HTENConfig, scale: int):
        super().__init__()
        self.config = config
        self.scale = scale
        
        # Downsample/upsample
        self.downsample = ScaleDownsample(scale, config.downsample_method)
        self.upsample = ScaleUpsample(scale, config.upsample_method)
        
        # Create a modified config for this scale
        scale_config = HTENConfig(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_eigenstates=config.eigenstates_per_scale,
            num_layers=1,  # Single layer per scale
            alpha_init_range=config.alpha_init_range,
            omega_init_scale=config.omega_init_scale,
            resonance_epsilon=config.resonance_epsilon,
            use_resonance=config.use_resonance,
            ffn_hidden_dim=config.ffn_hidden_dim,
            ffn_dropout=config.ffn_dropout,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            eigenvalue_constraint=config.eigenvalue_constraint,
            max_seq_length=config.max_seq_length // scale + 1,
            use_positional_encoding=False,  # No pos encoding at scale level
            init_std=config.init_std,
        )
        
        # TEN layer for this scale
        self.ten_layer = TENLayer(scale_config)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process input at this scale.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            state: Optional previous state
            use_parallel: Use parallel scan
        
        Returns:
            Tuple of (output at original resolution, new_state)
        """
        original_len = x.shape[1]
        
        # Downsample
        x_down = self.downsample(x)
        
        # Process with TEN
        h, new_state = self.ten_layer(x_down, state, use_parallel)
        
        # Upsample back to original resolution
        h_up = self.upsample(h, original_len)
        
        return h_up, new_state


class HTENLayer(nn.Module):
    """
    Hierarchical TEN layer (Section 5).
    
    Processes input at multiple temporal scales s ∈ {1, 2, 4, 8} simultaneously.
    """
    
    def __init__(self, config: HTENConfig):
        super().__init__()
        self.config = config
        
        # Pre-layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Scale-specific TEN modules
        self.scale_tens = nn.ModuleDict({
            str(s): ScaleTEN(config, s) for s in config.scales
        })
        
        # Scale fusion weights (Eq. 12)
        if config.scale_fusion == "learned_weights":
            self.scale_weights = nn.Parameter(torch.ones(len(config.scales)) / len(config.scales))
        elif config.scale_fusion == "concat":
            self.scale_proj = nn.Linear(config.hidden_dim * len(config.scales), config.hidden_dim)
        else:
            raise ValueError(f"Unknown scale_fusion: {config.scale_fusion}")
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.ffn_dropout),
            nn.Linear(config.ffn_hidden_dim, config.hidden_dim),
            nn.Dropout(config.ffn_dropout),
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[dict] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        HTEN layer forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            states: Optional dict of per-scale states
            use_parallel: Use parallel scan
        
        Returns:
            Tuple of (output, new_states)
        """
        residual = x
        x = self.norm1(x)
        
        # Process all scales in parallel (can be done with torch.jit.fork)
        scale_outputs = []
        new_states = {}
        
        for scale_str, scale_ten in self.scale_tens.items():
            scale = int(scale_str)
            state = states.get(scale) if states else None
            h_scale, new_state = scale_ten(x, state, use_parallel)
            scale_outputs.append(h_scale)
            new_states[scale] = new_state
        
        # Fuse scales (Eq. 12)
        if self.config.scale_fusion == "learned_weights":
            # h = Σ_s W_s · h_s
            weights = F.softmax(self.scale_weights, dim=0)
            h = sum(w * h_s for w, h_s in zip(weights, scale_outputs))
        elif self.config.scale_fusion == "concat":
            # Concatenate and project
            h = torch.cat(scale_outputs, dim=-1)
            h = self.scale_proj(h)
        
        # Output projection and residual
        h = self.output_proj(h)
        h = self.dropout(h)
        x = residual + h
        
        # Feedforward with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x, new_states


class HTEN(nn.Module):
    """
    Hierarchical Temporal Eigenstate Network (HTEN).
    
    Full model with multi-scale processing at each layer.
    
    Reference: Section 5, Section 6.1
    """
    
    def __init__(self, config: HTENConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        else:
            self.pos_embedding = None
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # HTEN layers
        self.layers = nn.ModuleList([
            HTENLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights (Appendix B.2)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[dict]] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        HTEN forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            states: Optional list of per-layer state dicts
            use_parallel: Use parallel scan
        
        Returns:
            Tuple of (hidden_states, new_states)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Positional encoding
        if self.pos_embedding is not None:
            positions = torch.arange(seq_len, device=device)
            x = x + self.pos_embedding(positions)
        
        x = self.embed_dropout(x)
        
        # Apply HTEN layers
        new_states = []
        for i, layer in enumerate(self.layers):
            layer_states = states[i] if states is not None else None
            x, new_layer_states = layer(x, layer_states, use_parallel)
            new_states.append(new_layer_states)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x, new_states
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_embedding is not None:
            n_params -= self.pos_embedding.weight.numel()
        return n_params
