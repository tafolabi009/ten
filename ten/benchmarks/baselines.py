"""
Baseline Model Implementations
==============================

Standard Transformer and S4 baselines for comparison.

Reference: Section 6.1 (Models), Table 1, Table 2
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class TransformerConfig:
    """
    Configuration for standard Transformer baseline.
    
    Reference: Section 6.1 - "8 attention heads, 6 layers"
    """
    vocab_size: int = 50257
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_hidden_dim: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 8192
    layer_norm_eps: float = 1e-6
    use_flash_attention: bool = True


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention (O(T²) complexity).
    
    Used as baseline for comparison.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b t (three h d) -> three b h t d', 
                           three=3, h=self.num_heads)
        
        # Use Flash Attention if available
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
        output = self.out_proj(attn_output)
        
        return output


class TransformerLayer(nn.Module):
    """Standard Transformer layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        self.attention = MultiHeadAttention(config)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = residual + self.attention(x, is_causal=is_causal)
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class TransformerBaseline(nn.Module):
    """
    Standard Transformer baseline model.
    
    Reference: Section 6.1, Table 1
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device)
        x = x + self.pos_embedding(positions)
        x = self.embed_dropout(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        result = {"logits": logits}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        
        return result
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class S4Config:
    """
    Configuration for S4 (Structured State Space) baseline.
    
    Reference: Section 2.3, Table 2
    """
    vocab_size: int = 50257
    hidden_dim: int = 512
    state_dim: int = 64  # N in S4
    num_layers: int = 6
    ffn_hidden_dim: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 8192
    layer_norm_eps: float = 1e-6
    
    # S4-specific
    dt_min: float = 0.001
    dt_max: float = 0.1
    use_hippo: bool = True  # HiPPO initialization


class S4Layer(nn.Module):
    """
    Simplified S4 layer implementation.
    
    Implements structured state space model:
        x'(t) = Ax(t) + Bu(t)
        y(t) = Cx(t) + Du(t)
    
    Reference: "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al.)
    """
    
    def __init__(self, config: S4Config):
        super().__init__()
        self.config = config
        self.H = config.hidden_dim
        self.N = config.state_dim
        
        # State space parameters
        if config.use_hippo:
            # HiPPO-LegS initialization
            A, B = self._make_hippo_matrices(self.N)
        else:
            # Random initialization
            A = torch.randn(self.N, self.N) / math.sqrt(self.N)
            B = torch.randn(self.N, 1)
        
        self.register_buffer('A', A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(torch.randn(self.H, self.N) / math.sqrt(self.N))
        self.D = nn.Parameter(torch.randn(self.H))
        
        # Discretization timestep
        log_dt = torch.rand(self.H) * (
            math.log(config.dt_max) - math.log(config.dt_min)
        ) + math.log(config.dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Output projection
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def _make_hippo_matrices(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create HiPPO-LegS matrices for state space initialization.
        """
        # HiPPO-LegS: Legendre polynomial basis
        P = torch.sqrt(2 * torch.arange(N).float() + 1)
        
        A = torch.zeros(N, N)
        for n in range(N):
            for k in range(n + 1):
                if k < n:
                    A[n, k] = P[n] * P[k]
                else:
                    A[n, k] = P[n] * P[k] / 2
        
        A = -A
        B = P.unsqueeze(1)
        
        return A, B
    
    def _discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous state space to discrete.
        
        Uses bilinear (Tustin) discretization:
            A_d = (I + dt/2 * A) @ inv(I - dt/2 * A)
            B_d = dt * inv(I - dt/2 * A) @ B
        """
        dt = torch.exp(self.log_dt)  # (H,)
        
        # For simplicity, use diagonal approximation
        # Full discretization would require matrix inverse
        A_d = torch.exp(dt.unsqueeze(1) * self.A.unsqueeze(0))  # (H, N, N) simplified
        B_d = dt.unsqueeze(1) * self.B  # (H, N, 1) simplified
        
        return A_d, B_d
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        S4 forward pass.
        
        Args:
            u: Input (batch, seq_len, hidden_dim)
        
        Returns:
            Output (batch, seq_len, hidden_dim)
        """
        batch, seq_len, H = u.shape
        
        # Discretize
        dt = torch.exp(self.log_dt)
        
        # Compute kernel using FFT convolution
        # K = C @ exp(A * dt * [0, 1, 2, ...]) @ B
        
        # Simplified version: diagonal state space
        t = torch.arange(seq_len, device=u.device, dtype=u.dtype)
        
        # Assume A is approximately diagonal for efficiency
        # This is a common simplification in S4 variants
        A_diag = torch.diag(self.A)  # (N,)
        
        # Kernel: K[t] = C @ exp(A * dt * t) @ B
        # Shape manipulation for efficient computation
        exp_At = torch.exp(A_diag.unsqueeze(0) * dt.unsqueeze(1).unsqueeze(0) * t.unsqueeze(1).unsqueeze(2))
        # exp_At: (seq_len, H, N)
        
        K = torch.einsum('hn,thn,nm->thm', self.C, exp_At, self.B.squeeze(-1).unsqueeze(-1))
        K = K.squeeze(-1).T  # (H, seq_len)
        
        # Causal convolution via FFT
        # Pad for FFT convolution
        fft_size = 2 * seq_len
        
        u_fft = torch.fft.rfft(u.transpose(1, 2), n=fft_size, dim=-1)  # (batch, H, fft_size//2+1)
        K_fft = torch.fft.rfft(K, n=fft_size, dim=-1)  # (H, fft_size//2+1)
        
        y_fft = u_fft * K_fft.unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=fft_size, dim=-1)[..., :seq_len]  # (batch, H, seq_len)
        
        y = y.transpose(1, 2)  # (batch, seq_len, H)
        
        # Add skip connection
        y = y + self.D * u
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y


class S4Block(nn.Module):
    """S4 block with layer norm and FFN."""
    
    def __init__(self, config: S4Config):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        self.s4 = S4Layer(config)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual + self.s4(x)
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class S4Baseline(nn.Module):
    """
    S4 (Structured State Space) baseline model.
    
    Reference: Section 2.3, Table 2
    """
    
    def __init__(self, config: S4Config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            S4Block(config) for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device)
        x = x + self.pos_embedding(positions)
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        result = {"logits": logits}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        
        return result
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
