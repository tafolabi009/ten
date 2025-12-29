"""
Core TEN Layer Implementation
=============================

Implements the Temporal Eigenstate Networks architecture exactly as specified in the paper.

Key Equations:
- Eq. 1: h_t = Re[Σ_k c_k(t) * v_k]  (Eigenstate decomposition)
- Eq. 2: c_k(t+1) = λ_k * c_k(t) + β_k(t)  (Temporal evolution)
- Eq. 3: c̃_k(t) = Σ_j R_kj * c_j(t)  (Resonance coupling)

Algorithm 1: TEN Forward Pass (Section 3.5)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ten.model.config import TENConfig


class ComplexTensor:
    """
    Helper class for complex tensor operations.
    
    Implements complex arithmetic as pairs of real tensors (Appendix B.1, Eq. 36-37):
    c_k = r_k + i * i_k
    
    This avoids PyTorch's native complex tensor limitations for gradients.
    """
    
    def __init__(self, real: torch.Tensor, imag: torch.Tensor):
        self.real = real
        self.imag = imag
    
    @staticmethod
    def from_polar(magnitude: torch.Tensor, phase: torch.Tensor) -> "ComplexTensor":
        """Create complex tensor from polar form: magnitude * exp(i * phase)"""
        return ComplexTensor(
            real=magnitude * torch.cos(phase),
            imag=magnitude * torch.sin(phase)
        )
    
    def __mul__(self, other: "ComplexTensor") -> "ComplexTensor":
        """Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        return ComplexTensor(
            real=self.real * other.real - self.imag * other.imag,
            imag=self.real * other.imag + self.imag * other.real
        )
    
    def __add__(self, other: "ComplexTensor") -> "ComplexTensor":
        """Complex addition"""
        return ComplexTensor(
            real=self.real + other.real,
            imag=self.imag + other.imag
        )
    
    def conj(self) -> "ComplexTensor":
        """Complex conjugate"""
        return ComplexTensor(self.real, -self.imag)
    
    def abs_squared(self) -> torch.Tensor:
        """Returns |c|² = real² + imag²"""
        return self.real ** 2 + self.imag ** 2
    
    def magnitude(self) -> torch.Tensor:
        """Returns |c|"""
        return torch.sqrt(self.abs_squared() + 1e-8)


class EigenstateEvolution(nn.Module):
    """
    Core eigenstate evolution mechanism (Section 3.3).
    
    Implements Equation 2:
        c_k(t+1) = λ_k · c_k(t) + β_k(t)
    
    where:
        λ_k = exp(α_k + i*ω_k) is the learned eigenvalue
        α_k controls decay/growth rate  
        ω_k controls oscillation frequency
        β_k(t) = ⟨x_t, v*_k⟩ is input projection
    
    Stability guarantee (Theorem 4):
        |λ_k| ≤ 1 ensures bounded energy growth via Lyapunov analysis.
    """
    
    def __init__(self, config: TENConfig):
        super().__init__()
        self.config = config
        self.K = config.num_eigenstates
        self.d = config.hidden_dim
        
        # Learned eigenvalues λ_k = exp(α_k + i*ω_k)
        # α_k: decay rates (Appendix B.2)
        # Initialize α_k ~ U(-3, 0) for exponential decay
        self.alpha = nn.Parameter(
            torch.empty(self.K).uniform_(*config.alpha_init_range)
        )
        
        # ω_k: oscillation frequencies (Appendix B.2)
        # Initialize ω_k = 2πk/K to spread across frequency spectrum
        omega_init = 2 * math.pi * torch.arange(self.K).float() / self.K * config.omega_init_scale
        self.omega = nn.Parameter(omega_init)
        
        # Learned eigenvectors v_k ∈ C^d (Section 3.2)
        # Initialize ~ N(0, 0.02/sqrt(d)) then orthonormalize
        self.eigenvectors_real = nn.Parameter(
            torch.randn(self.K, self.d) * config.init_std / math.sqrt(self.d)
        )
        self.eigenvectors_imag = nn.Parameter(
            torch.randn(self.K, self.d) * config.init_std / math.sqrt(self.d)
        )
    
    def get_eigenvalues(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigenvalues λ_k = exp(α_k + i*ω_k) (Eq. 35 in Appendix B.1).
        
        For stability (Theorem 4), we constrain |λ_k| ≤ 1:
            λ_k = σ(α_k) * exp(i*ω_k)
        where σ is sigmoid.
        """
        if self.config.eigenvalue_constraint == "sigmoid":
            # Constrain magnitude to [0, 1] for stability
            magnitude = torch.sigmoid(self.alpha)
        elif self.config.eigenvalue_constraint == "exp_clamp":
            # Clamp α to ensure exp(α) ≤ 1
            magnitude = torch.exp(torch.clamp(self.alpha, max=0.0))
        else:
            magnitude = torch.exp(self.alpha)
        
        # Return as real and imaginary components
        lambda_real = magnitude * torch.cos(self.omega)  # |λ| * cos(ω)
        lambda_imag = magnitude * torch.sin(self.omega)  # |λ| * sin(ω)
        
        return lambda_real, lambda_imag
    
    def project_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project input to eigenstate space: β_k(t) = ⟨x_t, v*_k⟩ (Algorithm 1, Line 6)
        
        Using Appendix B.4 efficient implementation:
            beta = einsum('kd,btd->btk', eigenvectors.conj(), input)
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        
        Returns:
            Tuple of (beta_real, beta_imag) each of shape (batch, seq_len, K)
        """
        # β_k = ⟨x, v*_k⟩ = x · conj(v_k) = x · (v_real - i*v_imag)
        # β_k = (x · v_real) + i*(-x · v_imag)
        # Since x is real: β_real = x · v_real, β_imag = -x · v_imag
        
        beta_real = torch.einsum('btd,kd->btk', x, self.eigenvectors_real)
        beta_imag = torch.einsum('btd,kd->btk', x, -self.eigenvectors_imag)  # Conjugate
        
        return beta_real, beta_imag
    
    def evolve_states(
        self,
        beta_real: torch.Tensor,
        beta_imag: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evolve eigenstate amplitudes through time (Algorithm 1, Line 7).
        
        Implements Equation 2:
            c_k(t+1) = λ_k · c_k(t) + β_k(t)
        
        Args:
            beta_real: Real part of input projections (batch, seq_len, K)
            beta_imag: Imaginary part of input projections (batch, seq_len, K)
            initial_state: Optional tuple of (c_real, c_imag) for continuation
        
        Returns:
            Tuple of (c_real, c_imag, final_state) where c has shape (batch, seq_len, K)
        """
        batch, seq_len, K = beta_real.shape
        device = beta_real.device
        dtype = beta_real.dtype
        
        # Get eigenvalues
        lambda_real, lambda_imag = self.get_eigenvalues()
        
        # Initialize states c_k(0) = 0 (Algorithm 1, Line 3)
        if initial_state is None:
            c_real = torch.zeros(batch, K, device=device, dtype=dtype)
            c_imag = torch.zeros(batch, K, device=device, dtype=dtype)
        else:
            c_real, c_imag = initial_state
        
        # Collect all timestep outputs
        c_real_all = []
        c_imag_all = []
        
        # Sequential evolution (can be parallelized, see parallel_scan version)
        for t in range(seq_len):
            # c_k(t+1) = λ_k · c_k(t) + β_k(t)
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            new_c_real = lambda_real * c_real - lambda_imag * c_imag + beta_real[:, t, :]
            new_c_imag = lambda_real * c_imag + lambda_imag * c_real + beta_imag[:, t, :]
            
            c_real = new_c_real
            c_imag = new_c_imag
            
            c_real_all.append(c_real)
            c_imag_all.append(c_imag)
        
        # Stack to (batch, seq_len, K)
        c_real_seq = torch.stack(c_real_all, dim=1)
        c_imag_seq = torch.stack(c_imag_all, dim=1)
        
        # Return final state for continuation
        final_state = (c_real, c_imag)
        
        return c_real_seq, c_imag_seq, final_state
    
    def evolve_states_parallel(
        self,
        beta_real: torch.Tensor,
        beta_imag: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parallel scan version of eigenstate evolution for training efficiency.
        
        Uses associative scan to compute all timesteps in O(log T) parallel steps.
        Reference: Section 7.1 - "eigenstates evolve independently, enabling parallel computation"
        """
        batch, seq_len, K = beta_real.shape
        device = beta_real.device
        dtype = beta_real.dtype
        
        # Get eigenvalues
        lambda_real, lambda_imag = self.get_eigenvalues()  # (K,)
        
        # Expand eigenvalues for broadcasting
        lambda_real = lambda_real.view(1, 1, K)  # (1, 1, K)
        lambda_imag = lambda_imag.view(1, 1, K)
        
        # Initialize
        if initial_state is None:
            c0_real = torch.zeros(batch, 1, K, device=device, dtype=dtype)
            c0_imag = torch.zeros(batch, 1, K, device=device, dtype=dtype)
        else:
            c0_real = initial_state[0].unsqueeze(1)
            c0_imag = initial_state[1].unsqueeze(1)
        
        # Parallel prefix sum using matrix exponential property
        # c_k(t) = λ_k^t * c_k(0) + Σ_{τ=0}^{t-1} λ_k^{t-1-τ} * β_k(τ)
        # This is a linear recurrence that can be solved with parallel scan
        
        # Compute powers of lambda for each position
        positions = torch.arange(seq_len, device=device, dtype=dtype).view(1, -1, 1)
        
        # λ^t = |λ|^t * exp(i*t*ω) 
        magnitude = torch.sqrt(lambda_real**2 + lambda_imag**2 + 1e-8)
        phase = torch.atan2(lambda_imag, lambda_real)
        
        # Compute λ^t for each timestep
        mag_powers = magnitude ** positions
        phase_mult = phase * positions
        
        lambda_t_real = mag_powers * torch.cos(phase_mult)
        lambda_t_imag = mag_powers * torch.sin(phase_mult)
        
        # Apply initial condition: λ^t * c_0
        init_contrib_real = lambda_t_real * c0_real - lambda_t_imag * c0_imag
        init_contrib_imag = lambda_t_real * c0_imag + lambda_t_imag * c0_real
        
        # Convolve with inputs using causal convolution
        # c_k(t) = Σ_{τ=0}^{t-1} λ_k^{t-1-τ} * β_k(τ) + λ^t * c_0
        # This is equivalent to a causal convolution with kernel λ^{T-1}, λ^{T-2}, ..., λ^0
        
        # Use FFT-based convolution for efficiency
        c_real_seq, c_imag_seq = self._causal_conv_complex(
            beta_real, beta_imag, lambda_real.squeeze(0).squeeze(0), lambda_imag.squeeze(0).squeeze(0), seq_len
        )
        
        # Add initial condition contribution
        c_real_seq = c_real_seq + init_contrib_real
        c_imag_seq = c_imag_seq + init_contrib_imag
        
        # Final state
        final_state = (c_real_seq[:, -1, :], c_imag_seq[:, -1, :])
        
        return c_real_seq, c_imag_seq, final_state
    
    def _causal_conv_complex(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        lambda_real: torch.Tensor,
        lambda_imag: torch.Tensor,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Causal convolution for eigenstate evolution using FFT.
        """
        batch, T, K = x_real.shape
        device = x_real.device
        
        # Build kernel: [1, λ, λ², ..., λ^{T-1}]
        t_idx = torch.arange(T, device=device).float()
        mag = torch.sqrt(lambda_real**2 + lambda_imag**2 + 1e-8)
        phase = torch.atan2(lambda_imag, lambda_real)
        
        mag_powers = mag.unsqueeze(0) ** t_idx.unsqueeze(1)  # (T, K)
        phase_mult = phase.unsqueeze(0) * t_idx.unsqueeze(1)
        
        kernel_real = mag_powers * torch.cos(phase_mult)  # (T, K)
        kernel_imag = mag_powers * torch.sin(phase_mult)
        
        # Pad for FFT convolution
        n_fft = 2 * T
        
        # FFT of inputs
        x_real_pad = F.pad(x_real, (0, 0, 0, T))  # (batch, 2T, K)
        x_imag_pad = F.pad(x_imag, (0, 0, 0, T))
        
        kernel_real_pad = F.pad(kernel_real, (0, 0, 0, T))  # (2T, K)
        kernel_imag_pad = F.pad(kernel_imag, (0, 0, 0, T))
        
        # Convert to complex
        x_complex = torch.complex(x_real_pad, x_imag_pad)  # (batch, 2T, K)
        kernel_complex = torch.complex(kernel_real_pad, kernel_imag_pad)  # (2T, K)
        
        # FFT
        x_fft = torch.fft.fft(x_complex, dim=1)
        kernel_fft = torch.fft.fft(kernel_complex, dim=0).unsqueeze(0)
        
        # Multiply in frequency domain
        result_fft = x_fft * kernel_fft
        
        # IFFT
        result = torch.fft.ifft(result_fft, dim=1)
        
        # Take causal part
        result = result[:, :T, :]
        
        return result.real, result.imag
    
    def reconstruct(
        self,
        c_real: torch.Tensor,
        c_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct hidden states from eigenstate amplitudes (Algorithm 1, Line 10).
        
        Implements Equation 1:
            h_t = Re[Σ_k c_k(t) * v_k]
        
        Args:
            c_real: Real part of amplitudes (batch, seq_len, K)
            c_imag: Imaginary part of amplitudes (batch, seq_len, K)
        
        Returns:
            Hidden states h of shape (batch, seq_len, hidden_dim)
        """
        # h = Re[Σ_k c_k * v_k]
        # c_k * v_k = (c_r + i*c_i)(v_r + i*v_i) = (c_r*v_r - c_i*v_i) + i*(c_r*v_i + c_i*v_r)
        # Re[...] = c_r*v_r - c_i*v_i
        
        h = torch.einsum('btk,kd->btd', c_real, self.eigenvectors_real) - \
            torch.einsum('btk,kd->btd', c_imag, self.eigenvectors_imag)
        
        return h


class ResonanceCoupling(nn.Module):
    """
    Resonance coupling between eigenstates (Section 3.4).
    
    Implements Equation 3:
        c̃_k(t) = Σ_j R_kj * c_j(t)
    
    The resonance matrix R enables interaction between eigenstates, similar
    to coupled oscillators in dynamical systems.
    
    Stability constraint: R = I + ε*M where ||ε|| << 1
    """
    
    def __init__(self, config: TENConfig):
        super().__init__()
        self.config = config
        K = config.num_eigenstates
        
        if config.use_resonance:
            # R = I + ε*M where M ~ N(0, 1/K) (Appendix B.2)
            self.M_real = nn.Parameter(torch.randn(K, K) / math.sqrt(K))
            self.M_imag = nn.Parameter(torch.randn(K, K) / math.sqrt(K))
        else:
            self.register_buffer('M_real', None)
            self.register_buffer('M_imag', None)
    
    def forward(
        self,
        c_real: torch.Tensor,
        c_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply resonance coupling (Algorithm 1, Line 9).
        
        Args:
            c_real: Real part of amplitudes (batch, seq_len, K)
            c_imag: Imaginary part of amplitudes (batch, seq_len, K)
        
        Returns:
            Coupled amplitudes (c̃_real, c̃_imag)
        """
        if not self.config.use_resonance or self.M_real is None:
            return c_real, c_imag
        
        eps = self.config.resonance_epsilon
        K = self.config.num_eigenstates
        
        # R = I + ε*M
        # R_real = I + ε*M_real, R_imag = ε*M_imag
        # c̃ = R * c (complex matrix-vector multiplication)
        
        # c̃_real = R_real * c_real - R_imag * c_imag
        # c̃_imag = R_real * c_imag + R_imag * c_real
        
        # Identity contribution
        c_tilde_real = c_real.clone()
        c_tilde_imag = c_imag.clone()
        
        # Coupling contribution
        c_tilde_real = c_tilde_real + eps * (
            torch.einsum('btj,kj->btk', c_real, self.M_real) -
            torch.einsum('btj,kj->btk', c_imag, self.M_imag)
        )
        c_tilde_imag = c_tilde_imag + eps * (
            torch.einsum('btj,kj->btk', c_imag, self.M_real) +
            torch.einsum('btj,kj->btk', c_real, self.M_imag)
        )
        
        return c_tilde_real, c_tilde_imag


class TENLayer(nn.Module):
    """
    Complete TEN layer (Section 3.6).
    
    Components:
    1. Input projection: x_t → eigenstate amplitudes
    2. Eigenstate evolution: Apply Eq. 2
    3. Resonance coupling: Mix eigenstates via R
    4. Output projection: Reconstruct hidden state
    5. Feedforward: Standard MLP with residual connection
    6. Layer normalization: Stabilize training
    """
    
    def __init__(self, config: TENConfig):
        super().__init__()
        self.config = config
        
        # Pre-layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Core TEN components
        self.eigenstate_evolution = EigenstateEvolution(config)
        self.resonance_coupling = ResonanceCoupling(config)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Feedforward network (Section 3.6)
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
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        TEN layer forward pass (Algorithm 1).
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            state: Optional previous eigenstate for continuation
            use_parallel: Use parallel scan for training
        
        Returns:
            Tuple of (output, new_state)
        """
        residual = x
        x = self.norm1(x)
        
        # Step 1: Project input to eigenstate space
        beta_real, beta_imag = self.eigenstate_evolution.project_input(x)
        
        # Step 2: Evolve eigenstates through time
        if use_parallel and self.training:
            c_real, c_imag, new_state = self.eigenstate_evolution.evolve_states_parallel(
                beta_real, beta_imag, state
            )
        else:
            c_real, c_imag, new_state = self.eigenstate_evolution.evolve_states(
                beta_real, beta_imag, state
            )
        
        # Step 3: Apply resonance coupling
        c_real, c_imag = self.resonance_coupling(c_real, c_imag)
        
        # Step 4: Reconstruct hidden state
        h = self.eigenstate_evolution.reconstruct(c_real, c_imag)
        
        # Output projection and residual
        h = self.output_proj(h)
        h = self.dropout(h)
        x = residual + h
        
        # Step 5-6: Feedforward with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x, new_state


class TEN(nn.Module):
    """
    Temporal Eigenstate Network (TEN).
    
    Full model stack with:
    - Token embeddings
    - Positional encoding (optional)
    - L TEN layers
    - Output projection
    
    Reference: Section 3.6, 6.1
    """
    
    def __init__(self, config: TENConfig):
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
        
        # TEN layers
        self.layers = nn.ModuleList([
            TENLayer(config) for _ in range(config.num_layers)
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
        states: Optional[list] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, list]:
        """
        TEN forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            states: Optional list of per-layer states for continuation
            use_parallel: Use parallel scan for training
        
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
        
        # Apply TEN layers
        new_states = []
        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            x, new_state = layer(x, layer_state, use_parallel)
            new_states.append(new_state)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x, new_states
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_embedding is not None:
            n_params -= self.pos_embedding.weight.numel()
        return n_params
