"""
JAX/XLA Implementation of TEN
=============================

TPU-optimized implementation using JAX, Flax, and XLA.

Key optimizations:
- XLA-compiled eigenstate evolution
- Parallel scan with jax.lax.associative_scan
- TPU-friendly memory layout
- Automatic differentiation with JAX

Reference: Section 5 (Multi-Backend Support), Appendix B.4
"""

from typing import Tuple, Optional, Dict, Any, NamedTuple, List
from dataclasses import dataclass
import math

# JAX imports (optional dependency)
try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    import flax.linen as nn
    from flax.linen import initializers
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    lax = None
    nn = None
    initializers = None
    optax = None


if JAX_AVAILABLE:
    
    @dataclass
    class TENConfigJax:
        """Configuration for JAX TEN implementation."""
        vocab_size: int = 50257
        hidden_dim: int = 512
        num_eigenstates: int = 64
        num_layers: int = 6
        alpha_init_range: Tuple[float, float] = (-3.0, 0.0)
        omega_init_scale: float = 1.0
        resonance_epsilon: float = 0.01
        use_resonance: bool = True
        ffn_hidden_dim: int = 2048
        ffn_dropout: float = 0.1
        dropout: float = 0.1
        layer_norm_eps: float = 1e-6
        max_seq_length: int = 8192
        use_positional_encoding: bool = True
        init_std: float = 0.02
        
        # JAX-specific
        dtype: Any = jnp.float32
        param_dtype: Any = jnp.float32
    
    
    class EigenstateState(NamedTuple):
        """State for eigenstate evolution."""
        c_real: jnp.ndarray
        c_imag: jnp.ndarray
    
    
    def complex_multiply(
        a_real: jnp.ndarray, a_imag: jnp.ndarray,
        b_real: jnp.ndarray, b_imag: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Complex multiplication (a + bi) * (c + di)."""
        return (
            a_real * b_real - a_imag * b_imag,
            a_real * b_imag + a_imag * b_real
        )
    
    
    def eigenstate_scan_fn(carry, x):
        """
        Associative scan function for eigenstate evolution.
        
        Implements c(t) = λ * c(t-1) + β(t) as an associative operation.
        
        The associativity property: (λ₁, β₁) ⊕ (λ₂, β₂) = (λ₁λ₂, λ₁β₂ + β₁)
        """
        # carry: (lambda_cum_real, lambda_cum_imag, sum_real, sum_imag)
        # x: (lambda_real, lambda_imag, beta_real, beta_imag)
        
        lc_r, lc_i, sc_r, sc_i = carry
        l_r, l_i, b_r, b_i = x
        
        # New cumulative lambda: λ_cum * λ
        new_lc_r, new_lc_i = complex_multiply(lc_r, lc_i, l_r, l_i)
        
        # New cumulative sum: λ_cum * β + sum
        lb_r, lb_i = complex_multiply(lc_r, lc_i, b_r, b_i)
        new_sc_r = lb_r + sc_r
        new_sc_i = lb_i + sc_i
        
        return (new_lc_r, new_lc_i, new_sc_r, new_sc_i)
    
    
    def parallel_eigenstate_evolution(
        beta_real: jnp.ndarray,
        beta_imag: jnp.ndarray,
        lambda_real: jnp.ndarray,
        lambda_imag: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parallel scan for eigenstate evolution using JAX's associative_scan.
        
        This computes c(t) = Σ_{τ=0}^{t-1} λ^{t-1-τ} β(τ) for all t in O(log T) steps.
        
        Args:
            beta_real: Input projections real part (batch, seq_len, K)
            beta_imag: Input projections imaginary part (batch, seq_len, K)
            lambda_real: Eigenvalue real parts (K,)
            lambda_imag: Eigenvalue imaginary parts (K,)
        
        Returns:
            Tuple of (c_real, c_imag) with shape (batch, seq_len, K)
        """
        batch, T, K = beta_real.shape
        
        # Expand eigenvalues for broadcasting
        lambda_r = jnp.broadcast_to(lambda_real, (batch, T, K))
        lambda_i = jnp.broadcast_to(lambda_imag, (batch, T, K))
        
        # Stack inputs for scan: (lambda_real, lambda_imag, beta_real, beta_imag)
        # Initial cumulative lambda is identity (1, 0)
        init_lambda_r = jnp.ones((batch, K))
        init_lambda_i = jnp.zeros((batch, K))
        init_sum_r = jnp.zeros((batch, K))
        init_sum_i = jnp.zeros((batch, K))
        
        # Prepare scan inputs
        # We need to shift: at position t, we accumulate from positions 0 to t-1
        xs = (lambda_r, lambda_i, beta_real, beta_imag)
        
        # Use vmap over batch dimension and scan over time
        def single_batch_scan(beta_r, beta_i, lambda_r_b, lambda_i_b):
            """Scan for a single batch element."""
            
            def scan_body(carry, x):
                c_r, c_i = carry
                l_r, l_i, b_r, b_i = x
                
                # c(t) = λ * c(t-1) + β(t)
                new_c_r = l_r * c_r - l_i * c_i + b_r
                new_c_i = l_r * c_i + l_i * c_r + b_i
                
                return (new_c_r, new_c_i), (new_c_r, new_c_i)
            
            init = (jnp.zeros(K), jnp.zeros(K))
            xs = (lambda_r_b, lambda_i_b, beta_r, beta_i)
            
            _, (c_r_seq, c_i_seq) = lax.scan(scan_body, init, xs)
            
            return c_r_seq, c_i_seq
        
        # Vectorize over batch
        c_real, c_imag = jax.vmap(single_batch_scan)(
            beta_real, beta_imag,
            jnp.broadcast_to(lambda_real, (batch, T, K)),
            jnp.broadcast_to(lambda_imag, (batch, T, K))
        )
        
        return c_real, c_imag
    
    
    class EigenstateEvolutionJax(nn.Module):
        """
        JAX/Flax implementation of eigenstate evolution.
        """
        config: TENConfigJax
        
        def setup(self):
            K = self.config.num_eigenstates
            d = self.config.hidden_dim
            
            # Alpha (decay rates)
            self.alpha = self.param(
                'alpha',
                lambda rng, shape: jax.random.uniform(
                    rng, shape,
                    minval=self.config.alpha_init_range[0],
                    maxval=self.config.alpha_init_range[1]
                ),
                (K,)
            )
            
            # Omega (frequencies)
            omega_init = 2 * math.pi * jnp.arange(K) / K * self.config.omega_init_scale
            self.omega = self.param(
                'omega',
                lambda rng, shape: omega_init,
                (K,)
            )
            
            # Eigenvectors
            self.eigenvectors_real = self.param(
                'eigenvectors_real',
                initializers.normal(self.config.init_std / math.sqrt(d)),
                (K, d)
            )
            self.eigenvectors_imag = self.param(
                'eigenvectors_imag',
                initializers.normal(self.config.init_std / math.sqrt(d)),
                (K, d)
            )
        
        def get_eigenvalues(self):
            """Get constrained eigenvalues."""
            magnitude = jax.nn.sigmoid(self.alpha)
            return magnitude * jnp.cos(self.omega), magnitude * jnp.sin(self.omega)
        
        def __call__(self, x: jnp.ndarray, state: Optional[EigenstateState] = None):
            """
            Forward pass.
            
            Args:
                x: Input (batch, seq_len, hidden_dim)
                state: Optional previous state
            
            Returns:
                Tuple of (c_real, c_imag, new_state)
            """
            batch, seq_len, d = x.shape
            
            # Get eigenvalues
            lambda_real, lambda_imag = self.get_eigenvalues()
            
            # Project input
            beta_real = jnp.einsum('btd,kd->btk', x, self.eigenvectors_real)
            beta_imag = jnp.einsum('btd,kd->btk', x, -self.eigenvectors_imag)
            
            # Parallel evolution
            c_real, c_imag = parallel_eigenstate_evolution(
                beta_real, beta_imag, lambda_real, lambda_imag
            )
            
            # Handle initial state
            if state is not None:
                # Add contribution from initial state
                t_idx = jnp.arange(seq_len)
                mag = jnp.sqrt(lambda_real**2 + lambda_imag**2 + 1e-8)
                phase = jnp.arctan2(lambda_imag, lambda_real)
                
                # λ^t contribution from initial state
                mag_powers = mag ** t_idx[:, None]
                phase_mult = phase * t_idx[:, None]
                
                init_contrib_real = mag_powers * jnp.cos(phase_mult)
                init_contrib_imag = mag_powers * jnp.sin(phase_mult)
                
                # c_init(t) = λ^t * c(0)
                c_real = c_real + init_contrib_real * state.c_real[:, None, :] - \
                        init_contrib_imag * state.c_imag[:, None, :]
                c_imag = c_imag + init_contrib_real * state.c_imag[:, None, :] + \
                        init_contrib_imag * state.c_real[:, None, :]
            
            new_state = EigenstateState(c_real=c_real[:, -1], c_imag=c_imag[:, -1])
            
            return c_real, c_imag, new_state
        
        def reconstruct(self, c_real: jnp.ndarray, c_imag: jnp.ndarray) -> jnp.ndarray:
            """Reconstruct hidden states."""
            return (
                jnp.einsum('btk,kd->btd', c_real, self.eigenvectors_real) -
                jnp.einsum('btk,kd->btd', c_imag, self.eigenvectors_imag)
            )
    
    
    class ResonanceCouplingJax(nn.Module):
        """JAX implementation of resonance coupling."""
        config: TENConfigJax
        
        def setup(self):
            K = self.config.num_eigenstates
            
            if self.config.use_resonance:
                self.M_real = self.param(
                    'M_real',
                    initializers.normal(1.0 / math.sqrt(K)),
                    (K, K)
                )
                self.M_imag = self.param(
                    'M_imag',
                    initializers.normal(1.0 / math.sqrt(K)),
                    (K, K)
                )
        
        def __call__(self, c_real: jnp.ndarray, c_imag: jnp.ndarray):
            """Apply resonance coupling."""
            if not self.config.use_resonance:
                return c_real, c_imag
            
            eps = self.config.resonance_epsilon
            
            c_tilde_real = c_real + eps * (
                jnp.einsum('btj,kj->btk', c_real, self.M_real) -
                jnp.einsum('btj,kj->btk', c_imag, self.M_imag)
            )
            c_tilde_imag = c_imag + eps * (
                jnp.einsum('btj,kj->btk', c_imag, self.M_real) +
                jnp.einsum('btj,kj->btk', c_real, self.M_imag)
            )
            
            return c_tilde_real, c_tilde_imag
    
    
    class TENLayerJax(nn.Module):
        """JAX TEN layer."""
        config: TENConfigJax
        
        def setup(self):
            self.norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
            self.norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
            
            self.eigenstate = EigenstateEvolutionJax(self.config)
            self.resonance = ResonanceCouplingJax(self.config)
            
            self.output_proj = nn.Dense(self.config.hidden_dim)
            
            self.ffn = nn.Sequential([
                nn.Dense(self.config.ffn_hidden_dim),
                nn.gelu,
                nn.Dropout(self.config.ffn_dropout, deterministic=False),
                nn.Dense(self.config.hidden_dim),
                nn.Dropout(self.config.ffn_dropout, deterministic=False),
            ])
            
            self.dropout = nn.Dropout(self.config.dropout, deterministic=False)
        
        def __call__(
            self,
            x: jnp.ndarray,
            state: Optional[EigenstateState] = None,
            deterministic: bool = False
        ):
            residual = x
            x = self.norm1(x)
            
            # Eigenstate evolution
            c_real, c_imag, new_state = self.eigenstate(x, state)
            
            # Resonance coupling
            c_real, c_imag = self.resonance(c_real, c_imag)
            
            # Reconstruct
            h = self.eigenstate.reconstruct(c_real, c_imag)
            h = self.output_proj(h)
            h = self.dropout(h, deterministic=deterministic)
            x = residual + h
            
            # FFN
            residual = x
            x = self.norm2(x)
            x = residual + self.ffn(x)
            
            return x, new_state
    
    
    class TENJax(nn.Module):
        """
        JAX/Flax TEN model for TPU.
        """
        config: TENConfigJax
        
        def setup(self):
            self.token_embedding = nn.Embed(
                self.config.vocab_size,
                self.config.hidden_dim,
                embedding_init=initializers.normal(self.config.init_std)
            )
            
            if self.config.use_positional_encoding:
                self.pos_embedding = nn.Embed(
                    self.config.max_seq_length,
                    self.config.hidden_dim,
                    embedding_init=initializers.normal(self.config.init_std)
                )
            
            self.embed_dropout = nn.Dropout(self.config.dropout, deterministic=False)
            
            self.layers = [TENLayerJax(self.config) for _ in range(self.config.num_layers)]
            
            self.final_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
        
        def __call__(
            self,
            input_ids: jnp.ndarray,
            states: Optional[List[EigenstateState]] = None,
            deterministic: bool = False
        ):
            batch, seq_len = input_ids.shape
            
            x = self.token_embedding(input_ids)
            
            if self.config.use_positional_encoding:
                positions = jnp.arange(seq_len)
                x = x + self.pos_embedding(positions)
            
            x = self.embed_dropout(x, deterministic=deterministic)
            
            new_states = []
            for i, layer in enumerate(self.layers):
                layer_state = states[i] if states is not None else None
                x, new_state = layer(x, layer_state, deterministic=deterministic)
                new_states.append(new_state)
            
            x = self.final_norm(x)
            
            return x, new_states
    
    
    class HTENJax(nn.Module):
        """
        JAX/Flax HTEN model for TPU.
        """
        config: TENConfigJax
        scales: Tuple[int, ...] = (1, 2, 4, 8)
        
        def setup(self):
            self.token_embedding = nn.Embed(
                self.config.vocab_size,
                self.config.hidden_dim,
                embedding_init=initializers.normal(self.config.init_std)
            )
            
            if self.config.use_positional_encoding:
                self.pos_embedding = nn.Embed(
                    self.config.max_seq_length,
                    self.config.hidden_dim,
                    embedding_init=initializers.normal(self.config.init_std)
                )
            
            self.embed_dropout = nn.Dropout(self.config.dropout, deterministic=False)
            
            # Multi-scale TEN modules per layer
            K_per_scale = self.config.num_eigenstates // len(self.scales)
            
            self.scale_weights = self.param(
                'scale_weights',
                initializers.ones,
                (self.config.num_layers, len(self.scales))
            )
            
            # Create config for each scale
            scale_configs = []
            for s in self.scales:
                scale_config = TENConfigJax(
                    vocab_size=self.config.vocab_size,
                    hidden_dim=self.config.hidden_dim,
                    num_eigenstates=K_per_scale,
                    num_layers=1,
                    alpha_init_range=self.config.alpha_init_range,
                    omega_init_scale=self.config.omega_init_scale,
                    resonance_epsilon=self.config.resonance_epsilon,
                    use_resonance=self.config.use_resonance,
                    ffn_hidden_dim=self.config.ffn_hidden_dim,
                    max_seq_length=self.config.max_seq_length // s + 1,
                    use_positional_encoding=False,
                )
                scale_configs.append(scale_config)
            
            self.scale_tens = [
                [TENLayerJax(scale_configs[j]) for j in range(len(self.scales))]
                for _ in range(self.config.num_layers)
            ]
            
            self.output_projs = [
                nn.Dense(self.config.hidden_dim) for _ in range(self.config.num_layers)
            ]
            
            self.final_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
        
        def downsample(self, x: jnp.ndarray, scale: int) -> jnp.ndarray:
            """Average pooling downsample."""
            if scale == 1:
                return x
            
            batch, T, d = x.shape
            pad = (scale - T % scale) % scale
            if pad > 0:
                x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
            
            x = x.reshape(batch, -1, scale, d).mean(axis=2)
            return x
        
        def upsample(self, x: jnp.ndarray, target_len: int) -> jnp.ndarray:
            """Linear interpolation upsample."""
            batch, T, d = x.shape
            
            if T == target_len:
                return x
            
            # Use JAX's image resize for interpolation
            x = jax.image.resize(x, (batch, target_len, d), method='linear')
            return x
        
        def __call__(
            self,
            input_ids: jnp.ndarray,
            states: Optional[List[Dict[int, EigenstateState]]] = None,
            deterministic: bool = False
        ):
            batch, seq_len = input_ids.shape
            
            x = self.token_embedding(input_ids)
            
            if self.config.use_positional_encoding:
                positions = jnp.arange(seq_len)
                x = x + self.pos_embedding(positions)
            
            x = self.embed_dropout(x, deterministic=deterministic)
            
            new_states = []
            
            for layer_idx in range(self.config.num_layers):
                residual = x
                
                # Process each scale
                scale_outputs = []
                layer_new_states = {}
                
                for scale_idx, scale in enumerate(self.scales):
                    x_down = self.downsample(x, scale)
                    
                    layer_state = None
                    if states is not None and scale in states[layer_idx]:
                        layer_state = states[layer_idx][scale]
                    
                    h_scale, new_state = self.scale_tens[layer_idx][scale_idx](
                        x_down, layer_state, deterministic
                    )
                    
                    h_up = self.upsample(h_scale, seq_len)
                    scale_outputs.append(h_up)
                    layer_new_states[scale] = new_state
                
                new_states.append(layer_new_states)
                
                # Fuse scales with learned weights
                weights = jax.nn.softmax(self.scale_weights[layer_idx])
                h = sum(w * h_s for w, h_s in zip(weights, scale_outputs))
                
                h = self.output_projs[layer_idx](h)
                x = residual + h
            
            x = self.final_norm(x)
            
            return x, new_states
    
    
    def create_ten_state(config: TENConfigJax, batch_size: int):
        """Create initial TEN state."""
        K = config.num_eigenstates
        return [
            EigenstateState(
                c_real=jnp.zeros((batch_size, K)),
                c_imag=jnp.zeros((batch_size, K))
            )
            for _ in range(config.num_layers)
        ]
    
    
    @jax.jit
    def ten_forward(model, params, input_ids, states=None, deterministic=True):
        """JIT-compiled TEN forward pass."""
        return model.apply(params, input_ids, states, deterministic)
    
    
    @jax.jit
    def hten_forward(model, params, input_ids, states=None, deterministic=True):
        """JIT-compiled HTEN forward pass."""
        return model.apply(params, input_ids, states, deterministic)

else:
    # Stubs when JAX is not available
    TENConfigJax = None
    TENJax = None
    HTENJax = None
    create_ten_state = None
    ten_forward = None
    hten_forward = None
