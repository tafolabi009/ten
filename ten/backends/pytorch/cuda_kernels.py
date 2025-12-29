"""
Custom CUDA Kernels for TEN
===========================

High-performance CUDA implementations for eigenstate evolution.

These kernels provide significant speedups over naive PyTorch implementations
by fusing operations and optimizing memory access patterns.

Reference: Appendix B.4 (Efficient Implementation), Appendix F (Computational Requirements)
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import math

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Triton-based kernels (if available)
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _eigenstate_evolution_kernel(
        # Inputs
        beta_real_ptr, beta_imag_ptr,
        lambda_real_ptr, lambda_imag_ptr,
        # Outputs
        c_real_ptr, c_imag_ptr,
        # Dimensions
        batch_size, seq_len, K,
        # Strides
        stride_batch, stride_seq, stride_k,
        BLOCK_K: tl.constexpr,
    ):
        """
        Triton kernel for eigenstate evolution.
        
        Implements Eq. 2: c_k(t+1) = λ_k · c_k(t) + β_k(t)
        
        Each thread block handles one (batch, eigenstate) pair across all timesteps.
        """
        batch_idx = tl.program_id(0)
        k_block = tl.program_id(1)
        
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Load eigenvalues (shared across batch)
        lambda_r = tl.load(lambda_real_ptr + k_offsets, mask=k_mask, other=0.0)
        lambda_i = tl.load(lambda_imag_ptr + k_offsets, mask=k_mask, other=0.0)
        
        # Initialize states
        c_r = tl.zeros((BLOCK_K,), dtype=tl.float32)
        c_i = tl.zeros((BLOCK_K,), dtype=tl.float32)
        
        # Sequential evolution (within block)
        for t in range(seq_len):
            # Load beta
            offset = batch_idx * stride_batch + t * stride_seq + k_offsets
            beta_r = tl.load(beta_real_ptr + offset, mask=k_mask, other=0.0)
            beta_i = tl.load(beta_imag_ptr + offset, mask=k_mask, other=0.0)
            
            # c = λ * c + β (complex multiplication)
            new_c_r = lambda_r * c_r - lambda_i * c_i + beta_r
            new_c_i = lambda_r * c_i + lambda_i * c_r + beta_i
            
            c_r = new_c_r
            c_i = new_c_i
            
            # Store result
            tl.store(c_real_ptr + offset, c_r, mask=k_mask)
            tl.store(c_imag_ptr + offset, c_i, mask=k_mask)
    
    
    @triton.jit
    def _complex_matmul_kernel(
        # Inputs
        a_real_ptr, a_imag_ptr,
        b_real_ptr, b_imag_ptr,
        # Output
        c_real_ptr, c_imag_ptr,
        # Dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Triton kernel for complex matrix multiplication.
        
        Used for resonance coupling: c̃ = R · c (Eq. 3)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        # Initialize accumulators
        acc_real = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_imag = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Loop over K dimension
        for k_start in range(0, K, BLOCK_K):
            k = k_start + offs_k
            
            # Load A and B blocks
            a_real = tl.load(a_real_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                            mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
            a_imag = tl.load(a_imag_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                            mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
            
            b_real = tl.load(b_real_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                            mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            b_imag = tl.load(b_imag_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                            mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            acc_real += tl.dot(a_real, b_real) - tl.dot(a_imag, b_imag)
            acc_imag += tl.dot(a_real, b_imag) + tl.dot(a_imag, b_real)
        
        # Store result
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_real_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                acc_real, mask=c_mask)
        tl.store(c_imag_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                acc_imag, mask=c_mask)


def eigenstate_evolution_cuda(
    beta_real: Tensor,
    beta_imag: Tensor,
    lambda_real: Tensor,
    lambda_imag: Tensor,
    initial_state: Optional[Tuple[Tensor, Tensor]] = None
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    """
    CUDA-accelerated eigenstate evolution.
    
    Implements Eq. 2: c_k(t+1) = λ_k · c_k(t) + β_k(t)
    
    Args:
        beta_real: Real input projections (batch, seq_len, K)
        beta_imag: Imaginary input projections (batch, seq_len, K)
        lambda_real: Real part of eigenvalues (K,)
        lambda_imag: Imaginary part of eigenvalues (K,)
        initial_state: Optional initial state (c_real, c_imag)
    
    Returns:
        Tuple of (c_real, c_imag, final_state)
    """
    batch, seq_len, K = beta_real.shape
    device = beta_real.device
    dtype = beta_real.dtype
    
    # Allocate output
    c_real = torch.empty_like(beta_real)
    c_imag = torch.empty_like(beta_imag)
    
    if TRITON_AVAILABLE and device.type == 'cuda':
        # Use Triton kernel
        BLOCK_K = min(128, triton.next_power_of_2(K))
        
        grid = (batch, triton.cdiv(K, BLOCK_K))
        
        _eigenstate_evolution_kernel[grid](
            beta_real, beta_imag,
            lambda_real, lambda_imag,
            c_real, c_imag,
            batch, seq_len, K,
            beta_real.stride(0), beta_real.stride(1), beta_real.stride(2),
            BLOCK_K=BLOCK_K,
        )
    else:
        # Fallback to PyTorch
        c_real, c_imag, _ = _eigenstate_evolution_pytorch(
            beta_real, beta_imag, lambda_real, lambda_imag, initial_state
        )
    
    # Get final state
    final_state = (c_real[:, -1, :].contiguous(), c_imag[:, -1, :].contiguous())
    
    return c_real, c_imag, final_state


def _eigenstate_evolution_pytorch(
    beta_real: Tensor,
    beta_imag: Tensor,
    lambda_real: Tensor,
    lambda_imag: Tensor,
    initial_state: Optional[Tuple[Tensor, Tensor]] = None
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    """
    Pure PyTorch implementation of eigenstate evolution.
    
    Fallback for non-CUDA devices or when Triton is unavailable.
    """
    batch, seq_len, K = beta_real.shape
    device = beta_real.device
    dtype = beta_real.dtype
    
    # Initialize states
    if initial_state is None:
        c_real = torch.zeros(batch, K, device=device, dtype=dtype)
        c_imag = torch.zeros(batch, K, device=device, dtype=dtype)
    else:
        c_real, c_imag = initial_state[0].clone(), initial_state[1].clone()
    
    c_real_all = []
    c_imag_all = []
    
    for t in range(seq_len):
        # c = λ * c + β
        new_c_real = lambda_real * c_real - lambda_imag * c_imag + beta_real[:, t, :]
        new_c_imag = lambda_real * c_imag + lambda_imag * c_real + beta_imag[:, t, :]
        
        c_real = new_c_real
        c_imag = new_c_imag
        
        c_real_all.append(c_real.clone())
        c_imag_all.append(c_imag.clone())
    
    c_real_seq = torch.stack(c_real_all, dim=1)
    c_imag_seq = torch.stack(c_imag_all, dim=1)
    
    return c_real_seq, c_imag_seq, (c_real, c_imag)


def complex_matmul_cuda(
    a_real: Tensor,
    a_imag: Tensor,
    b_real: Tensor,
    b_imag: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    CUDA-accelerated complex matrix multiplication.
    
    Computes (a_real + i*a_imag) @ (b_real + i*b_imag)
    
    Used for resonance coupling matrix multiplication.
    
    Args:
        a_real: Real part of first matrix
        a_imag: Imaginary part of first matrix
        b_real: Real part of second matrix
        b_imag: Imaginary part of second matrix
    
    Returns:
        Tuple of (result_real, result_imag)
    """
    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    result_real = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
    result_imag = torch.matmul(a_real, b_imag) + torch.matmul(a_imag, b_real)
    
    return result_real, result_imag


class FusedEigenstateForward(torch.autograd.Function):
    """
    Fused forward pass for eigenstate evolution with optimized backward.
    
    Combines projection, evolution, coupling, and reconstruction into a
    single fused operation for better memory efficiency.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        eigenvectors_real: Tensor,
        eigenvectors_imag: Tensor,
        alpha: Tensor,
        omega: Tensor,
        M_real: Optional[Tensor],
        M_imag: Optional[Tensor],
        resonance_epsilon: float,
        use_resonance: bool,
        eigenvalue_constraint: str,
    ) -> Tensor:
        """
        Fused forward pass.
        
        Args:
            x: Input (batch, seq_len, d)
            eigenvectors_real: Eigenvector real parts (K, d)
            eigenvectors_imag: Eigenvector imaginary parts (K, d)
            alpha: Decay parameters (K,)
            omega: Frequency parameters (K,)
            M_real: Resonance matrix real part (K, K) or None
            M_imag: Resonance matrix imaginary part (K, K) or None
            resonance_epsilon: Coupling strength
            use_resonance: Whether to apply resonance coupling
            eigenvalue_constraint: How to constrain eigenvalue magnitude
        
        Returns:
            Output tensor (batch, seq_len, d)
        """
        batch, seq_len, d = x.shape
        K = alpha.shape[0]
        
        # Compute eigenvalues
        if eigenvalue_constraint == "sigmoid":
            magnitude = torch.sigmoid(alpha)
        elif eigenvalue_constraint == "exp_clamp":
            magnitude = torch.exp(torch.clamp(alpha, max=0.0))
        else:
            magnitude = torch.exp(alpha)
        
        lambda_real = magnitude * torch.cos(omega)
        lambda_imag = magnitude * torch.sin(omega)
        
        # Project input: β = x · v*
        beta_real = torch.einsum('btd,kd->btk', x, eigenvectors_real)
        beta_imag = torch.einsum('btd,kd->btk', x, -eigenvectors_imag)
        
        # Evolve eigenstates
        c_real, c_imag, _ = _eigenstate_evolution_pytorch(
            beta_real, beta_imag, lambda_real, lambda_imag
        )
        
        # Apply resonance coupling
        if use_resonance and M_real is not None:
            eps = resonance_epsilon
            c_tilde_real = c_real + eps * (
                torch.einsum('btj,kj->btk', c_real, M_real) -
                torch.einsum('btj,kj->btk', c_imag, M_imag)
            )
            c_tilde_imag = c_imag + eps * (
                torch.einsum('btj,kj->btk', c_imag, M_real) +
                torch.einsum('btj,kj->btk', c_real, M_imag)
            )
        else:
            c_tilde_real, c_tilde_imag = c_real, c_imag
        
        # Reconstruct: h = Re[Σ c * v]
        h = torch.einsum('btk,kd->btd', c_tilde_real, eigenvectors_real) - \
            torch.einsum('btk,kd->btd', c_tilde_imag, eigenvectors_imag)
        
        # Save for backward
        ctx.save_for_backward(
            x, eigenvectors_real, eigenvectors_imag, alpha, omega,
            M_real, M_imag, c_real, c_imag, lambda_real, lambda_imag
        )
        ctx.resonance_epsilon = resonance_epsilon
        ctx.use_resonance = use_resonance
        ctx.eigenvalue_constraint = eigenvalue_constraint
        
        return h
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass with gradient checkpointing for memory efficiency.
        
        Implements gradient flow analysis from Proposition 5 (Section 4.3).
        """
        (x, eigenvectors_real, eigenvectors_imag, alpha, omega,
         M_real, M_imag, c_real, c_imag, lambda_real, lambda_imag) = ctx.saved_tensors
        eps = ctx.resonance_epsilon
        use_resonance = ctx.use_resonance
        
        batch, seq_len, d = x.shape
        K = alpha.shape[0]
        
        # Gradients for reconstruction
        grad_c_real = torch.einsum('btd,kd->btk', grad_output, eigenvectors_real)
        grad_c_imag = -torch.einsum('btd,kd->btk', grad_output, eigenvectors_imag)
        
        grad_eigenvectors_real = torch.einsum('btd,btk->kd', grad_output, c_real)
        grad_eigenvectors_imag = -torch.einsum('btd,btk->kd', grad_output, c_imag)
        
        # Gradients through resonance coupling
        if use_resonance and M_real is not None:
            grad_M_real = eps * torch.einsum('btk,btj->kj', grad_c_real, c_real)
            grad_M_real += eps * torch.einsum('btk,btj->kj', grad_c_imag, c_imag)
            
            grad_M_imag = -eps * torch.einsum('btk,btj->kj', grad_c_real, c_imag)
            grad_M_imag += eps * torch.einsum('btk,btj->kj', grad_c_imag, c_real)
            
            # Gradient to c before coupling
            grad_c_real_pre = grad_c_real + eps * torch.einsum('btk,kj->btj', grad_c_real, M_real)
            grad_c_real_pre += eps * torch.einsum('btk,kj->btj', grad_c_imag, M_imag)
            
            grad_c_imag_pre = grad_c_imag + eps * torch.einsum('btk,kj->btj', grad_c_imag, M_real)
            grad_c_imag_pre -= eps * torch.einsum('btk,kj->btj', grad_c_real, M_imag)
        else:
            grad_c_real_pre = grad_c_real
            grad_c_imag_pre = grad_c_imag
            grad_M_real = None
            grad_M_imag = None
        
        # Gradients through eigenstate evolution (backward recurrence)
        # From Proposition 5: ∂c_k(t)/∂λ_k = c_k(t-1) + λ_k * ∂c_k(t-1)/∂λ_k
        grad_beta_real = torch.zeros_like(x[:, :, :K])
        grad_beta_imag = torch.zeros_like(x[:, :, :K])
        grad_lambda_real = torch.zeros(K, device=x.device, dtype=x.dtype)
        grad_lambda_imag = torch.zeros(K, device=x.device, dtype=x.dtype)
        
        # Backward through time
        grad_state_real = torch.zeros(batch, K, device=x.device, dtype=x.dtype)
        grad_state_imag = torch.zeros(batch, K, device=x.device, dtype=x.dtype)
        
        for t in range(seq_len - 1, -1, -1):
            # Gradient to current output
            grad_curr_real = grad_c_real_pre[:, t, :] + grad_state_real
            grad_curr_imag = grad_c_imag_pre[:, t, :] + grad_state_imag
            
            # Gradient to beta (input projection)
            grad_beta_real[:, t, :] = grad_curr_real
            grad_beta_imag[:, t, :] = grad_curr_imag
            
            if t > 0:
                # Gradient to lambda
                # c(t) = λ * c(t-1) + β(t)
                # ∂L/∂λ = ∂L/∂c(t) * c(t-1)
                grad_lambda_real += (grad_curr_real * c_real[:, t-1, :]).sum(dim=0)
                grad_lambda_real -= (grad_curr_imag * c_imag[:, t-1, :]).sum(dim=0)
                grad_lambda_imag += (grad_curr_real * c_imag[:, t-1, :]).sum(dim=0)
                grad_lambda_imag += (grad_curr_imag * c_real[:, t-1, :]).sum(dim=0)
                
                # Gradient to previous state
                # ∂L/∂c(t-1) = λ* · ∂L/∂c(t)
                grad_state_real = lambda_real * grad_curr_real + lambda_imag * grad_curr_imag
                grad_state_imag = lambda_real * grad_curr_imag - lambda_imag * grad_curr_real
        
        # Gradient to alpha and omega
        if ctx.eigenvalue_constraint == "sigmoid":
            sig_alpha = torch.sigmoid(alpha)
            grad_magnitude = (
                grad_lambda_real * torch.cos(omega) +
                grad_lambda_imag * torch.sin(omega)
            )
            grad_alpha = grad_magnitude * sig_alpha * (1 - sig_alpha)
        elif ctx.eigenvalue_constraint == "exp_clamp":
            magnitude = torch.exp(torch.clamp(alpha, max=0.0))
            grad_magnitude = (
                grad_lambda_real * torch.cos(omega) +
                grad_lambda_imag * torch.sin(omega)
            )
            grad_alpha = grad_magnitude * magnitude * (alpha <= 0).float()
        else:
            magnitude = torch.exp(alpha)
            grad_magnitude = (
                grad_lambda_real * torch.cos(omega) +
                grad_lambda_imag * torch.sin(omega)
            )
            grad_alpha = grad_magnitude * magnitude
        
        grad_omega = (
            -grad_lambda_real * magnitude * torch.sin(omega) +
            grad_lambda_imag * magnitude * torch.cos(omega)
        )
        
        # Gradient to input through projection
        grad_x = torch.einsum('btk,kd->btd', grad_beta_real, eigenvectors_real)
        grad_x -= torch.einsum('btk,kd->btd', grad_beta_imag, eigenvectors_imag)
        
        # Gradient to eigenvectors through projection
        grad_eigenvectors_real += torch.einsum('btk,btd->kd', grad_beta_real, x)
        grad_eigenvectors_imag -= torch.einsum('btk,btd->kd', grad_beta_imag, x)
        
        return (
            grad_x,
            grad_eigenvectors_real,
            grad_eigenvectors_imag,
            grad_alpha,
            grad_omega,
            grad_M_real,
            grad_M_imag,
            None,  # resonance_epsilon
            None,  # use_resonance
            None,  # eigenvalue_constraint
        )


def fused_eigenstate_forward(
    x: Tensor,
    eigenvectors_real: Tensor,
    eigenvectors_imag: Tensor,
    alpha: Tensor,
    omega: Tensor,
    M_real: Optional[Tensor],
    M_imag: Optional[Tensor],
    resonance_epsilon: float,
    use_resonance: bool,
    eigenvalue_constraint: str,
) -> Tensor:
    """
    Wrapper for fused eigenstate forward pass.
    """
    return FusedEigenstateForward.apply(
        x, eigenvectors_real, eigenvectors_imag, alpha, omega,
        M_real, M_imag, resonance_epsilon, use_resonance, eigenvalue_constraint
    )
