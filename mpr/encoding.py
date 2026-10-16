"""
Exact multilinear encoding of a K-bit codebook.

A K-bit element has 2^K states with complex responses v_q (q = 0..2^K-1).
Its K control bits x in {0,1}^K select the state q = map(s), where s is the
integer whose bit k is x_k.  Any such table V[s] = v_{map(s)} is represented
EXACTLY by a multilinear polynomial

    X(x) = sum_{S subset of {0..K-1}} c_S prod_{k in S} x_k ,

with coefficients given by the Moebius (subset-sum inverse) transform

    c_S = sum_{T subset S} (-1)^{|S|-|T|} V[T].

Coefficients here are indexed by bitmask S (binary order).  `binary_to_doc`
reorders them to the graded-lexicographic order used in the original note
(c0, c1 x1, c2 x2, c3 x3, c4 x1x2, c5 x1x3, c6 x2x3, c7 x1x2x3, ...).
"""
from itertools import combinations
import math
import torch


# ----------------------------------------------------------------------------
# bit-pattern -> state mappings
def gray_decode(s: int) -> int:
    q = 0
    while s:
        q ^= s
        s >>= 1
    return q


def bit_to_state_map(K: int, mapping: str = "gray") -> torch.Tensor:
    """state index q selected by bit pattern s, for s = 0..2^K-1."""
    S = 1 << K
    if mapping == "binary":
        return torch.arange(S)
    if mapping == "gray":
        # adjacent phase states differ in exactly one bit
        return torch.tensor([gray_decode(s) for s in range(S)])
    raise ValueError(mapping)


# ----------------------------------------------------------------------------
# codebooks (length 2^K complex tensors, indexed by state q)
def ideal_phase_codebook(K: int, dtype=torch.complex128) -> torch.Tensor:
    q = torch.arange(1 << K, dtype=torch.float64)
    return torch.exp(1j * 2 * math.pi * q / (1 << K)).to(dtype)


def practical_ris_codebook(K: int, beta_min=0.2, phi=0.43 * math.pi, alpha=1.6,
                           dtype=torch.complex128) -> torch.Tensor:
    """Phase-dependent amplitude model of Abeywickrama et al. (IEEE TCOM 2020):
    beta(theta) = (1-beta_min) * ((sin(theta - phi) + 1)/2)^alpha + beta_min.
    Default parameters are illustrative; replace with measured unit-cell data."""
    q = torch.arange(1 << K, dtype=torch.float64)
    theta = -math.pi + 2 * math.pi * q / (1 << K)
    beta = (1 - beta_min) * ((torch.sin(theta - phi) + 1) / 2) ** alpha + beta_min
    return (beta * torch.exp(1j * theta)).to(dtype)


def perturbed_codebook(K: int, phase_err_deg=10.0, amp_err=0.1, seed=0,
                       dtype=torch.complex128) -> torch.Tensor:
    """Ideal phases with fixed random phase/amplitude errors (fabrication spread)."""
    g = torch.Generator().manual_seed(seed)
    v = ideal_phase_codebook(K, torch.complex128)
    dphi = torch.randn(1 << K, generator=g, dtype=torch.float64) * math.radians(phase_err_deg)
    amp = 1 + amp_err * torch.randn(1 << K, generator=g, dtype=torch.float64)
    return (v * amp * torch.exp(1j * dphi)).to(dtype)


# ----------------------------------------------------------------------------
def moebius(V: torch.Tensor) -> torch.Tensor:
    """In-place-style fast Moebius transform over the last dim (size 2^K)."""
    c = V.clone()
    S = c.shape[-1]
    K = S.bit_length() - 1
    for k in range(K):
        b = 1 << k
        idx = torch.tensor([s for s in range(S) if s & b], device=c.device)
        c[..., idx] = c[..., idx] - c[..., idx ^ b]
    return c


def zeta(c: torch.Tensor) -> torch.Tensor:
    """Inverse of `moebius`: V[s] = sum_{T subset s} c_T."""
    V = c.clone()
    S = V.shape[-1]
    K = S.bit_length() - 1
    for k in range(K):
        b = 1 << k
        idx = torch.tensor([s for s in range(S) if s & b], device=V.device)
        V[..., idx] = V[..., idx] + V[..., idx ^ b]
    return V


def codebook_to_coeffs(codebook: torch.Tensor, mapping: str = "gray") -> torch.Tensor:
    """codebook: (..., 2^K) indexed by state.  Returns bitmask-ordered c (..., 2^K)."""
    K = codebook.shape[-1].bit_length() - 1
    m = bit_to_state_map(K, mapping).to(codebook.device)
    V = codebook[..., m]           # V[s] = v_{map(s)}
    return moebius(V)


def doc_order(K: int):
    """bitmasks in graded-lexicographic order."""
    return [sum(1 << l for l in combo)
            for d in range(K + 1) for combo in combinations(range(K), d)]


def binary_to_doc(c_bin: torch.Tensor) -> torch.Tensor:
    K = c_bin.shape[-1].bit_length() - 1
    return c_bin[..., torch.tensor(doc_order(K), device=c_bin.device)]


def doc_to_binary(c_doc: torch.Tensor) -> torch.Tensor:
    K = c_doc.shape[-1].bit_length() - 1
    out = torch.empty_like(c_doc)
    out[..., torch.tensor(doc_order(K), device=c_doc.device)] = c_doc
    return out
