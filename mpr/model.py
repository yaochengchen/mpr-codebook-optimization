"""
Objective  y = sum_i a_i |z_i|^2 ,  z_i = d_i + sum_j b_ij X_j(x_j)
with X_j an exact multilinear encoding of a K-bit codebook (see encoding.py).

Closed forms used throughout (g_j = sum_i a_i conj(z_i) b_ij, h_j = sum_i a_i |b_ij|^2):

  gradient        dy/dx_{j,k}         = 2 Re[ g_j * dX_j/dx_{j,k} ]
  exact move gain y(X_j -> X_j + D)   = y + 2 Re[g_j D] + h_j |D|^2

g for all j costs one (M x N) matvec, so the gradient w.r.t. all N*K bits and
the exact gain of ALL single-bit flips (or all 2^K state changes) cost O(MN).
No autograd graph and no N^2 coupling matrix are ever formed.

All methods accept a leading batch dimension P (independent candidates).
"""
from itertools import combinations
import torch

from .encoding import zeta, doc_order


class Objective:
    def __init__(self, a, B, c_bin, d=None):
        """
        a     : (M,) real weights
        B     : (M, N) complex
        c_bin : (2^K,) shared or (N, 2^K) per-element, bitmask order
        d     : (M,) complex offset (e.g. direct path), optional
        """
        self.device = B.device
        self.cdtype = B.dtype
        self.rdtype = a.dtype
        self.a = a
        self.B = B
        self.Bt = B.transpose(0, 1).contiguous()        # (N, M), rows = columns of B
        self.d = d
        self.M, self.N = B.shape
        self.S = c_bin.shape[-1]
        self.K = self.S.bit_length() - 1
        self.shared = c_bin.dim() == 1
        self.c = c_bin.to(self.cdtype)
        self.V = zeta(self.c)                            # response per bit pattern
        self.h = (a[:, None] * (B.real ** 2 + B.imag ** 2)).sum(0)   # (N,)
        # D[..., m, k] = c[..., m | 2^k] if bit k not in m else 0
        D = torch.zeros(*self.c.shape[:-1], self.S, self.K,
                        dtype=self.cdtype, device=self.device)
        for k in range(self.K):
            m = torch.tensor([s for s in range(self.S) if not (s >> k) & 1],
                             device=self.device)
            D[..., m, k] = self.c[..., m | (1 << k)]
        self.D = D
        self.pow2 = (1 << torch.arange(self.K, device=self.device))

    # ------------------------------------------------------------------ basics
    def monomials(self, x):
        """(..., N, K) real -> (..., N, 2^K), column m = prod_{k in m} x_k."""
        mono = torch.ones_like(x[..., :1])
        for k in range(self.K):
            mono = torch.cat([mono, mono * x[..., k:k + 1]], dim=-1)
        return mono.to(self.cdtype)

    def X_relaxed(self, x, with_grad=False):
        mono = self.monomials(x)
        if self.shared:
            X = mono @ self.c
            dX = (mono @ self.D) if with_grad else None
        else:
            X = (mono * self.c).sum(-1)
            dX = torch.einsum('...ns,nsk->...nk', mono, self.D) if with_grad else None
        return X, dX

    def X_states(self, s):
        """s: (..., N) integer bit patterns -> element responses (..., N)."""
        if self.shared:
            return self.V[s]
        return torch.gather(self.V.expand(*s.shape[:-1], -1, -1), -1,
                            s.unsqueeze(-1)).squeeze(-1)

    def bits_to_states(self, bits):
        return (bits.long() * self.pow2).sum(-1)

    def states_to_bits(self, s):
        return ((s.unsqueeze(-1) >> torch.arange(self.K, device=s.device)) & 1)

    def field(self, X):
        z = X @ self.B.transpose(0, 1)
        return z if self.d is None else z + self.d

    def value_from_field(self, z):
        return (self.a * (z.real ** 2 + z.imag ** 2)).sum(-1)

    def g_from_field(self, z):
        return (self.a * z.conj()) @ self.B              # (..., N)

    # ------------------------------------------------------------- evaluation
    def value_states(self, s):
        return self.value_from_field(self.field(self.X_states(s)))

    @torch.no_grad()
    def value_grad(self, x):
        """relaxed x in [0,1]^{N x K}: returns y (...,) and dy/dx (..., N, K)."""
        X, dX = self.X_relaxed(x, with_grad=True)
        z = self.field(X)
        y = self.value_from_field(z)
        g = self.g_from_field(z)
        return y, 2.0 * (g.unsqueeze(-1) * dX).real

    @torch.no_grad()
    def flip_gains(self, s):
        """exact change of y for every single-bit flip: (..., N, K)."""
        X = self.X_states(s)
        g = self.g_from_field(self.field(X))
        out = []
        for k in range(self.K):
            dX = self.X_states(s ^ (1 << k)) - X
            out.append(2 * (g * dX).real + self.h * (dX.real ** 2 + dX.imag ** 2))
        return torch.stack(out, -1)

    # ---------------------------------------------------------- reference impl
    def value_grad_autograd(self, x):
        """Naive PyTorch-autograd baseline: polynomial written term by term in the
        document's order, then backward().  Used only for correctness/timing."""
        x = x.detach().clone().requires_grad_(True)
        X = 0
        for m in doc_order(self.K):
            term = self.c[..., m]
            for k in range(self.K):
                if (m >> k) & 1:
                    term = term * x[..., k]
            X = X + term
        z = self.field(X)
        y = (self.a * z.abs() ** 2).sum(-1)
        y.sum().backward()
        return y.detach(), x.grad
