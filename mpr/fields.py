"""
General field objectives sharing one set of control bits.

    F( |z_1|^2, ..., |z_C|^2 ),   z_c = d_c + B_c X^(c)(x)

Each component c (e.g. a frequency of a dual-band measured codebook) has its own
channel B_c, offset d_c and codebook coefficients c^(c) - all driven by the SAME
bits x.  F is a scalariser of the row powers:

  Quadratic(a_list)  : F = sum_c sum_i a_ci |z_ci|^2       (the paper's eq. (1))
  SoftMinPower(...)  : F = s*tau*log sum_u exp(-(P_u/s)/tau) + leakage penalty
                       (smooth max-min fairness over users; P_u = mean power of
                        user u over its rows / subcarriers)

Chain rule: with row weights w_ci = dF/d|z_ci|^2 (constant for Quadratic,
softmax weights for SoftMinPower)

  dF/dx_jk = sum_c 2 Re[ g_cj dX^(c)_j/dx_jk ],  g_c = B_c^T (w_c * conj z_c)

i.e. exactly the closed form of model.py with data-dependent weights, still one
pair of matrix-vector products per component and step.
Refinement (`refine`) is exact for any scalariser: every alternative state of
element j is evaluated through the updated fields, O(M 2^K) per element.
"""
import torch

from .encoding import zeta


class Quadratic:
    def __init__(self, a_list):
        self.a = a_list

    def value(self, pows):
        return sum((a * p).sum(-1) for a, p in zip(self.a, pows))

    def weights(self, pows):
        return [a.expand_as(p) for a, p in zip(self.a, pows)]


class SoftMinPower:
    """user_idx[c][i] = user of row i in component c (-1 for leakage rows)."""

    def __init__(self, user_idx, n_users, tau=0.1, scale=1.0, leak_weight=1.0):
        self.u = [ui.long() for ui in user_idx]
        self.U = n_users
        self.tau, self.s, self.lw = tau, scale, leak_weight
        dev = self.u[0].device
        cnt = torch.zeros(n_users, device=dev)
        nleak = 0
        for ui in self.u:
            cnt += torch.bincount(ui[ui >= 0], minlength=n_users).to(cnt.dtype)
            nleak += int((ui < 0).sum())
        self.cnt = cnt
        self.nleak = max(nleak, 1)

    def _user_power(self, pows):
        P = None
        for ui, p in zip(self.u, pows):
            m = ui >= 0
            acc = torch.zeros(*p.shape[:-1], self.U, dtype=p.dtype, device=p.device)
            acc.index_add_(-1, ui[m], p[..., m])
            P = acc if P is None else P + acc
        return P / self.cnt.to(P.dtype)

    def _leak(self, pows):
        return sum(p[..., ui < 0].sum(-1) for ui, p in zip(self.u, pows)) / self.nleak

    def value(self, pows):
        P = self._user_power(pows) / self.s
        soft = self.tau * torch.logsumexp(-P / self.tau, dim=-1)        # = -softmin
        return self.s * soft + self.lw * self._leak(pows)

    def weights(self, pows):
        P = self._user_power(pows) / self.s
        wu = torch.softmax(-P / self.tau, dim=-1) / self.cnt.to(P.dtype)  # dF/dP_u * dP_u/dp
        out = []
        for ui, p in zip(self.u, pows):
            w = torch.full_like(p, self.lw / self.nleak)
            m = ui >= 0
            w[..., m] = -torch.gather(wu, -1, ui[m].expand(*p.shape[:-1], -1))
            out.append(w)
        return out

    def min_user_power(self, pows):
        return self._user_power(pows).min(-1).values


class FieldObjective:
    def __init__(self, B_list, c_list, scalarizer, d_list=None):
        self.B = B_list
        self.Bt = [B.transpose(0, 1).contiguous() for B in B_list]
        self.c = c_list
        self.d = d_list or [None] * len(B_list)
        self.F = scalarizer
        self.device = B_list[0].device
        self.cdtype = B_list[0].dtype
        self.rdtype = torch.float32 if self.cdtype == torch.complex64 else torch.float64
        self.N = B_list[0].shape[1]
        self.S = c_list[0].shape[-1]
        self.K = self.S.bit_length() - 1
        self.V = [zeta(c) for c in c_list]
        self.D = []
        for c in c_list:
            D = torch.zeros(*c.shape[:-1], self.S, self.K, dtype=c.dtype, device=c.device)
            for k in range(self.K):
                m = torch.tensor([s for s in range(self.S) if not (s >> k) & 1], device=c.device)
                D[..., m, k] = c[..., m | (1 << k)]
            self.D.append(D)
        self.pow2 = 1 << torch.arange(self.K, device=self.device)
        self.shared = all(c.dim() == 1 for c in c_list)

    # -------------------------------------------------------------- helpers
    def bits_to_states(self, bits):
        return (bits.long() * self.pow2).sum(-1)

    def _monomials(self, x):
        mono = torch.ones_like(x[..., :1])
        for k in range(self.K):
            mono = torch.cat([mono, mono * x[..., k:k + 1]], dim=-1)
        return mono.to(self.cdtype)

    def _X(self, mono, c, D, grad):
        if c.dim() == 1:
            return mono @ c, (mono @ D if grad else None)
        return (mono * c).sum(-1), (torch.einsum('...ns,nsk->...nk', mono, D) if grad else None)

    def _Xs(self, s, V):
        if V.dim() == 1:
            return V[s]
        return torch.gather(V.expand(*s.shape[:-1], -1, -1), -1, s.unsqueeze(-1)).squeeze(-1)

    def fields_states(self, s):
        out = []
        for B, d, V in zip(self.B, self.d, self.V):
            z = self._Xs(s, V) @ B.transpose(0, 1)
            out.append(z if d is None else z + d)
        return out

    @staticmethod
    def _pow(z):
        return z.real ** 2 + z.imag ** 2

    # ----------------------------------------------------------- interface
    def value_states(self, s):
        return self.F.value([self._pow(z) for z in self.fields_states(s)])

    @torch.no_grad()
    def value_grad(self, x):
        mono = self._monomials(x)
        zs, dXs = [], []
        for B, d, c, D in zip(self.B, self.d, self.c, self.D):
            X, dX = self._X(mono, c, D, True)
            z = X @ B.transpose(0, 1)
            zs.append(z if d is None else z + d)
            dXs.append(dX)
        pows = [self._pow(z) for z in zs]
        y = self.F.value(pows)
        grad = 0
        for z, w, B, dX in zip(zs, self.F.weights(pows), self.B, dXs):
            g = (w.to(self.rdtype) * z.conj()) @ B
            grad = grad + 2.0 * (g.unsqueeze(-1) * dX).real
        return y, grad

    @torch.no_grad()
    def refine(self, s, max_sweeps=10, seed=0, tol=1e-9):
        """Exact element-wise best-state refinement for any scalariser."""
        s = s.clone()
        P = s.shape[0]
        gen = torch.Generator().manual_seed(seed)
        zs = self.fields_states(s)
        ar = torch.arange(P, device=self.device)
        for sweep in range(max_sweeps):
            moved = 0
            y_cur = self.F.value([self._pow(z) for z in zs])
            thr = tol * y_cur.abs().mean().item() / self.N
            for j in torch.randperm(self.N, generator=gen).tolist():
                cand, dXs = [], []
                for z, V, Bt in zip(zs, self.V, self.Bt):
                    Vj = V if V.dim() == 1 else V[j]
                    dX = Vj[None, :] - Vj[s[:, j]][:, None]               # (P,S)
                    cand.append(self._pow(z[:, None, :] + dX[..., None] * Bt[j]))
                    dXs.append(dX)
                yc = self.F.value(cand)                                     # (P,S)
                ybest, q = yc.min(-1)
                ycur = yc[ar, s[:, j]]
                move = ybest < ycur - thr
                q = torch.where(move, q, s[:, j])
                for c_, (z, dX, Bt) in enumerate(zip(zs, dXs, self.Bt)):
                    step = torch.where(move, dX[ar, q], torch.zeros_like(dX[:, 0]))
                    zs[c_] = z + step[:, None] * Bt[j][None, :]
                s[:, j] = q
                moved += int(move.sum())
            zs = self.fields_states(s)
            if moved == 0:
                break
        return s, sweep + 1


def from_objective(obj):
    """Wrap a single-component quadratic Objective (model.py) as a FieldObjective."""
    return FieldObjective([obj.B], [obj.c], Quadratic([obj.a]), [obj.d])
