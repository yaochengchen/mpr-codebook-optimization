"""
Optimisers.  Every routine works on a batch of P independent candidates and
returns the best one.  Objective is MINIMISED.

  mpr(...)            proposed: multilinear-polynomial relaxation
                      (Adam on x in [0,1]^{NK} + annealed binarisation penalty,
                       closed-form gradient) -> rounding -> exact refinement
  successive_refinement(...)  element-wise exact best-state update (Wu & Zhang
                      style), O(M) per element using the move-gain formula
  cont_quantize(...)  continuous unit-modulus phase optimisation -> nearest-phase
                      quantisation (optionally + refinement)
  genetic(...)        batched GA over bit strings
  random_search(...)  best of P random codebook configurations
"""
import math
import time
import torch


def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()


def _result(obj, s, t0, name, **extra):
    _sync(obj.device)
    y = obj.value_states(s)
    i = torch.argmin(y)
    return dict(name=name, y=y[i].item(), s=s[i].clone(), y_all=y,
                time=time.perf_counter() - t0, **extra)


# ----------------------------------------------------------------------------
@torch.no_grad()
def successive_refinement(obj, s, max_sweeps=10, tol=0.0, order="random", seed=0):
    """s: (P, N) bit patterns.  In each sweep every element in turn is moved to
    the codebook state with the largest exact decrease of y.  Monotone."""
    s = s.clone()
    P = s.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = obj.field(obj.X_states(s))
    ar = torch.arange(P, device=obj.device)
    if obj.shared:
        Vall = obj.V                                          # (S,)
    for sweep in range(max_sweeps):
        perm = torch.randperm(obj.N, generator=gen) if order == "random" else torch.arange(obj.N)
        improved = torch.zeros((), device=obj.device)
        # relative threshold avoids flip-flopping on round-off
        thr = max(tol, 1e-9 if obj.rdtype == torch.float64 else 1e-6) * (obj.a.abs() * (z.real ** 2 + z.imag ** 2)).sum(-1).mean().item() / obj.N
        for j in perm.tolist():
            bj = obj.Bt[j]                                    # (M,)
            gj = ((obj.a * z.conj()) * bj).sum(-1)            # (P,)
            Vj = Vall if obj.shared else obj.V[j]
            cur = Vj[s[:, j]]                                 # (P,)
            dX = Vj[None, :] - cur[:, None]                   # (P, S)
            gain = 2 * (gj[:, None] * dX).real + obj.h[j] * (dX.real ** 2 + dX.imag ** 2)
            gmin, q = gain.min(-1)
            move = gmin < -thr                                # (P,)
            q = torch.where(move, q, s[:, j])
            dsel = torch.where(move, dX[ar, q], torch.zeros_like(cur))
            z = z + dsel[:, None] * bj[None, :]
            s[:, j] = q
            improved = improved + move.sum()
        z = obj.field(obj.X_states(s))                        # remove drift
        if improved.item() == 0:
            break
    return s, sweep + 1


# ----------------------------------------------------------------------------
@torch.no_grad()
def diversify(obj, s, y, min_frac=0.02, seed=0):
    """Replace near-duplicate candidates by random configurations.

    s: (P, N) rounded states, y: (P,) their objective.  Candidates are visited
    from best to worst; one is kept if it differs from every kept candidate in at
    least max(1, min_frac * N * K) control bits, otherwise it is replaced by a
    uniformly random configuration.  When the relaxation keeps its candidates
    apart nothing changes; when they collapse (e.g. a dominant offset d makes the
    gradient nearly independent of the start) the refinement stage gets back the
    diversity of random starts.  Returns (s, number of replaced candidates)."""
    P, N = s.shape
    bits = ((s[..., None] >> torch.arange(obj.K, device=s.device)) & 1).reshape(P, -1).to(torch.int16)
    dmin = max(1, int(round(min_frac * N * obj.K)))
    kept = []
    for i in torch.argsort(y).tolist():
        if all(int((bits[i] - bits[k]).abs().sum()) >= dmin for k in kept):
            kept.append(i)
    rep = [i for i in range(P) if i not in set(kept)]
    if rep:
        g = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        s = s.clone()
        s[rep] = torch.randint(0, obj.S, (len(rep), N), generator=g).to(s.device, s.dtype)
    return s, len(rep)


def mpr(obj, P=64, iters=300, lr=0.05, lam_max=2.0, refine_sweeps=5, seed=0, init_spread=1.0,
        grad_fn=None, return_trace=False, dedup=False, dedup_frac=0.02):
    """Proposed method.  lam_max is the final penalty weight relative to the
    per-bit scale |y_0|/(N K) of the initial objective.
    dedup=True ("MPR+SR-dedup"): before the refinement, candidates that differ
    from a better one in fewer than dedup_frac*N*K bits are replaced by random
    configurations (see diversify)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    grad_fn = grad_fn or obj.value_grad
    x = 0.5 + init_spread * (torch.rand(P, obj.N, obj.K, dtype=obj.rdtype, device=obj.device) - 0.5)
    y0, _ = grad_fn(x)
    scale = y0.abs().mean().clamp_min(1e-30) / (obj.N * obj.K)
    m = torch.zeros_like(x); v = torch.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    trace = []
    for t in range(1, iters + 1):
        y, gr = grad_fn(x)
        lam = lam_max * scale * (t / iters) ** 2
        gr = gr + lam * (1 - 2 * x)                        # d/dx [lam x(1-x)]
        m.mul_(b1).add_(gr, alpha=1 - b1)
        v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        x = (x - lr * mh / (vh.sqrt() + eps)).clamp_(0, 1)
        if return_trace and (t % 10 == 0 or t == 1):
            trace.append((t, y.min().item()))
    s = obj.bits_to_states(x > 0.5)
    y_s = obj.value_states(s)
    y_round = y_s.min().item()
    _sync(obj.device); t_relax = time.perf_counter() - t0
    n_dup = 0
    if dedup and refine_sweeps > 0:
        s, n_dup = diversify(obj, s, y_s, dedup_frac, seed)
    sweeps = 0
    if refine_sweeps > 0:
        if hasattr(obj, "refine"):            # FieldObjective: exact generic refinement
            s, sweeps = obj.refine(s, refine_sweeps, seed=seed)
        else:                                 # quadratic Objective: O(M) closed-form gains
            s, sweeps = successive_refinement(obj, s, refine_sweeps, seed=seed)
    return _result(obj, s, t0, "MPR", y_round=y_round, t_relax=t_relax,
                   sweeps=sweeps, trace=trace, n_dup=n_dup)


# ----------------------------------------------------------------------------
@torch.no_grad()
def cont_quantize(obj, codebook_phases, P=64, iters=300, lr=0.05, refine_sweeps=0,
                  seed=0):
    """Optimise X_j = exp(j theta_j) (ideal continuous phase shifter) with Adam,
    dy/dtheta_j = 2 Re[g_j * j X_j], then pick the codebook state with the nearest
    phase.  codebook_phases: (2^K,) phases of the states per BIT PATTERN s."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    th = 2 * math.pi * torch.rand(P, obj.N, dtype=obj.rdtype, device=obj.device)
    m = torch.zeros_like(th); v = torch.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        X = torch.polar(torch.ones_like(th), th).to(obj.cdtype)
        g = obj.g_from_field(obj.field(X))
        gr = 2 * (g * 1j * X).real
        m.mul_(b1).add_(gr, alpha=1 - b1)
        v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
        th = th - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
    diff = torch.remainder(th[..., None] - codebook_phases.to(obj.device) + math.pi,
                           2 * math.pi) - math.pi
    s = diff.abs().argmin(-1)
    name = "CQ"
    if refine_sweeps > 0:
        s, _ = successive_refinement(obj, s, refine_sweeps, seed=seed)
        name = "CQ+SR"
    return _result(obj, s, t0, name)


@torch.no_grad()
def random_sr(obj, P=64, sweeps=10, seed=0):
    t0 = time.perf_counter()
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, obj.S, (P, obj.N), generator=g).to(obj.device)
    s, n = successive_refinement(obj, s, sweeps, seed=seed)
    return _result(obj, s, t0, "SR", sweeps=n)


@torch.no_grad()
def random_search(obj, P=1024, seed=0):
    t0 = time.perf_counter()
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, obj.S, (P, obj.N), generator=g).to(obj.device)
    return _result(obj, s, t0, "Random")


# ----------------------------------------------------------------------------
@torch.no_grad()
def genetic(obj, pop=256, gens=500, tour=3, pc=0.9, pm=None, elite=2, seed=0):
    """Plain GA on the N*K bit string (tournament, uniform crossover, bit-flip
    mutation, elitism).  Fitness evaluated in batch through obj.value_states."""
    t0 = time.perf_counter()
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = obj.device
    L = obj.N * obj.K
    pm = pm if pm is not None else 1.0 / L
    bits = torch.randint(0, 2, (pop, obj.N, obj.K), generator=g).to(dev)
    fit = obj.value_states(obj.bits_to_states(bits))
    for _ in range(gens):
        order = torch.argsort(fit)
        elites = bits[order[:elite]]
        idx = torch.randint(0, pop, (2, pop, tour), generator=g).to(dev)
        win = torch.gather(idx, 2, fit[idx].argmin(-1, keepdim=True)).squeeze(-1)
        pa, pb = bits[win[0]], bits[win[1]]
        mask = torch.rand(pop, obj.N, obj.K, generator=g).to(dev) < 0.5
        do_x = (torch.rand(pop, 1, 1, generator=g).to(dev) < pc)
        child = torch.where(mask & do_x, pb, pa)
        flip = torch.rand(pop, obj.N, obj.K, generator=g).to(dev) < pm
        child = child ^ flip.long()
        child[:elite] = elites
        bits = child
        fit = obj.value_states(obj.bits_to_states(bits))
    return _result(obj, obj.bits_to_states(bits), t0, "GA")


# ----------------------------------------------------------------------------
@torch.no_grad()
def manifold_cq(obj, codebook_phases, P=64, iters=300, refine_sweeps=0, seed=0,
                armijo=0.5, c1=1e-4):
    """Manifold-optimisation baseline (MO): Riemannian conjugate gradient
    (Polak-Ribiere+, Armijo backtracking) on the complex circle manifold
    |X_j| = 1, i.e. the ideal continuous-phase model, followed by nearest-phase
    quantisation (and optional exact SR)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    th = 2 * math.pi * torch.rand(P, obj.N, dtype=obj.rdtype, device=obj.device)
    X = torch.polar(torch.ones_like(th), th).to(obj.cdtype)

    def f_and_rgrad(X):
        z = obj.field(X)
        y = obj.value_from_field(z)
        e = obj.g_from_field(z).conj()                    # Euclidean dy/dX^*  (up to 2)
        rg = e - (e * X.conj()).real * X                  # tangent projection
        return y, rg

    y, rg = f_and_rgrad(X)
    d = -rg
    step = 0.1 / (rg.abs() ** 2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)  # ~0.1 rad
    for _ in range(iters):
        slope = (rg.conj() * d).real.sum(-1, keepdim=True)
        bad = slope >= 0                                    # restart with steepest descent
        d = torch.where(bad, -rg, d)
        slope = torch.where(bad, -(rg.abs() ** 2).sum(-1, keepdim=True), slope)
        a = step * 4.0
        for _ls in range(20):                               # batched Armijo backtracking
            Xn = X + a * d
            Xn = Xn / Xn.abs().clamp_min(1e-12)
            yn = obj.value_from_field(obj.field(Xn))
            ok = (yn <= y + c1 * a.squeeze(-1) * slope.squeeze(-1))
            if ok.all():
                break
            a = torch.where(ok[:, None], a, a * armijo)
        step = a
        yn_, rgn = f_and_rgrad(Xn)
        rg_t = rg - (rg * Xn.conj()).real * Xn              # vector transport (projection)
        d_t = d - (d * Xn.conj()).real * Xn
        beta = ((rgn.conj() * (rgn - rg_t)).real.sum(-1, keepdim=True) /
                (rg.abs() ** 2).sum(-1, keepdim=True).clamp_min(1e-30)).clamp_min(0)
        d = -rgn + beta * d_t
        X, y, rg = Xn, yn_, rgn
    X_cont = X[torch.argmin(y)].clone()                 # best continuous (unquantised) solution
    y_cont = y.min().item()
    diff = torch.remainder(X.angle()[..., None].to(obj.rdtype) - codebook_phases.to(obj.device)
                           + math.pi, 2 * math.pi) - math.pi
    s = diff.abs().argmin(-1)
    name = "MO"
    if refine_sweeps > 0:
        s, _ = successive_refinement(obj, s, refine_sweeps, seed=seed)
        name = "MO+SR"
    return _result(obj, s, t0, name, X_cont=X_cont, y_cont=y_cont)


# ----------------------------------------------------------------------------
@torch.no_grad()
def single_user_optimum(b, V, d=None, chunk=512):
    """GLOBAL optimum of  max |d + sum_j b_j X_j|^2,  X_j in V_j  (single receive
    point), for arbitrary complex codebooks V (shape (S,) or (N,S)).

    Projection argument: max_X |w(X)| = max_phi sum_j max_q Re(e^{-i phi} b_j v_q)
    (+ Re(e^{-i phi} d)).  For each element the arg-max over q changes only where
    two candidates tie, phi = arg(u_q - u_r) +- pi/2, so evaluating one phi inside
    every interval between consecutive breakpoints is exhaustive.
    Returns (best power, best state indices (N,)).  Cost O(N^2 S^2) - fine on GPU
    for N <= ~4096, S <= 32.
    """
    N = b.shape[0]
    Vn = V if V.dim() == 2 else V[None, :].expand(N, -1)
    U = b[:, None] * Vn                                        # (N,S)
    S = U.shape[1]
    iu, ju = torch.triu_indices(S, S, 1, device=b.device)
    dif = U[:, iu] - U[:, ju]                                  # (N, S(S-1)/2)
    bp = torch.cat([dif.angle() + math.pi / 2, dif.angle() - math.pi / 2], -1).reshape(-1)
    bp = torch.sort(torch.remainder(bp, 2 * math.pi)).values
    mids = torch.cat([(bp[:-1] + bp[1:]) / 2, ((bp[-1] + bp[0] + 2 * math.pi) / 2)[None]])
    best_p, best_s = -1.0, None
    for i in range(0, mids.numel(), chunk):
        ph = mids[i:i + chunk]
        proj = (torch.cos(ph)[:, None, None] * U.real[None] +
                torch.sin(ph)[:, None, None] * U.imag[None])              # Re(e^{-i ph} U)
        q = proj.argmax(-1)                                               # (C,N)
        w = torch.gather(U[None].expand(len(ph), -1, -1), 2, q[..., None]).squeeze(-1).sum(-1)
        if d is not None:
            w = w + d
        p = w.abs() ** 2
        k = int(p.argmax())
        if p[k].item() > best_p:
            best_p, best_s = p[k].item(), q[k].clone()
    return best_p, best_s
