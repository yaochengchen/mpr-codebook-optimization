"""
Problem builders.  Both scenarios map onto  y = sum_i a_i |d_i + sum_j b_ij X_j|^2
(minimised): negative a_i reward power, positive a_i penalise it.

1) Antenna array beam synthesis with K-bit phase shifters (ULA / UPA).
   b_ij = exp(j 2 pi (x_j u_i + y_j v_i)),  a_i < 0 in the main-beam region,
   a_i > 0 in the sidelobe region.

2) RIS-assisted multi-user link with K-bit unit cells.
   b_ij = h_ij g_j (cascaded BS->RIS->user channel, optional several subcarriers),
   d_i  = direct BS->user channel, a_i < 0 for served users, a_i > 0 for
   leakage (eavesdropper / interference) points.

3) Random instance matching the abstract spec (M<=500, N<=2000).
"""
import math
import torch


def upa_positions(Nx, Ny, spacing=0.5, dtype=torch.float64, device="cpu"):
    ix = torch.arange(Nx, dtype=dtype, device=device) - (Nx - 1) / 2
    iy = torch.arange(Ny, dtype=dtype, device=device) - (Ny - 1) / 2
    X, Y = torch.meshgrid(ix, iy, indexing="ij")
    return spacing * X.reshape(-1), spacing * Y.reshape(-1)    # in wavelengths


def steering(px, py, u, v, freq_ratio=1.0):
    """(len(u), N) matrix exp(j 2 pi f/fc (x u + y v))."""
    ph = 2 * math.pi * freq_ratio * (u[:, None] * px[None, :] + v[:, None] * py[None, :])
    return torch.polar(torch.ones_like(ph), ph)


# ----------------------------------------------------------------------------
def array_problem(Nx=45, Ny=45, M=500, main_dir=(0.3, 0.2), main_frac=0.1,
                  null_factor=1.5, side_weight=1.0, spacing=0.5, seed=0,
                  cdtype=torch.complex128, device="cpu"):
    """Returns (a, B, info).  Main-beam samples fill a disc of radius ~ half the
    3-dB beamwidth around main_dir; sidelobe samples are drawn uniformly in the
    visible region outside null_factor x first-null radius."""
    g = torch.Generator().manual_seed(seed)
    rdtype = torch.float64
    px, py = upa_positions(Nx, Ny, spacing, rdtype)
    Lx = max(Nx, 1) * spacing
    Ly = max(Ny, 1) * spacing
    r_null = 1.0 / max(Lx, Ly if Ny > 1 else Lx)          # first-null radius in u
    r_main = 0.5 * 0.886 / max(Lx, Ly if Ny > 1 else Lx)  # half 3-dB width
    u0, v0 = main_dir
    Mm = max(1, int(M * main_frac))
    Ms = M - Mm

    def sample_disc(n, r, cu, cv):
        rho = r * torch.sqrt(torch.rand(n, generator=g, dtype=rdtype))
        th = 2 * math.pi * torch.rand(n, generator=g, dtype=rdtype)
        return cu + rho * torch.cos(th), cv + rho * torch.sin(th)

    if Ny == 1:   # 1-D: uniform grid, v = 0
        um = u0 + torch.linspace(-r_main, r_main, Mm, dtype=rdtype)
        vm = torch.zeros_like(um)
        cand = torch.linspace(-1, 1, 20 * Ms, dtype=rdtype)
        cand = cand[(cand - u0).abs() > null_factor * r_null]
        us = cand[torch.linspace(0, len(cand) - 1, Ms).long()]
        vs = torch.zeros_like(us)
    else:
        um, vm = sample_disc(Mm, r_main, u0, v0)
        us, vs = [], []
        while sum(len(t) for t in us) < Ms:
            uu, vv = sample_disc(4 * Ms, 1.0, 0.0, 0.0)
            keep = ((uu - u0) ** 2 + (vv - v0) ** 2).sqrt() > null_factor * r_null
            us.append(uu[keep]); vs.append(vv[keep])
        us = torch.cat(us)[:Ms]; vs = torch.cat(vs)[:Ms]

    u = torch.cat([um, us]); v = torch.cat([vm, vs])
    B = steering(px, py, u, v).to(cdtype)
    a = torch.cat([-torch.ones(Mm, dtype=rdtype) / Mm,
                   side_weight * torch.ones(Ms, dtype=rdtype) / Ms])
    info = dict(kind="array", px=px, py=py, u=u, v=v, main=torch.arange(Mm),
                u0=u0, v0=v0, r_null=r_null, null_factor=null_factor, Nx=Nx, Ny=Ny)
    return a.to(device), B.to(device), info


def array_metrics(X, info, grid=512):
    """X: (N,) complex weights.  Returns gain toward main_dir (normalised to the
    ideal N^2 of a perfectly phased unit-amplitude array, dB) and peak sidelobe
    level (dB, relative to main-beam peak) on a dense grid."""
    px, py = info["px"].to(X.device), info["py"].to(X.device)
    N = px.numel()
    u0, v0 = info["u0"], info["v0"]
    rdt = px.dtype
    if info["Ny"] == 1:
        u = torch.linspace(-1, 1, grid * 8, dtype=rdt, device=X.device)
        v = torch.zeros_like(u)
    else:
        t = torch.linspace(-1, 1, grid, dtype=rdt, device=X.device)
        U, Vv = torch.meshgrid(t, t, indexing="ij")
        vis = U ** 2 + Vv ** 2 <= 1
        u, v = U[vis], Vv[vis]
    P = torch.cat([(steering(px, py, u[i:i + 20000], v[i:i + 20000]).to(X.dtype) @ X).abs() ** 2
                   for i in range(0, len(u), 20000)])
    dist = ((u - u0) ** 2 + (v - v0) ** 2).sqrt()
    main = dist <= info["null_factor"] * info["r_null"]
    g0 = (steering(px, py, torch.tensor([u0], dtype=rdt, device=X.device),
                   torch.tensor([v0], dtype=rdt, device=X.device)).to(X.dtype) @ X).abs() ** 2
    peak_main = P[main].max()
    psl = P[~main].max() / peak_main
    return dict(gain_dB=10 * torch.log10(g0 / N ** 2).item(),
                psl_dB=10 * torch.log10(psl).item())


# ----------------------------------------------------------------------------
def ris_problem(Nx=32, Ny=32, n_users=8, n_leak=4, n_freq=1, bw_ratio=0.1,
                kappa=10.0, direct=True, direct_gain=0.1, leak_weight=1.0,
                spacing=0.5, seed=0, cdtype=torch.complex128, device="cpu",
                freq_ratios=None):
    """Weighted sum received power at users minus leakage power, over n_freq
    subcarriers spanning a fractional bandwidth bw_ratio.  M = (n_users+n_leak)*n_freq.
    Path-loss magnitudes are normalised to 1 per element; direct_gain sets the
    direct-link strength relative to one RIS element."""
    g = torch.Generator().manual_seed(seed)
    rdt = torch.float64
    px, py = upa_positions(Nx, Ny, spacing, rdt)
    N = px.numel()
    n_pts = n_users + n_leak

    def rand_dir(n):
        th = math.radians(60) * torch.sqrt(torch.rand(n, generator=g, dtype=rdt))
        ph = 2 * math.pi * torch.rand(n, generator=g, dtype=rdt)
        return torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph)

    ub, vb = rand_dir(1)
    uu, vu = rand_dir(n_pts)
    if freq_ratios is not None:              # explicit f/fc list (e.g. two measured bands)
        freqs = torch.tensor(list(freq_ratios), dtype=rdt)
    else:
        freqs = torch.linspace(1 - bw_ratio / 2, 1 + bw_ratio / 2, n_freq, dtype=rdt) \
            if n_freq > 1 else torch.ones(1, dtype=rdt)

    def cn(*shape):
        return torch.complex(torch.randn(*shape, generator=g, dtype=rdt),
                             torch.randn(*shape, generator=g, dtype=rdt)) / math.sqrt(2)

    nlos = cn(n_pts, N)          # frequency-flat NLoS part for simplicity
    Bs, ds = [], []
    for f in freqs:
        gvec = steering(px, py, ub, vb, f)[0]                        # BS -> RIS (LoS)
        hlos = steering(px, py, uu, vu, f)                           # RIS -> points
        h = math.sqrt(kappa / (kappa + 1)) * hlos + math.sqrt(1 / (kappa + 1)) * nlos
        Bs.append(h * gvec[None, :])
        ds.append(direct_gain * math.sqrt(N) * cn(n_pts) if direct else torch.zeros(n_pts, dtype=torch.complex128))
    B = torch.cat(Bs).to(cdtype)
    d = torch.cat(ds).to(cdtype) if direct else None
    w = torch.cat([-torch.ones(n_users, dtype=rdt) / n_users,
                   leak_weight * torch.ones(n_leak, dtype=rdt) / max(n_leak, 1)])
    a = w.repeat(len(freqs)) / len(freqs)
    user_mask = torch.cat([torch.ones(n_users, dtype=torch.bool),
                           torch.zeros(n_leak, dtype=torch.bool)]).repeat(len(freqs))
    user_idx = torch.cat([torch.arange(n_users), -torch.ones(n_leak, dtype=torch.long)]).repeat(len(freqs))
    info = dict(kind="ris", user_mask=user_mask, N=N, user_idx=user_idx,
                n_users=n_users, n_freq=len(freqs))
    return a.to(device), B.to(device), (d.to(device) if d is not None else None), info


def ris_metrics(z, info):
    """z: (M,) received fields.  Powers normalised by N^2 (ideal coherent gain)."""
    p = (z.abs() ** 2) / info["N"] ** 2
    um = info["user_mask"].to(z.device)
    out = dict(user_mean_dB=10 * torch.log10(p[um].mean()).item(),
               user_min_dB=10 * torch.log10(p[um].min()).item())
    if (~um).any():
        out["leak_mean_dB"] = 10 * torch.log10(p[~um].mean()).item()
    return out


# ----------------------------------------------------------------------------
def random_problem(M=500, N=2000, seed=0, cdtype=torch.complex128, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(M, generator=g, dtype=torch.float64)
    B = torch.complex(torch.randn(M, N, generator=g, dtype=torch.float64),
                      torch.randn(M, N, generator=g, dtype=torch.float64)).to(cdtype)
    return a.to(device), B.to(device)
