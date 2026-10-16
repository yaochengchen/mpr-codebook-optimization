"""
BRISC measured RIS channels (Piana et al., arXiv:2602.21102; Zenodo
doi:10.5281/zenodo.18714621, CC BY 4.0).

Dataset: CSI at 5.53 GHz (242 OFDM subcarriers) between a 1-antenna TX at nine
positions and a fixed RX (log-periodic or dipole antenna), reflected by a
256-element (16x16) binary RIS.  ~10 000 RIS configurations per TX position:
16 + 512 block configurations (2x2 and 3x3 macro-blocks) followed by
element-wise random ones.  Several Wi-Fi frames per configuration.

Files (one per TX position and RX antenna, MATLAB v7.3 = HDF5):
    antennaLog_pos<k>.mat / antennaDipole_pos<k>.mat
        csi    : [242 x frame_N] complex
        confId : [1 x frame_N]  configuration index 1..10000
        (confStr, sqn, coarsets are not needed here)
    configurations_10000.mat  (MATLAB v5): the 10 000 configurations as
        hexadecimal strings '!0X<64 hex digits>' (Open Source RIS command
        format: bit 255 = MSB = element 1, elements in reading order).

Model.  As in the BRISC paper (linear model with bias, LMB), the channel of
subcarrier f is affine in the element states x_j in {0,1}:
    h_f(x) = d_f + sum_j beta_fj x_j .
With X_j = 2 x_j - 1 in {+1,-1} this is exactly the paper's field model
    z_f = d'_f + sum_j b_fj X_j,   b_fj = beta_fj / 2,  d'_f = d_f + sum_j beta_fj / 2,
i.e. a 1-bit codebook {+1, -1}; every row f is one subcarrier.

Usage
    python -m mpr.brisc prepare --data data/brisc       # .mat -> cache/*.npz (once)
"""
import argparse
import glob
import os
import re

import numpy as np

N_ELEM = 256
N_SUB = 242
N_BLOCK_CFG = 16 + 512          # 2x2 and 3x3 macro-block configurations come first


# ----------------------------------------------------------------------------- configurations
def hex_to_bits(s, n=N_ELEM):
    """'!0X...' -> uint8 array (n,), entry e = state of element e+1 (reading order)."""
    s = str(s).strip()
    s = s.lstrip("!")
    if s[:2] in ("0x", "0X"):
        s = s[2:]
    s = re.sub(r"[^0-9a-fA-F]", "", s)
    if len(s) * 4 < n:
        raise ValueError(f"configuration string too short: {s!r}")
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(s[: (n + 7) // 8 * 2]), dtype=np.uint8))
    return bits[:n].astype(np.uint8)


def _strings_from_mat_value(v):
    """Flatten whatever scipy.io.loadmat returns for a char matrix / cell array of strings."""
    v = np.asarray(v)
    if v.dtype.kind in "US":
        if v.ndim == 2 and v.shape[1] > 1 and all(len(str(x)) <= 1 for x in v[0]):
            return ["".join(r) for r in v]            # char matrix kept as single chars
        return [str(x) for x in v.ravel()]
    if v.dtype == object:
        out = []
        for x in v.ravel():
            x = np.asarray(x)
            out.append(str(x.ravel()[0]) if x.dtype.kind in "US" and x.size == 1 else
                       "".join(map(str, x.ravel())))
        return out
    raise TypeError(f"cannot read configuration strings from dtype {v.dtype}")


def load_configurations(path):
    """configurations_10000.mat (v5 or v7.3) or a text file with one hex string per line
    -> uint8 (n_cfg, 256)."""
    if path.endswith(".txt"):
        strs = [l for l in open(path).read().split() if l.strip()]
    else:
        try:
            from scipy.io import loadmat
            m = loadmat(path)
            keys = [k for k in m if not k.startswith("__")]
            strs = None
            for k in sorted(keys, key=lambda k: -np.asarray(m[k]).size):
                try:
                    strs = _strings_from_mat_value(m[k])
                    break
                except TypeError:
                    continue
            if strs is None:
                raise TypeError(f"no string variable in {path}: {keys}")
        except NotImplementedError:                    # v7.3 -> HDF5
            import h5py
            with h5py.File(path, "r") as f:
                k = [k for k in f.keys() if not k.startswith("#")][0]
                ds = f[k]
                if ds.dtype == object:                 # cell of refs
                    strs = ["".join(chr(c) for c in np.asarray(f[r]).ravel()) for r in ds[()].ravel()]
                else:                                  # uint16 char matrix, stored transposed
                    a = np.asarray(ds[()])
                    a = a.T if a.shape[0] < a.shape[1] else a
                    strs = ["".join(chr(c) for c in row) for row in a]
    return np.stack([hex_to_bits(s) for s in strs])


# ----------------------------------------------------------------------------- CSI
def prepare(mat_path, n_cfg, out_npz=None, chunk=100_000, verbose=True):
    """Average the CSI of all frames that share a configuration.

    Returns dict(H=(n_cfg, 242) complex64 mean CSI, count=(n_cfg,) frames per
    configuration, frames=total frames) and optionally saves it to out_npz."""
    import h5py
    with h5py.File(mat_path, "r") as f:
        ds = f["csi"]
        cid = np.asarray(f["confId"][()]).ravel().astype(np.int64) - 1        # MATLAB 1-based
        shp = ds.shape
        frame_axis = 0 if shp[1] == N_SUB or shp[0] != N_SUB else 1
        n_frames = shp[frame_axis]
        if len(cid) != n_frames:
            raise ValueError(f"{mat_path}: {len(cid)} confIds for {n_frames} frames")
        sums = np.zeros((n_cfg, N_SUB), np.complex128)
        cnt = np.zeros(n_cfg, np.int64)
        for i0 in range(0, n_frames, chunk):
            i1 = min(i0 + chunk, n_frames)
            raw = ds[i0:i1] if frame_axis == 0 else ds[:, i0:i1].T
            if raw.dtype.names:                                     # compound (real, imag)
                c = raw["real"].astype(np.float64) + 1j * raw["imag"].astype(np.float64)
            else:
                c = raw.astype(np.complex128)
            ids = cid[i0:i1]
            ok = (ids >= 0) & (ids < n_cfg) & np.isfinite(c).all(1)
            c, ids = c[ok], ids[ok]
            order = np.argsort(ids, kind="stable")
            ids, c = ids[order], c[order]
            u, start = np.unique(ids, return_index=True)
            sums[u] += np.add.reduceat(c, start, axis=0)
            cnt[u] += np.diff(np.r_[start, len(ids)])
            if verbose:
                print(f"  {os.path.basename(mat_path)}: {i1}/{n_frames} frames", flush=True)
    H = np.where(cnt[:, None] > 0, sums / np.maximum(cnt, 1)[:, None], 0).astype(np.complex64)
    out = dict(H=H, count=cnt, frames=n_frames)
    if out_npz:
        os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
        np.savez_compressed(out_npz, **out)
    return out


def cache_path(data_dir, antenna, pos):
    return os.path.join(data_dir, "cache", f"antenna{antenna.capitalize()}_pos{pos}.npz")


def mat_path(data_dir, antenna, pos):
    return os.path.join(data_dir, f"antenna{antenna.capitalize()}_pos{pos}.mat")


def load_position(data_dir, antenna, pos, bits, verbose=True):
    """Prepared (cached) per-configuration mean CSI for one TX position."""
    cp = cache_path(data_dir, antenna, pos)
    if os.path.exists(cp):
        z = np.load(cp)
        return dict(H=z["H"], count=z["count"], frames=int(z["frames"]))
    mp = mat_path(data_dir, antenna, pos)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"neither {cp} nor {mp} exists (see data/brisc/README.md)")
    if verbose:
        print(f"preparing {mp} -> {cp}", flush=True)
    return prepare(mp, len(bits), cp, verbose=verbose)


def available_positions(data_dir, antenna):
    pos = set()
    for pat in (f"antenna{antenna.capitalize()}_pos*.mat", f"cache/antenna{antenna.capitalize()}_pos*.npz"):
        for p in glob.glob(os.path.join(data_dir, pat)):
            m = re.search(r"_pos(\d+)\.", p)
            if m:
                pos.add(int(m.group(1)))
    return sorted(pos)


# ----------------------------------------------------------------------------- model
def fit_lmb(H, bits, idx, lam=1e-4):
    """Ridge fit of h_f(x) = d_f + sum_j beta_fj x_j on configurations idx.

    H: (n_cfg, M) complex, bits: (n_cfg, N) {0,1}.  lam is relative to the mean
    diagonal of the Gram matrix; the bias is not regularised.
    Returns d (M,), beta (M, N)."""
    X = np.c_[bits[idx].astype(np.float64), np.ones(len(idx))]
    G = X.T @ X
    reg = lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(G + reg, X.T @ H[idx].astype(np.complex128))       # (N+1, M)
    return W[-1], W[:-1].T


def predict(d, beta, bits):
    return d[None, :] + bits.astype(np.float64) @ beta.T


def to_pm1(d, beta):
    """x in {0,1} -> X = 2x-1 in {+1,-1}:  h = d' + B X."""
    return d + beta.sum(1) / 2, beta / 2


def nmse_db(Hhat, H):
    return 10 * np.log10(np.sum(np.abs(Hhat - H) ** 2) / np.sum(np.abs(H) ** 2))


def wideband_power(H):
    return np.mean(np.abs(H) ** 2, axis=-1)


# ----------------------------------------------------------------------------- synthetic
def synthetic(n_cfg=3000, n_elem=N_ELEM, n_sub=N_SUB, snr_db=20.0, seed=0):
    """Fake dataset with the BRISC structure (for smoke tests without the download).
    Returns bits (n_cfg, N), H (n_cfg, M) and the true (d, beta)."""
    g = np.random.default_rng(seed)
    f = np.linspace(-0.5, 0.5, n_sub)
    tau = g.uniform(0, 3, n_elem)                              # element delays (samples)
    amp = 0.05 * (1 + 0.3 * g.standard_normal(n_elem))
    ph0 = g.uniform(0, 2 * np.pi, n_elem)
    beta = (amp * np.exp(1j * ph0))[None, :] * np.exp(-2j * np.pi * f[:, None] * tau[None, :])
    d = 0.6 * np.exp(-2j * np.pi * f * 1.5) + 0.3 * np.exp(-2j * np.pi * f * 7.0)
    bits = np.r_[np.repeat(g.integers(0, 2, (N_BLOCK_CFG, 1)), n_elem, 1),
                 g.integers(0, 2, (n_cfg - N_BLOCK_CFG, n_elem))].astype(np.uint8)
    H = predict(d, beta, bits)
    noise = np.sqrt(np.mean(np.abs(H) ** 2) * 10 ** (-snr_db / 10) / 2)
    H = H + noise * (g.standard_normal(H.shape) + 1j * g.standard_normal(H.shape))
    return bits, H.astype(np.complex64), d, beta


# ----------------------------------------------------------------------------- CLI
def _main():
    ap = argparse.ArgumentParser(description="BRISC preprocessing: .mat -> cache/*.npz")
    ap.add_argument("cmd", choices=["prepare"])
    ap.add_argument("--data", default="data/brisc")
    ap.add_argument("--antenna", default="log,dipole")
    a = ap.parse_args()
    cfg = os.path.join(a.data, "configurations_10000.mat")
    bits = load_configurations(cfg)
    print(f"{cfg}: {bits.shape[0]} configurations x {bits.shape[1]} elements")
    np.save(os.path.join(a.data, "configurations_bits.npy"), bits)
    for ant in a.antenna.split(","):
        for pos in available_positions(a.data, ant):
            if os.path.exists(cache_path(a.data, ant, pos)):
                print("cached:", cache_path(a.data, ant, pos)); continue
            r = load_position(a.data, ant, pos, bits)
            used = (r["count"] > 0).sum()
            print(f"{ant} pos{pos}: {r['frames']} frames, {used} configurations, "
                  f"median {np.median(r['count'][r['count'] > 0]):.0f} frames/config", flush=True)


if __name__ == "__main__":
    _main()
