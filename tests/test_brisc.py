"""Tests for the BRISC loader / model (mpr/brisc.py) on files written in the
same layout as the Zenodo dataset (MATLAB v5 configurations, v7.3/HDF5 CSI)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mpr import brisc


def _write_fake(dirname, n_cfg=1200, frames_per=3, seed=0):
    import h5py
    from scipy.io import savemat
    bits, H, d, beta = brisc.synthetic(n_cfg=n_cfg, snr_db=60, seed=seed)
    hexs = ["!0X" + np.packbits(b).tobytes().hex().upper() for b in bits]
    savemat(os.path.join(dirname, "configurations_10000.mat"), {"configurations": np.array(hexs)})
    g = np.random.default_rng(seed)
    cid = np.repeat(np.arange(1, n_cfg + 1), frames_per)
    g.shuffle(cid)
    csi = H[cid - 1] + 1e-4 * (g.standard_normal((len(cid), brisc.N_SUB)) + 1j * g.standard_normal((len(cid), brisc.N_SUB)))
    comp = np.empty(csi.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
    comp["real"], comp["imag"] = csi.real, csi.imag
    with h5py.File(os.path.join(dirname, "antennaLog_pos1.mat"), "w") as f:
        f["csi"] = comp                                    # MATLAB [242 x frames] -> HDF5 (frames, 242)
        f["confId"] = cid.astype(np.float64)[:, None]
    return bits, H, d, beta


def test_hex_roundtrip():
    b = np.zeros(256, np.uint8); b[0] = 1; b[255] = 1
    s = "!0X" + np.packbits(b).tobytes().hex().upper()
    assert s.startswith("!0X80") and s.endswith("01")      # element 1 = MSB (Open Source RIS)
    assert (brisc.hex_to_bits(s) == b).all()


def test_load_prepare_fit():
    with tempfile.TemporaryDirectory() as t:
        bits, H, d, beta = _write_fake(t)
        cb = brisc.load_configurations(os.path.join(t, "configurations_10000.mat"))
        assert (cb == bits).all()
        assert brisc.available_positions(t, "log") == [1]
        r = brisc.load_position(t, "log", 1, cb, verbose=False)
        assert os.path.exists(brisc.cache_path(t, "log", 1))
        assert (r["count"] == 3).all()
        assert np.abs(r["H"] - H).max() < 1e-3
        d_hat, beta_hat = brisc.fit_lmb(r["H"].astype(np.complex128), cb, np.arange(len(cb)), lam=1e-8)
        assert np.abs(beta_hat - beta).max() < 2e-3 * np.abs(beta).max() + 1e-3
        dp, Bp = brisc.to_pm1(d_hat, beta_hat)
        X = 2.0 * cb[:5] - 1
        assert np.allclose(brisc.predict(d_hat, beta_hat, cb[:5]), dp[None] + X @ Bp.T)
        r2 = brisc.load_position(t, "log", 1, cb, verbose=False)   # from cache
        assert np.allclose(r2["H"], r["H"])


if __name__ == "__main__":
    test_hex_roundtrip(); test_load_prepare_fit(); print("test_brisc: OK")
