"""Tests for MPR+SR-dedup (optimize.diversify)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mpr import Objective, codebook_to_coeffs, ideal_phase_codebook, mpr, diversify, single_user_optimum


def _obj(N=64, K=2, M=1, seed=0, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    B = torch.complex(torch.randn(M, N, generator=g), torch.randn(M, N, generator=g)).to(torch.complex128)
    d = torch.full((M,), offset * N, dtype=torch.complex128)
    return Objective(-torch.ones(M, dtype=torch.float64), B, codebook_to_coeffs(ideal_phase_codebook(K)), d)


def test_diversify_replaces_only_duplicates():
    obj = _obj()
    s = torch.randint(0, obj.S, (8, obj.N))
    s[3] = s[1]; s[5] = s[1]; s[6, :] = s[2]; s[6, 0] = (s[2, 0] + 1) % obj.S   # 1 state off
    y = obj.value_states(s)
    s2, n = diversify(obj, s, y, min_frac=0.05)
    assert n == 3                                    # rows 3, 5 (exact) and 6 (near) or their twins
    assert len({tuple(r.tolist()) for r in s2}) == 8
    distinct = torch.randint(0, obj.S, (8, obj.N))
    s3, n3 = diversify(obj, distinct, obj.value_states(distinct))
    assert n3 == 0 and torch.equal(s3, distinct)     # nothing changes when candidates differ


def test_dedup_never_worse_than_its_kept_candidates():
    for seed in range(3):
        obj = _obj(N=96, seed=seed, offset=3.0)
        p_opt, _ = single_user_optimum(obj.B[0], obj.V, obj.d[0])
        r0 = mpr(obj, P=16, iters=150, refine_sweeps=10, seed=seed)
        r1 = mpr(obj, P=16, iters=150, refine_sweeps=10, seed=seed, dedup=True)
        assert r1["y"] <= -0.999 * p_opt + 1e-6 or r1["y"] <= r0["y"] + 1e-9 * abs(r0["y"])
        assert r1["n_dup"] >= 0


if __name__ == "__main__":
    test_diversify_replaces_only_duplicates(); test_dedup_never_worse_than_its_kept_candidates()
    print("test_dedup: OK")
