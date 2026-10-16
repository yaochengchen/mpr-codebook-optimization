"""Correctness checks.  Run:  python -m pytest -q tests   (or python tests/test_core.py)"""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mpr import *

torch.manual_seed(0)
C = torch.complex128


def small_obj(K=3, N=7, M=11, shared=True, direct=True):
    a = torch.randn(M, dtype=torch.float64)
    B = torch.randn(M, N, dtype=C)
    cb = torch.randn(1 << K, dtype=C) if shared else torch.randn(N, 1 << K, dtype=C)
    d = torch.randn(M, dtype=C) if direct else None
    return Objective(a, B, moebius(cb), d), cb


def test_moebius_exact():
    for K in range(1, 6):
        for mapping in ("binary", "gray"):
            v = torch.randn(1 << K, dtype=C)
            c = codebook_to_coeffs(v, mapping)
            m = bit_to_state_map(K, mapping)
            for s in range(1 << K):
                bits = [(s >> k) & 1 for k in range(K)]
                val = sum(c[S] * all(bits[k] for k in range(K) if (S >> k) & 1)
                          for S in range(1 << K))
                assert torch.allclose(val, v[m[s]])
            assert torch.allclose(binary_to_doc(doc_to_binary(c)), c)


def test_doc_order_matches_note():
    # K=3 note: c0, c1 x1, c2 x2, c3 x3, c4 x1x2, c5 x1x3, c6 x2x3, c7 x1x2x3
    from mpr.encoding import doc_order
    assert doc_order(3) == [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111]


def test_gradient_vs_autograd():
    for shared in (True, False):
        for K in (1, 2, 3, 5):
            obj, _ = small_obj(K=K, shared=shared)
            x = torch.rand(4, obj.N, K, dtype=torch.float64)
            y1, g1 = obj.value_grad(x)
            y0, g0 = obj.value_grad_autograd(x)
            assert torch.allclose(y1, y0) and torch.allclose(g1, g0), (shared, K)


def test_relaxed_equals_states_at_vertices():
    obj, _ = small_obj(K=4)
    s = torch.randint(0, 16, (5, obj.N))
    y1, _ = obj.value_grad(obj.states_to_bits(s).double())
    assert torch.allclose(y1, obj.value_states(s))


def test_flip_gains_exact():
    for shared in (True, False):
        obj, _ = small_obj(K=3, shared=shared)
        s = torch.randint(0, 8, (3, obj.N))
        G = obj.flip_gains(s)
        y = obj.value_states(s)
        for j in range(obj.N):
            for k in range(3):
                s2 = s.clone(); s2[:, j] ^= (1 << k)
                assert torch.allclose(obj.value_states(s2) - y, G[:, j, k])


def test_sr_monotone_and_local_opt():
    obj, _ = small_obj(K=2, N=6, M=9)
    s0 = torch.randint(0, 4, (8, obj.N))
    s, _ = successive_refinement(obj, s0, 50)
    assert (obj.value_states(s) <= obj.value_states(s0) + 1e-12).all()
    # 1-element optimality: no single-element state change improves
    y = obj.value_states(s)
    for j in range(obj.N):
        for q in range(4):
            s2 = s.clone(); s2[:, j] = q
            assert (obj.value_states(s2) >= y - 1e-9).all()


def test_mpr_finds_global_opt_tiny():
    # brute force on 4 elements x 2 bits = 256 configurations, 20 random
    # instances (mixed-sign a: hard, non-convex).  Heuristic -> require >= 18/20.
    allS = torch.tensor(list(itertools.product(range(4), repeat=4)))
    hits = 0
    for seed in range(20):
        torch.manual_seed(seed)
        obj, _ = small_obj(K=2, N=4, M=6, direct=False)
        ybest = obj.value_states(allS).min().item()
        r = mpr(obj, P=16, iters=200)
        hits += abs(r["y"] - ybest) <= 1e-9 * max(1, abs(ybest))
    assert hits >= 18, hits


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
