"""Tests for the IEEE Access additions.  Run: python tests/test_access.py
(The LUT used here is SYNTHETIC test data, not a measurement.)"""
import itertools, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mpr import *

C = torch.complex128
torch.manual_seed(0)


def synthetic_lut_file(n=14):
    v = torch.linspace(0, 5, n)
    mag = -1 - 18 * torch.exp(-((v - 1.25) / 0.4) ** 2)          # resonance dip
    ph = torch.rad2deg(torch.remainder(torch.linspace(0.6, 6.0, n), 2 * math.pi) - math.pi)
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write("voltage[V] ; Magnitude[dB] ; Phase[degrees]\n")
    for a, b, c in zip(v, mag, ph):
        f.write(f"{a:.2f} ; {b:.3f} ; {c:.3f}\n")
    f.close()
    return f.name, v, mag, ph


def test_lut_parser_and_selection():
    path, v, mag, ph = synthetic_lut_file()
    lut = load_lut(path)
    assert len(lut) == 14
    assert torch.allclose(lut.amp, 10 ** (mag.double() / 20), atol=1e-4)
    info = select_states(lut, 3)
    assert info.codebook.shape == (8,) and len(set(info.hw_index)) == 8
    ang = torch.remainder(info.codebook.angle() + math.pi, 2 * math.pi)
    assert (torch.diff(ang) > 0).all(), "states must be phase ordered"
    best = max(min_circular_gap(lut.phase[list(s)]) for s in itertools.combinations(range(14), 8))
    assert abs(min_circular_gap(lut.phase[info.hw_index]) - best) < 1e-9
    info2 = select_states(lut, 3, amp_floor_dB=-10)
    assert all(20 * math.log10(lut.amp[i]) >= -10 for i in info2.hw_index)
    os.unlink(path)


def test_state_table_template():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "data", "codebooks", "TEMPLATE_16state.csv")
    fr = table_frequencies(p)
    lut = load_state_table(p, fr[0])
    assert len(lut) == 16
    info = select_states(lut, 4, rule="all")
    assert info.codebook.shape == (16,)


def _ref_value(obj, x):
    x = x.clone().requires_grad_(True)
    mono = obj._monomials(x)
    pows = []
    for B, d, c in zip(obj.B, obj.d, obj.c):
        X = mono @ c if c.dim() == 1 else (mono * c).sum(-1)
        z = X @ B.T
        z = z if d is None else z + d
        pows.append(z.real ** 2 + z.imag ** 2)
    y = obj.F.value(pows)
    y.sum().backward()
    return y.detach(), x.grad


def _comp(M, N, K, per_elem=False):
    B = torch.randn(M, N, dtype=C)
    cb = torch.randn(N, 1 << K, dtype=C) if per_elem else torch.randn(1 << K, dtype=C)
    return B, moebius(cb), torch.randn(M, dtype=C)


def test_field_quadratic_matches_objective():
    B, c, d = _comp(9, 6, 3)
    a = torch.randn(9, dtype=torch.float64)
    o1 = Objective(a, B, c, d)
    o2 = FieldObjective([B], [c], Quadratic([a]), [d])
    x = torch.rand(4, 6, 3, dtype=torch.float64)
    y1, g1 = o1.value_grad(x); y2, g2 = o2.value_grad(x)
    assert torch.allclose(y1, y2) and torch.allclose(g1, g2)


def test_field_multicomponent_and_softmin_gradients():
    K, N = 3, 7
    (B1, c1, d1), (B2, c2, d2) = _comp(10, N, K), _comp(10, N, K, per_elem=True)
    a = [torch.randn(10, dtype=torch.float64), torch.randn(10, dtype=torch.float64)]
    oq = FieldObjective([B1, B2], [c1, c2], Quadratic(a), [d1, d2])
    u = [torch.tensor([0, 1, 2, 0, 1, 2, -1, -1, 0, 1]), torch.tensor([2, 2, 1, 0, -1, 0, 1, -1, 2, 0])]
    os_ = FieldObjective([B1, B2], [c1, c2], SoftMinPower(u, 3, tau=0.3, scale=20.0, leak_weight=0.5), [d1, d2])
    x = torch.rand(3, N, K, dtype=torch.float64)
    for o in (oq, os_):
        y, g = o.value_grad(x)
        y0, g0 = _ref_value(o, x)
        assert torch.allclose(y, y0) and torch.allclose(g, g0, atol=1e-9)


def test_generic_refine_monotone_and_local_opt():
    K, N = 2, 6
    B, c, d = _comp(8, N, K)
    u = [torch.tensor([0, 0, 1, 1, 2, 2, -1, -1])]
    o = FieldObjective([B], [c], SoftMinPower(u, 3, tau=0.2, scale=10.0), [d])
    s0 = torch.randint(0, 4, (5, N))
    s, _ = o.refine(s0, 50)
    y = o.value_states(s)
    assert (y <= o.value_states(s0) + 1e-12).all()
    for j in range(N):
        for q in range(4):
            s2 = s.clone(); s2[:, j] = q
            assert (o.value_states(s2) >= y - 1e-9).all()


def test_single_user_optimum_bruteforce():
    for trial in range(5):
        N, S = 5, 4
        b = torch.randn(N, dtype=C); V = torch.randn(S, dtype=C); d = torch.randn((), dtype=C)
        allS = torch.tensor(list(itertools.product(range(S), repeat=N)))
        p_bf = ((V[allS] * b).sum(-1) + d).abs().pow(2).max().item()
        p, s = single_user_optimum(b, V, d)
        assert abs(p - p_bf) < 1e-9 * max(1, p_bf), (p, p_bf)
        assert abs(((V[s] * b).sum() + d).abs().item() ** 2 - p) < 1e-9 * max(1, p)
        Vn = torch.randn(N, S, dtype=C)                     # element-dependent codebooks
        p_bf = ((Vn[torch.arange(N), allS] * b).sum(-1) + d).abs().pow(2).max().item()
        assert abs(single_user_optimum(b, Vn, d)[0] - p_bf) < 1e-9 * max(1, p_bf)


def test_manifold_baseline_runs():
    a, B, info = array_problem(8, 8, M=100, main_dir=(0.2, 0.1), side_weight=30)
    obj = Objective(a, B, codebook_to_coeffs(ideal_phase_codebook(3)))
    r = manifold_cq(obj, obj.V.angle(), P=4, iters=60)
    rr = random_search(obj, P=256)
    assert r["y"] < rr["y"]
    r2 = manifold_cq(obj, obj.V.angle(), P=4, iters=60, refine_sweeps=3)
    assert r2["y"] <= r["y"] + 1e-9


def test_per_element_variation_shapes():
    cb = per_element_variation(ideal_phase_codebook(2), 10, 10.0, 0.1)
    assert cb.shape == (10, 4)
    assert torch.allclose(per_element_variation(ideal_phase_codebook(2), 3, 0, 0),
                          ideal_phase_codebook(2).expand(3, -1))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
