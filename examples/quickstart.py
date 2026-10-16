"""Minimal example: configure a 32x32 RIS with a measured 3-bit unit-cell table.

    python examples/quickstart.py            # CPU is fine (a few seconds)

Steps: load the measured look-up table, select 8 states without interpolation,
build a multi-user RIS objective, run MPR (relaxation) and MPR+SR (relaxation
followed by successive refinement), and map the optimizer states back to the
bias voltages that produce them.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from mpr import (load_lut, select_states, codebook_to_coeffs, bit_to_state_map, Objective,
                 ris_problem, ris_metrics, mpr)

dev = "cuda" if torch.cuda.is_available() else "cpu"

# 1) measured unit cell (Fara et al., Zenodo 10.5281/zenodo.7674708, CC BY 4.0)
lut = load_lut(os.path.join(ROOT, "data", "codebooks", "unit_cell_look_up_table.txt"))
info = select_states(lut, K=3)                       # 8 measured rows, max-min phase gap
print("selected bias values [V]:", info.control)

# 2) RIS problem: 8 users, 8 leakage points, 16 subcarriers, direct link
a, B, d, prob = ris_problem(Nx=32, Ny=32, n_users=8, n_leak=8, n_freq=16, seed=0)
obj = Objective(a.to(dev, torch.float32), B.to(dev, torch.complex64),
                codebook_to_coeffs(info.codebook).to(dev, torch.complex64),
                d.to(dev, torch.complex64))

# 3) optimize: relaxation only, then relaxation + exact refinement
for name, sweeps in (("MPR", 0), ("MPR+SR", 10)):
    r = mpr(obj, P=64, iters=300, refine_sweeps=sweeps, seed=0)
    s = r["s"]                                        # control-bit pattern of each element
    m = ris_metrics(obj.field(obj.X_states(s)), prob)
    print(f"{name:7s} objective {r['y']:.6g}   time {r['time']:.2f} s   "
          f"mean user {m['user_mean_dB']:.2f} dB   weakest {m['user_min_dB']:.2f} dB")

# 4) hardware control values: bit pattern -> codebook state (Gray map) -> measured row
q = bit_to_state_map(3)[s.cpu()]                     # index into info.codebook
assert torch.allclose(info.codebook[q].to(torch.complex64), obj.X_states(s).cpu())
bias = [info.control[k] for k in q.tolist()]
print("first 10 elements -> bias [V]:", bias[:10])
