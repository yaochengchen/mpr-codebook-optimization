"""
Hardware codebooks: measured look-up tables, state selection, per-element variation.

Conventions
-----------
* A codebook is a complex tensor of length 2^K indexed by *state* q = 0..2^K-1.
* States are ordered by phase (ascending, wrapped to [-pi, pi)), so that with the
  Gray bit-to-state mapping used by `codebook_to_coeffs(..., "gray")` neighbouring
  phases differ in exactly one control bit.  This ordering is purely internal to
  the optimiser: `CodebookInfo.hw_index[q]` gives the hardware row (bias voltage,
  digital code, ...) to apply for optimiser state q.
"""
from dataclasses import dataclass, field
from itertools import combinations
import csv
import math
import re
import torch


@dataclass
class MeasuredLUT:
    """One measured table: control value, |Gamma| (linear), phase (rad)."""
    control: list            # e.g. bias voltage [V] or digital code
    amp: torch.Tensor        # linear magnitude
    phase: torch.Tensor      # radians
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def response(self) -> torch.Tensor:
        return torch.polar(self.amp, self.phase).to(torch.complex128)

    def __len__(self):
        return len(self.control)


@dataclass
class CodebookInfo:
    codebook: torch.Tensor   # (2^K,) complex, phase-ordered
    hw_index: list           # hardware row of each optimiser state
    control: list            # control value of each optimiser state
    K: int
    note: str = ""


# ----------------------------------------------------------------------------
_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def load_lut(path, mag_unit="dB", phase_unit="deg", columns=(0, 1, 2)) -> MeasuredLUT:
    """Parse a text/CSV table with rows  control ; magnitude ; phase.

    Separators ';', ',', tab or spaces are accepted; header/comment lines
    (no three numbers) are skipped.  `columns` selects (control, mag, phase).
    mag_unit: 'dB' (20 log10 |Gamma|) or 'lin'.  phase_unit: 'deg' or 'rad'.
    """
    ctrl, mag, ph = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip().startswith(("#", "%", "//")):
                continue
            nums = _num.findall(line)
            if len(nums) < max(columns) + 1:
                continue
            c, m, p = (float(nums[i]) for i in columns)
            ctrl.append(c); mag.append(m); ph.append(p)
    if not ctrl:
        raise ValueError(f"no numeric rows found in {path}")
    mag = torch.tensor(mag, dtype=torch.float64)
    ph = torch.tensor(ph, dtype=torch.float64)
    amp = 10 ** (mag / 20) if mag_unit.lower() == "db" else mag
    phase = torch.deg2rad(ph) if phase_unit.lower().startswith("deg") else ph
    return MeasuredLUT(ctrl, amp, phase, source=str(path),
                       meta=dict(mag_unit=mag_unit, phase_unit=phase_unit))


def load_state_table(path, freq=None) -> MeasuredLUT:
    """CSV with header  code,freq_GHz,amp_dB,phase_deg  (one row per state and
    frequency), as in data/codebooks/TEMPLATE_16state.csv.  If `freq` is given,
    rows with that frequency are returned."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("code", "").strip():
                continue
            if freq is not None and abs(float(r["freq_GHz"]) - freq) > 1e-6:
                continue
            rows.append(r)
    if not rows:
        raise ValueError(f"no rows for freq={freq} in {path}")
    rows.sort(key=lambda r: int(r["code"]))
    amp = 10 ** (torch.tensor([float(r["amp_dB"]) for r in rows], dtype=torch.float64) / 20)
    ph = torch.deg2rad(torch.tensor([float(r["phase_deg"]) for r in rows], dtype=torch.float64))
    return MeasuredLUT([int(r["code"]) for r in rows], amp, ph, source=str(path),
                       meta=dict(freq_GHz=freq))


def table_frequencies(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return sorted({float(r["freq_GHz"]) for r in csv.DictReader(f) if r.get("code", "").strip()})


# ----------------------------------------------------------------------------
def _wrap(p):
    return torch.remainder(p + math.pi, 2 * math.pi) - math.pi


def min_circular_gap(phases) -> float:
    p = torch.sort(_wrap(torch.as_tensor(phases, dtype=torch.float64))).values
    gaps = torch.diff(torch.cat([p, p[:1] + 2 * math.pi]))
    return gaps.min().item()


def select_states(lut: MeasuredLUT, K: int, rule="max_min_gap", amp_floor_dB=None,
                  indices=None) -> CodebookInfo:
    """Choose 2^K of the measured rows as a K-bit codebook (no interpolation).

    rule='max_min_gap' : exhaustive search over subsets maximising the smallest
                         circular phase gap (best phase coverage); ties broken by
                         the larger mean amplitude.
    rule='all'         : use all rows (requires len(lut) == 2^K).
    indices            : explicit list of row indices (overrides rule).
    amp_floor_dB       : discard rows with |Gamma| below this level first.
    """
    n = 1 << K
    cand = list(range(len(lut)))
    if amp_floor_dB is not None:
        cand = [i for i in cand if 20 * math.log10(lut.amp[i].item()) >= amp_floor_dB]
    if indices is not None:
        chosen = list(indices)
    elif rule == "all":
        if len(cand) != n:
            raise ValueError(f"table has {len(cand)} usable rows, need {n}")
        chosen = cand
    elif rule == "max_min_gap":
        if len(cand) < n:
            raise ValueError(f"only {len(cand)} usable rows for {n} states")
        best, best_key = None, None
        for sub in combinations(cand, n):
            key = (round(min_circular_gap(lut.phase[list(sub)]), 9),
                   lut.amp[list(sub)].mean().item())
            if best_key is None or key > best_key:
                best, best_key = sub, key
        chosen = list(best)
    else:
        raise ValueError(rule)
    chosen = sorted(chosen, key=lambda i: _wrap(lut.phase[i]).item())   # phase order
    cb = lut.response[chosen]
    note = (f"{len(chosen)} of {len(lut)} measured rows; min phase gap "
            f"{math.degrees(min_circular_gap(lut.phase[chosen])):.1f} deg")
    return CodebookInfo(cb, chosen, [lut.control[i] for i in chosen], K, note)


def phase_order(codebook: torch.Tensor):
    """Sort any codebook by phase (returns sorted codebook, permutation)."""
    perm = torch.argsort(_wrap(codebook.angle()))
    return codebook[perm], perm


# ----------------------------------------------------------------------------
def per_element_variation(codebook: torch.Tensor, N: int, phase_std_deg=0.0,
                          amp_std=0.0, seed=0) -> torch.Tensor:
    """(N, 2^K) element-dependent codebooks: fixed random phase error (deg, std)
    and relative amplitude error per element and state (fabrication spread)."""
    g = torch.Generator().manual_seed(seed)
    S = codebook.shape[-1]
    dphi = torch.randn(N, S, generator=g, dtype=torch.float64) * math.radians(phase_std_deg)
    damp = (1 + amp_std * torch.randn(N, S, generator=g, dtype=torch.float64)).clamp_min(0.05)
    return (codebook.to(torch.complex128)[None, :] * damp * torch.exp(1j * dphi))
