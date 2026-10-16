# Measured codebooks

`run_experiments.sh` (via `experiments/run_campaign.py`) looks for the two files below and
skips the corresponding experiments if a file is missing.

## 1. Measured unit-cell look-up table (Fara et al., 3-bit codebook)

- File: `unit_cell_look_up_table.txt` (original name, unchanged)
- Source: R. Fara et al., "Look-up table of a prototype of reconfigurable intelligent surface
  with continuous control of the reflection phase," Zenodo, 2023,
  doi:10.5281/zenodo.7674708, **CC BY 4.0**; described in R. Fara et al.,
  *IEEE Wireless Commun.*, vol. 29, no. 1, pp. 70–77, 2022.
- Format: one row per bias voltage, `voltage[V];Magnitude[dB];Phase[degrees]`;
  header and comment lines are skipped automatically.
- Use: `select_states()` *selects* 2^K of the measured rows (default K = 3, i.e. 8 states)
  by maximizing the minimum circular phase gap (ties: larger mean amplitude).
  No interpolation or fitting. An amplitude floor can be set with `--amp_floor_dB`.
- The selected rows, voltages, amplitudes and phases are written to `e6_codebooks.json`
  in the results folder (`e6_codebooks_table.json` for the 16-state table).

Check:
```bash
python -c "from mpr import *; l=load_lut('data/codebooks/unit_cell_look_up_table.txt'); i=select_states(l,3); print(i.note, i.control)"
```

## 2. 16-state table of a 4-bit prototype (Zhang et al.)

- File: `measured_16state.csv` — **digitized** from Fig. 6 of J. W. Zhang et al.,
  *Adv. Electron. Mater.*, e00446, 2025, doi:10.1002/aelm.202500446, at 3.6 and 4.0 GHz.
  These are not the authors' raw data; procedure and accuracy (±2° phase, ±0.2–0.3 dB
  amplitude) are documented in `measured_16state_SOURCE.md`.
- Format (see `TEMPLATE_16state.csv`, which contains placeholders, not measurements):

```
code,freq_GHz,amp_dB,phase_deg
0,3.6,<amplitude dB>,<phase deg>
...
```

- Each frequency is used as its own 4-bit codebook. With two frequencies, a dual-band design
  that shares one set of control bits is also computed (two components of `FieldObjective`)
  and cross-evaluated against single-band designs.
- Linear amplitudes must be converted to dB (20·log10|Γ|).

## Hardware mapping

The optimizer labels states internally by phase order and Gray code. To deploy a result,
map each element's bit pattern to the codebook state with `bit_to_state_map(K)` and then to
the hardware row with `CodebookInfo.hw_index` / `CodebookInfo.control`
(see `examples/quickstart.py`).
