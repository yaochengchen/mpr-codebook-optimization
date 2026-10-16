# measured_16state.csv — provenance

**Source.** J. W. Zhang et al., "A general and efficient framework for the rapid design of
miniaturized, wideband, and high-bit RIS," *Adv. Electron. Mater.*, e00446, 2025,
doi:10.1002/aelm.202500446 (open access). Figure 6(d)–(g): measured reflection amplitude
and phase of the 20×10 varactor RIS prototype for the 16 bias states "Exp-0" … "Exp-15",
3.4–3.8 GHz (d, e) and 3.8–4.2 GHz (f, g).

**These are digitized values, not the authors' raw data.** The raw data are "available from
the corresponding author upon reasonable request"; if they arrive, overwrite this file
(same format) and rerun e6.

## How the numbers were obtained (2026-10-09)

* The embedded raster of Fig. 6 was extracted from the article PDF (1500×504 px, JPEG).
* Axes were calibrated from the tick labels: amplitude (d) 15.67 px/dB, (f) 13.20 px/dB;
  phase (e) 0.502 px/deg, (g) 0.538 px/deg; frequency 63 px per 0.1 GHz.
* **Phase** at 3.6 GHz (panel e) and 4.0 GHz (panel g): the 16 phase curves do not cross,
  are separated by ~11–12 px at these frequencies, and their vertical order was identified
  from the legend colours (top to bottom: 7, 6, 5, 4, 3, 2, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0).
  Each value is the darkness-weighted centre of the curve, averaged over 3–5 pixel columns
  around the target frequency. Estimated uncertainty about ±2°.
  Check: the 16 phases are spaced 21.7–23.6° (3.6 GHz) and 21.3–23.8° (4.0 GHz),
  i.e. the expected uniform 22.5° grid of a 4-bit cell.
* **Amplitude** at 3.6 GHz (panel d) and 4.0 GHz (panel f): the curves cross, so each state
  was located visually on magnified crops and then refined by a colour-matched line fit over
  ±5 pixel columns. Estimated uncertainty ±0.2 dB, about ±0.3 dB for states 1, 6, 8, 13
  (overlapping dark/light curves). The amplitude pairs cross symmetrically around the
  resonance (e.g. states 3/10, 2/11, 15/12, 14/13 at 3.6 GHz), as expected physically.
* Phases are wrapped to [-180°, 180°). `code` is the state label of the legend ("Exp-code").

## What to write in the paper

> "The 16-state responses at 3.6 and 4.0 GHz were digitized from Fig. 6 of [zhang2025]
> (estimated accuracy ±2° in phase and ±0.2–0.3 dB in amplitude)."
