"""Supplementary analysis / consistency checks for the IEEE Access results.

make_figures.py makes the figures, tables, in-text macros (tables/numbers.tex)
and results_summary.md.  This script collects the extra checks that were used
to verify statements in the text that are NOT a single macro: win rates,
worst cases per aperture, per-instance spreads, the beam-pattern differences
and the Gray/P ablation numbers.

Also the checks for the measured-data additions (2026-10): the digitized
4-bit table (e6 table, e11 measured4) and the BRISC channels (e12).

Usage:  python analyze_results.py [--res DIR[,DIR,...]] [--out FILE]
        default --res: $MPR_RESULTS, else experiments/results_v2 if a new campaign
        exists, else the archived results of the paper (results/main, measured, ablation)
Output: outputs/analysis_checks.md
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # repository root
PKG_RES = [os.path.join(ROOT, "results", d) for d in ("main", "measured", "ablation")]
NEW_RES = os.path.join(ROOT, "experiments", "results_v2")    # a fresh run_experiments.sh campaign
E4_DIRS = [PKG_RES[2], os.path.join(ROOT, "experiments", "results")]


def default_res():
    if os.path.isdir(NEW_RES):
        return NEW_RES
    return ",".join(d for d in PKG_RES if os.path.isdir(d))
OUT = []


def load(res, name):
    """res: folder or comma-separated folders; name.csv and name_part*.csv are merged."""
    files = []
    for r in str(res).split(","):
        p = os.path.join(r, f"{name}.csv")
        files += ([p] if os.path.exists(p) else []) + sorted(glob.glob(os.path.join(r, f"{name}_part*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def gap(df, keys):
    b = df.groupby(keys).y.transform("min")
    df = df.copy()
    df["gap"] = ((df.y - b) / (df.y_rand - b)).clip(lower=0)
    return df


def sec(title, *blocks):
    OUT.append(f"## {title}\n")
    for b in blocks:
        OUT.append(b.to_markdown() if isinstance(b, (pd.DataFrame, pd.Series)) else str(b))
        OUT.append("")


def check_main(res):
    e2, e3 = load(res, "e2_array"), load(res, "e3_ris")
    if e2 is not None:
        d = gap(e2, ["N", "K", "seed"])
        D = d[d.N == d.N.max()]
        sec("E2 array, N=2025: fraction of instances where a method attains the best found (ties counted)",
            D.assign(best=D.gap <= 1e-9).groupby("method").best.mean().round(3))
        sec("E2 array, N=2025: mean gain / PSL by K (dB) -- are the physical metrics distinguishable?",
            D.groupby(["K", "method"]).gain_dB.mean().unstack().round(3),
            D.groupby(["K", "method"]).psl_dB.mean().unstack().round(3))
        sec("E2 array, N=2025: mean gap x1e3 by K", (D.groupby(["K", "method"]).gap.mean().unstack() * 1e3).round(4))
    if e3 is not None:
        d = gap(e3, ["N", "K", "seed", "codebook"])
        D = d[d.N == d.N.max()]
        for cb in D.codebook.unique():
            E = D[D.codebook == cb]
            sec(f"E3 RIS {cb}, N=2025: best-found fraction", E.assign(best=E.gap <= 1e-9).groupby("method").best.mean().round(3))
            sec(f"E3 RIS {cb}, N=2025: which method has the smallest mean gap per K",
                (E.groupby(["K", "method"]).gap.mean().unstack() * 1e3).idxmin(axis=1))
        P = D[D.codebook == "practical"].pivot_table(index=["K", "seed"], columns="method", values="user_mean_dB")
        sec("E3 RIS practical, N=2025: paired user-power loss vs MPR+SR (dB), per K",
            pd.DataFrame({m: (P["MPR+SR"] - P[m]).groupby("K").mean() for m in P.columns if m != "MPR+SR"}).round(3))


def check_optimality(res):
    e = load(res, "e11_optimality")
    if e is None:
        return
    g = e.groupby(["N", "method"]).loss_dB
    sec("E11: worst-case and mean loss to the global optimum (dB) per aperture",
        g.max().unstack(0).round(4), g.mean().unstack(0).round(5))
    sec("E11: fraction of cases within 1e-4 dB of the optimum, per aperture",
        e.assign(ok=e.loss_dB < 1e-4).groupby(["N", "method"]).ok.mean().unstack(0).round(3))
    sec("E11: five worst MPR+SR cases", e[e.method == "MPR+SR"].nlargest(5, "loss_dB")[["N", "codebook", "K", "seed", "loss_dB"]])
    sec("E11: most negative loss (float32 round-off check)", f"{e.loss_dB.min():.3e} dB")


def check_anytime(res):
    e = load(res, "e5_anytime")
    if e is None:
        return
    e = gap(e, ["scenario", "seed"])
    w = e[(e.method == "MPR") & (e.budget == 10)].groupby("scenario").time_s.describe()
    sec("E5: first-call warm-up check (MPR, T=10; the median is used in the figure)", w.round(4))
    for sc in e.scenario.unique():
        d = e[(e.scenario == sc) & e.method.isin(["SR", "MPR+SR"]) & (e.budget_kind == "sweeps")]
        sec(f"E5 {sc}: median gap x1e3 per number of sweeps, MPR+SR vs SR",
            (d.groupby(["budget", "method"]).gap.median().unstack() * 1e3).round(4))


def check_measured(res):
    e = load(res, "e6_measured")
    if e is None:
        return
    P = e.pivot_table(index=["N", "seed"], columns="method", values="user_mean_dB")
    L = {m: (P["MPR+SR"] - P[m]).groupby("N") for m in P.columns if m != "MPR+SR"}
    sec("E6: paired user-power loss vs MPR+SR (dB): mean / min / max over instances",
        pd.DataFrame({m: l.mean() for m, l in L.items()}).round(3),
        pd.DataFrame({m: l.min() for m, l in L.items()}).round(2),
        pd.DataFrame({m: l.max() for m, l in L.items()}).round(2))
    mp = os.path.join(res, "e6_codebooks.json")
    if os.path.exists(mp):
        lut = json.load(open(mp))["lut"]
        ph = np.array(lut["phase_deg"])
        ideal = (np.arange(8) * 45 + 180) % 360 - 180
        near = [lut["control"][int(np.argmin(np.abs((p - ph + 180) % 360 - 180)))] for p in ideal]
        sec("E6: nearest measured state (bias V) for each uniform 3-bit phase",
            pd.Series(near, index=[f"{p:g} deg" for p in ideal]))


def check_variation_fairness(res):
    e = load(res, "e7_variation")
    if e is not None:
        e = gap(e, ["scenario", "seed", "phase_std", "amp_std"])
        sec("E7: max mean gap x1e3 of MPR+SR and SR over all spreads",
            (e[e.method.isin(["MPR+SR", "SR"])].groupby(["scenario", "phase_std", "amp_std", "method"]).gap.mean()
             .unstack() * 1e3).max().round(3))
    f = load(res, "e8_fairness")
    if f is not None:
        sec("E8: median leakage per method (dB)", f.groupby("method").leak_mean_dB.median().round(2))


def check_pattern(res):
    p = load(res, "e10_pattern")
    if p is None:
        return
    m = p["MPR+SR"] > -40
    sec("E10: |CQ - MPR+SR| in the cut where MPR+SR > -40 dB",
        (p.CQ - p["MPR+SR"]).abs()[m].describe().round(3))
    sec("E10: expected 2-bit uniform quantization loss", f"{-10 * np.log10((np.sin(np.pi / 4) / (np.pi / 4)) ** 2):.3f} dB")


def check_ablation(res):
    e = load(res, "e4_ablation")
    if e is None:
        e = load(",".join(d for d in E4_DIRS if os.path.isdir(d)), "e4_ablation")
    if e is None:
        return
    b = e.groupby(["scenario", "K", "seed"]).y.transform("min")
    e = e.assign(rl=100 * (e.y_round - b) / b.abs(), fl=100 * (e.y - b) / b.abs())
    sec(f"E4 ({e.seed.nunique()} instances): loss after rounding (%) at P=256",
        e[e.P == 256].groupby(["scenario", "mapping", "K"]).rl.mean().unstack().round(4))
    sec("E4: final gap (%) vs P, Gray mapping", e[e.mapping == "gray"].groupby(["scenario", "P"]).fl.mean().unstack().round(4))
    sec("E4: relaxation time (s) vs P, Gray mapping", e[e.mapping == "gray"].groupby("P").t_relax.mean().round(3))


def check_measured4(res):
    """digitized 4-bit table of Zhang et al. (e6 table, e11 measured4)."""
    t = load(res, "e6_measured_table")
    if t is not None:
        P = t.pivot_table(index=["codebook", "N", "seed"], columns="method", values="user_mean_dB")
        loss = P.drop(columns="MPR+SR").rsub(P["MPR+SR"], axis=0)
        sec("E6 4-bit: paired loss in mean user power vs MPR+SR (dB), mean / max over 20 instances",
            loss.groupby(level=["codebook", "N"]).mean().round(3),
            loss.groupby(level=["codebook", "N"]).max().round(3))
        lut = load(res, "e6_measured")
        if lut is not None:
            Q = lut.pivot_table(index=["N", "seed"], columns="method", values="user_mean_dB")
            l3 = Q.drop(columns="MPR+SR").rsub(Q["MPR+SR"], axis=0).xs(2025, level="N").mean()
            l4 = loss.xs(2025, level="N").groupby(level="codebook").mean().T
            sec("Measured codebooks compared at N=2025: mean loss vs MPR+SR (dB)",
                pd.concat([l3.rename("3-bit varactor LUT"), l4], axis=1).round(3))
    d = load(res, "e6_dualband_table")
    if d is not None:
        sec("E6 dual-band: mean user power (dB) by N / design / band",
            d.groupby(["N", "design", "band"]).user_mean_dB.mean().unstack().round(2))
    o = load(res, "e11_measured4")
    if o is not None:
        o = o.assign(loss_dB=o.loss_dB.clip(lower=0))
        sec("E11 4-bit: loss to the exact optimum (dB): max and fraction < 1e-4 dB",
            o.groupby(["N", "codebook", "method"]).loss_dB.max().unstack().round(4),
            o.assign(ok=o.loss_dB < 1e-4).groupby(["N", "method"]).ok.mean().unstack().round(3))


def check_brisc(res):
    """BRISC measured channels (e12)."""
    f = load(res, "e12_brisc_fit")
    if f is not None:
        f = f.sort_values("pos")
        f["range_dB"] = f.meas_max_dB - f.meas_min_dB
        sec("E12 model check per TX position",
            f.set_index("pos")[["nmse_dB", "perr_med_dB", "perr_p90_dB", "perr_bias_dB", "spearman",
                                "test_pick_rank", "test_best_meas_dB", "test_pick_meas_dB", "range_dB"]].round(3))
    d = load(res, "e12_brisc")
    if d is None:
        return
    W = d[d.task == "wb-max"]
    per = W[W.method != "best-measured"].groupby(["pos", "method"]).p_dB.median().unstack()
    bm = W[W.method == "best-measured"].groupby("pos")[["p_dB", "p_meas_dB"]].first()
    sec("E12 wideband max: predicted power (dB) per position, best measured (measured / predicted)",
        per.join(bm).round(2))
    sec("E12 wideband max: SR - MPR+SR per position (dB, >0 = SR better)", (per["SR"] - per["MPR+SR"]).round(3))
    sec("E12 wideband max: spread over the 5 seeds (max - min, dB)",
        W[W.method != "best-measured"].groupby(["pos", "method"]).p_dB.agg(lambda v: v.max() - v.min()).unstack().round(3))
    S = d[d.task == "sc-max"].assign(loss_dB=lambda x: x.loss_dB.clip(lower=0))
    S = S[S.method != "exact-opt"]
    sec("E12 one subcarrier: loss to exact optimum (dB)",
        S.groupby("method").loss_dB.describe(percentiles=[.5, .9]).round(4),
        "fraction < 1e-4 dB by position:",
        S.assign(ok=S.loss_dB < 1e-4).groupby(["pos", "method"]).ok.mean().unstack().round(2))
    m = S[S.method == "MPR+SR"]
    sec("E12 one subcarrier: MPR+SR result identical over the 5 seeds? (std of loss per pos/sub)",
        m.groupby(["pos", "sub"]).loss_dB.std().unstack().round(4))
    M = d[d.task == "wb-min"]
    if not M.empty:
        pm = M[M.method != "best-measured"].groupby(["pos", "method"]).p_dB.median().unstack()
        bmin = M[M.method == "best-measured"].groupby("pos")[["p_dB", "p_meas_dB"]].first()
        sec("E12 wideband min (not used in the paper: model error at nulls)", pm.join(bmin).round(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.environ.get("MPR_RESULTS") or default_res())
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "analysis_checks.md"))
    a = ap.parse_args()
    OUT.append(f"# Analysis checks (generated by analysis/analyze_results.py)\n\nsource: `{', '.join(os.path.basename(r) for r in a.res.split(','))}`\n")
    for f in (check_main, check_optimality, check_anytime, check_measured, check_variation_fairness,
              check_pattern, check_ablation, check_measured4, check_brisc):
        f(a.res)
    txt = "\n".join(OUT) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
