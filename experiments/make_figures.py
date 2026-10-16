"""Figures, tables and in-text numbers for the IEEE Access version.

Usage:  python make_figures.py [--res DIR[,DIR,...]] [--out ../outputs]

Inputs : $MPR_RESULTS if set; else experiments/results_v2 if a new campaign was
         run (run_experiments.sh); else the archived results of the paper
         (results/main, results/measured, results/ablation).
         Several folders can be given with commas; files are merged, and split
         runs  <exp>_part_*.csv  are merged automatically.  The Gray/P ablation
         (e4) is not part of the v2 campaign and is read from results/ablation.
Outputs: <out>/figs/*.pdf         all figures (default <out> = outputs/)
         <out>/tables/*.tex       table bodies (array.tex, ris.tex)
         <out>/tables/numbers.tex \\newcommand macros for every number quoted
                                  in the text (so the text follows the data)
         <res>/results_summary.md human-readable analysis of all experiments
Every item is skipped silently if its CSV is missing, so this can run at any
stage.  Gap statistics: normalized gap (y - y*)/(mean y_rand - y*), y* = best
found by any method on that instance."""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # repository root
_pkg_res = [os.path.join(ROOT, "results", d) for d in ("main", "measured", "ablation")]
if os.environ.get("MPR_RESULTS"):
    RES_DEFAULT = os.path.abspath(os.environ["MPR_RESULTS"])
elif os.path.isdir(os.path.join(HERE, "results_v2")):     # a fresh campaign run with run_experiments.sh
    RES_DEFAULT = os.path.join(HERE, "results_v2")
else:                                                      # the archived results of the paper
    RES_DEFAULT = ",".join(d for d in _pkg_res if os.path.isdir(d))
RES_OLD = _pkg_res[2] if os.path.isdir(_pkg_res[2]) else os.path.join(HERE, "results")   # e4 ablation
PAPER_DEFAULT = os.path.join(ROOT, "outputs")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.fontsize": 6.5,
    "legend.frameon": False, "figure.dpi": 150, "pdf.fonttype": 42})
W1, W2 = 3.5, 7.16           # IEEE single / double column (in)
# validated 8-slot categorical order + distinct markers / dashes (never colour alone)
STYLE = {"MPR+SR": ("#2a78d6", "o", "-"), "MPR": ("#eb6834", "^", "-"),
         "SR": ("#1baf7a", "s", "--"), "CQ+SR": ("#eda100", "D", "--"),
         "CQ": ("#e87ba4", "x", ":"), "GA": ("#008300", "v", ":"),
         "MO": ("#4a3aa7", "P", ":"), "MO+SR": ("#e34948", "*", "--"),
         "MPR+SR-dedup": ("#0d3b78", "p", "-"),
         "ideal-design": ("#52514e", "h", "-."), "nominal-design": ("#52514e", "h", "-.")}
ORDER = ["CQ", "CQ+SR", "MO", "MO+SR", "GA", "SR", "MPR", "MPR+SR", "MPR+SR-dedup", "ideal-design", "nominal-design"]
FLOOR = 1e-4                 # plotting floor for gaps in units of 1e-3 (log axes)

NUM = {}                     # macro name -> formatted string  (tables/numbers.tex)
REPORT = []                  # lines of results_summary.md


def num(name, value, fmt="{:.2f}"):
    NUM[name] = fmt.format(value) if not isinstance(value, str) else value
    return NUM[name]


def rep(*lines):
    REPORT.extend(lines)


# ------------------------------------------------------------------ loading
def load(res, name):
    """name.csv plus any split runs name_part*.csv; res is one folder or a list of folders."""
    dirs = res if isinstance(res, (list, tuple)) else [res]
    files = []
    for r in dirs:
        p = os.path.join(r, f"{name}.csv")
        files += ([p] if os.path.exists(p) else []) + sorted(glob.glob(os.path.join(r, f"{name}_part*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def find(res, fname):
    """first existing res/<fname> (res: folder or list of folders), else None."""
    for r in (res if isinstance(res, (list, tuple)) else [res]):
        p = os.path.join(r, fname)
        if os.path.exists(p):
            return p
    return None


def gap(df, keys):
    best = df.groupby(keys)["y"].transform("min")
    df = df.copy()
    df["gap"] = ((df["y"] - best) / (df["y_rand"] - best)).clip(lower=0)
    return df


def save(fig, out, name):
    if not fig.legends:
        fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "figs", name))
    plt.close(fig)
    print("  fig", name)


def style(m):
    return STYLE.get(m, ("#52514e", "o", "-"))


def methods_in(d):
    return [m for m in ORDER if m in set(d.method)]


def time_to_reach(curve, target):
    """curve: DataFrame(t, g) of median time / median gap over a budget ladder.
    Returns the (log-log interpolated) time at which g first falls to target."""
    c = curve.sort_values("t")
    t, g = c.t.values, np.maximum(c.g.values, 1e-12)
    if g[0] <= target:
        return t[0]
    for i in range(1, len(t)):
        if g[i] <= target:
            f = (np.log(target) - np.log(g[i - 1])) / (np.log(g[i]) - np.log(g[i - 1]))
            return float(np.exp(np.log(t[i - 1]) + f * (np.log(t[i]) - np.log(t[i - 1]))))
    return np.nan


# ------------------------------------------------------------------ E1b scaling
def do_scaling(df, out):
    d = df.sort_values("N")
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.62, 2.3))
    for col, lab, c, mk, ls in (("t_autograd_ms", "gradient, autograd", "#eb6834", "s", "--"),
                                ("t_grad_ms", "gradient, closed form", "#2a78d6", "o", "-")):
        e = d[d[col].notna()]
        ax[0].loglog(e.N, e[col], ls, color=c, marker=mk, label=lab)
    ax[0].set_ylabel("time per evaluation (ms)"); ax[0].set_xlabel("elements $N$")
    ax[0].legend()
    ax[1].loglog(d.N, d.t_mpr_s, "-", color="#2a78d6", marker="o", label="MPR, full run ($T=300$)")
    ax[1].loglog(d.N, d.t_sr_sweep_s, "--", color="#1baf7a", marker="s", label="SR, one sweep")
    ax[1].set_ylabel("time (s)"); ax[1].set_xlabel("elements $N$"); ax[1].legend()
    save(fig, out, "scaling.pdf")
    if d.mem_mpr_MB.notna().any():
        fig, a = plt.subplots(figsize=(W1, 2.0))
        a.loglog(d.N, d.mem_mpr_MB, "-o", color="#2a78d6", label="MPR peak GPU memory")
        a.loglog(d.N, d.mem_B_MB, "--s", color="#52514e", label="channel matrix $B$")
        a.set_xlabel("elements $N$"); a.set_ylabel("memory (MiB)"); a.legend()
        save(fig, out, "memory.pdf")
    e = d[d.t_autograd_ms.notna()]
    sp = e.t_autograd_ms / e.t_grad_ms
    small = d[d.N <= 4096]
    big = d.iloc[-1]
    num("ScaleGradSmallMin", small.t_grad_ms.min()); num("ScaleGradSmallMax", small.t_grad_ms.max())
    num("ScaleAutogradMin", e.t_autograd_ms.min(), "{:.1f}"); num("ScaleAutogradMax", e.t_autograd_ms.max(), "{:.1f}")
    num("ScaleSpeedupMax", sp.max(), "{:.0f}"); num("ScaleSpeedupMin", sp.min(), "{:.1f}")
    num("ScaleSpeedupMinN", f"{int(e.N[sp.idxmin()])}")
    num("ScaleSpeedupSmallMin", (small.t_autograd_ms / small.t_grad_ms).min(), "{:.0f}")
    num("ScaleNmax", f"{int(big.N)}"); num("ScaleGradNmax", big.t_grad_ms)
    num("ScaleMPRsmall", small.t_mpr_s.mean(), "{:.2f}"); num("ScaleMPRNmax", big.t_mpr_s, "{:.2f}")
    num("ScaleSRsweepNmax", big.t_sr_sweep_s, "{:.1f}")
    num("ScaleSRsweepPerKelem", 1e3 * (d.t_sr_sweep_s / d.N).median(), "{:.2f}")
    num("ScaleMemNmax", big.mem_mpr_MB / 1024, "{:.2f}"); num("ScaleMemBNmax", big.mem_B_MB, "{:.0f}")
    num("ScaleMemNmin", d.iloc[0].mem_mpr_MB, "{:.0f}")
    rep("## E1b scaling (M=500, K=5, P=64)", "",
        d[["N", "t_grad_ms", "t_autograd_ms", "t_mpr_s", "t_sr_sweep_s", "mem_mpr_MB"]].round(3).to_markdown(index=False), "",
        f"- closed form vs autograd: {sp.min():.1f}x (N={int(e.N[sp.idxmin()])}) to {sp.max():.0f}x; "
        f"closed form flat at ~{small.t_grad_ms.mean():.2f} ms up to N=4096 (launch bound), then ~linear in N.",
        f"- MPR full run ~{small.t_mpr_s.mean():.2f} s up to N=4096, {big.t_mpr_s:.2f} s at N={int(big.N)}; "
        f"SR one sweep grows linearly ({big.t_sr_sweep_s:.1f} s at N={int(big.N)}).", "")


# ------------------------------------------------------------------ E2/E3 main
def do_main(e2, e3, out):
    """Tables at N=Nmax over all K, gap-vs-K figure, numbers."""
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.25), sharey=True)
    panels = []
    if e2 is not None:
        panels.append(("array, ideal codebook", gap(e2, ["N", "K", "seed"])))
    if e3 is not None:
        g3 = gap(e3, ["N", "K", "seed", "codebook"])
        panels.append(("RIS, ideal codebook", g3[g3.codebook == "ideal"]))
        panels.append(("RIS, practical codebook", g3[g3.codebook == "practical"]))
    for ax, (title, df) in zip(axes, panels):
        d = df[df.N == df.N.max()]
        for m in methods_in(d):
            if m == "GA":
                continue
            c, mk, ls = style(m)
            g = d[d.method == m].groupby("K").gap.mean() * 1e3
            ax.semilogy(g.index, np.maximum(g.values, FLOOR), ls, color=c, marker=mk, label=m)
        ax.set_title(f"{title}, $N={d.N.max()}$", fontsize=7.5)
        ax.set_xlabel("bits per element $K$"); ax.set_xticks(sorted(d.K.unique()))
    axes[0].set_ylabel(r"mean normalized gap ($10^{-3}$)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="upper center", fontsize=6, bbox_to_anchor=(0.5, 1.0))
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    save(fig, out, "gap_vs_K.pdf")

    if e2 is not None:
        d = gap(e2, ["N", "K", "seed"])
        write_table(d[d.N == d.N.max()], "array.tex", ["gain_dB", "psl_dB"], out)
        rep("## E2 array (100 instances x 3 sizes x K=1..5)", "",
            "mean gap x1e3 by N:", "", (d.groupby(["N", "method"]).gap.mean().unstack() * 1e3).round(4).to_markdown(), "",
            "mean time (s) by N:", "", d.groupby(["N", "method"]).time_s.mean().unstack().round(3).to_markdown(), "")
        D = d[d.N == d.N.max()]
        G = D.groupby("method").agg(g=("gap", "mean"), t=("time_s", "mean"))
        num("ArrGapMPR", 1e3 * G.g["MPR"], "{:.3f}"); num("ArrGapMPRSR", 1e3 * G.g["MPR+SR"], "{:.3f}")
        num("ArrGapSR", 1e3 * G.g["SR"], "{:.3f}"); num("ArrGapCQ", 1e3 * G.g["CQ"], "{:.2f}")
        num("ArrGapMO", 1e3 * G.g["MO"], "{:.2f}"); num("ArrGapGA", 1e3 * G.g["GA"], "{:.0f}")
        num("ArrTimeMPR", G.t["MPR"]); num("ArrTimeSR", G.t["SR"], "{:.1f}"); num("ArrTimeMPRSR", G.t["MPR+SR"], "{:.1f}")
        num("ArrTimeMO", G.t["MO"], "{:.1f}"); num("ArrTimeCQ", G.t["CQ"])
        num("ArrRatioCQMPR", G.g["CQ"] / G.g["MPR"], "{:.0f}")
        best = D.assign(b=D.gap <= 1e-9).groupby("method").b.mean()
        num("ArrBestMPRSR", 100 * best["MPR+SR"], "{:.0f}"); num("ArrBestSR", 100 * best["SR"], "{:.0f}")
        psl = D.groupby("method").psl_dB.mean()
        num("ArrPSLspread", psl.drop("GA").max() - psl.drop("GA").min(), "{:.2f}")
    if e3 is not None:
        g3 = gap(e3, ["N", "K", "seed", "codebook"])
        D = g3[(g3.N == g3.N.max())]
        write_table(D[D.codebook == "practical"], "ris.tex", ["user_mean_dB", "user_min_dB", "leak_mean_dB"], out)
        for cb, tag in (("ideal", "Id"), ("practical", "Pr")):
            E = D[D.codebook == cb]
            G = E.groupby("method").agg(g=("gap", "mean"), t=("time_s", "mean"))
            for m, s in (("MPR", "MPR"), ("MPR+SR", "MPRSR"), ("SR", "SR"), ("CQ", "CQ"), ("MO", "MO"),
                         ("CQ+SR", "CQSR"), ("MO+SR", "MOSR")):
                num(f"Ris{tag}Gap{s}", 1e3 * G.g[m], "{:.3f}" if G.g[m] < 1e-2 else "{:.0f}")
            num(f"Ris{tag}TimeMPR", G.t["MPR"]); num(f"Ris{tag}TimeSR", G.t["SR"], "{:.1f}")
            rep(f"## E3 RIS, {cb} codebook", "",
                "mean gap x1e3 by N:", "", (g3[g3.codebook == cb].groupby(["N", "method"]).gap.mean().unstack() * 1e3).round(3).to_markdown(), "",
                f"mean gap x1e3 by K (N={D.N.max()}):", "", (E.groupby(["K", "method"]).gap.mean().unstack() * 1e3).round(3).to_markdown(), "",
                "mean user power (dB) by K:", "", E.groupby(["K", "method"]).user_mean_dB.mean().unstack().round(2).to_markdown(), "")
        P = D[D.codebook == "practical"].pivot_table(index=["K", "seed"], columns="method", values="user_mean_dB")
        lossCQ = (P["MPR+SR"] - P["CQ"]).groupby("K").mean()
        lossMO = (P["MPR+SR"] - P["MO"]).groupby("K").mean()
        num("RisCQlossKthree", lossCQ.loc[3]); num("RisCQlossKfive", lossCQ.loc[5])
        num("RisCQlossKtwo", lossCQ.loc[2]); num("RisMOlossKfive", lossMO.loc[5])
        I = D[D.codebook == "ideal"].groupby(["K", "method"]).gap.mean().unstack() * 1e3
        num("RisIdMOSRKfour", I.loc[4, "MO+SR"], "{:.3f}"); num("RisIdMPRSRKfour", I.loc[4, "MPR+SR"], "{:.3f}")
        S = D[(D.codebook == "practical") & (D.method == "MPR+SR")]
        num("RisSpreadMeanMin", (S.user_mean_dB - S.user_min_dB).mean(), "{:.0f}")
        rep("practical codebook: mean user-power loss of CQ / MO vs MPR+SR by K (dB):", "",
            pd.DataFrame({"CQ": lossCQ, "MO": lossMO}).round(2).to_markdown(), "")


def write_table(d, fname, cols, out):
    with open(os.path.join(out, "tables", fname), "w") as f:
        for m in methods_in(d):
            e = d[d.method == m]
            g = 1e3 * e.gap
            fmt = "{:.3f}" if g.mean() < 10 else "{:.1f}"
            cells = [(fmt + r" $\pm$ " + fmt).format(g.mean(), g.std())]
            cells += [f"${e[c].mean():.2f}$" for c in cols]
            cells += [f"{e.time_s.mean():.2f}"]
            name = r"\textbf{" + m + "}" if m.startswith("MPR") else m
            f.write(name + " & " + " & ".join(cells) + " \\\\\n")
    print("  table", fname)


# ------------------------------------------------------------------ E5 anytime
def do_anytime(df, out):
    df = gap(df, ["scenario", "seed"])
    scen = list(df.scenario.unique())
    curves = {}
    fig, axes = plt.subplots(1, len(scen), figsize=(W2 * 0.75, 2.4), sharey=True, squeeze=False)
    for ax, sc in zip(axes[0], scen):
        d = df[df.scenario == sc]
        for m in methods_in(d):
            e = d[d.method == m]
            g = e.groupby("budget").agg(t=("time_s", "median"), g=("gap", "median"),
                                        lo=("gap", lambda s: s.quantile(.25)),
                                        hi=("gap", lambda s: s.quantile(.75))).sort_values("t")
            curves[(sc, m)] = g
            c, mk, ls = style(m)
            ax.plot(g.t, np.maximum(1e3 * g.g, FLOOR), ls, color=c, marker=mk, label=m)
            ax.fill_between(g.t, np.maximum(1e3 * g.lo, FLOOR), np.maximum(1e3 * g.hi, FLOOR),
                            color=c, alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title({"array": "array, ideal 3-bit", "ris_practical": "RIS, practical 3-bit"}.get(sc, sc),
                     fontsize=7.5)
        ax.set_xlabel("wall-clock time (s)")
    axes[0][0].set_ylabel(r"median normalized gap ($10^{-3}$)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="upper center", fontsize=6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    save(fig, out, "anytime.pdf")
    rep("## E5 anytime (N=2025, K=3, 20 instances): median time / median gap x1e3 per budget", "")
    for sc in scen:
        rows = []
        for m in methods_in(df[df.scenario == sc]):
            g = curves[(sc, m)]
            rows += [dict(method=m, budget=b, t=r.t, gap=1e3 * r.g) for b, r in g.iterrows()]
        rep(f"### {sc}", "", pd.DataFrame(rows).round(4).to_markdown(index=False), "")
    # equal-quality speed-ups: time for SR (and MO+SR) to reach MPR's default (T=300) median gap
    for sc, tag in (("array", "Arr"), ("ris_practical", "Ris")):
        if (sc, "MPR") not in curves:
            continue
        mp = curves[(sc, "MPR")]
        r300 = mp.loc[300] if 300 in mp.index else mp.iloc[-1]
        t_sr = time_to_reach(curves[(sc, "SR")], r300.g)
        num(f"Any{tag}MPRtime", r300.t, "{:.3f}"); num(f"Any{tag}MPRgap", 1e3 * r300.g, "{:.4f}")
        num(f"Any{tag}SRtime", t_sr, "{:.1f}"); num(f"Any{tag}Speedup", t_sr / r300.t, "{:.0f}")
        best_cq = min(curves[(sc, "CQ")].g.min(), curves[(sc, "MO")].g.min())
        num(f"Any{tag}CQfloor", 1e3 * best_cq, "{:.2f}" if 1e3 * best_cq < 10 else "{:.0f}")
        rep(f"- {sc}: MPR T=300 median {r300.t:.3f} s, gap {1e3 * r300.g:.4f}e-3; SR needs {t_sr:.2f} s "
            f"for the same gap ({t_sr / r300.t:.0f}x); best CQ/MO plateau {1e3 * best_cq:.3f}e-3")
        if 100 in mp.index:
            r100 = mp.loc[100]
            t1 = time_to_reach(curves[(sc, "SR")], r100.g)
            num(f"Any{tag}MPRhundredTime", 1e3 * r100.t, "{:.0f}")
            num(f"Any{tag}MPRhundredGap", 1e3 * r100.g, "{:.3f}" if 1e3 * r100.g < 1 else "{:.1f}")
            num(f"Any{tag}SRhundredTime", t1, "{:.1f}")
            # speed of a pedestrian whose coherence time (0.423 c / (f_c v), f_c = 3.5 GHz) equals the run time
            num(f"Any{tag}MPRhundredKmh", 3.6 * 0.423 * 3e8 / (3.5e9 * r100.t), "{:.0f}")
            num(f"Any{tag}MPRKmh", 3.6 * 0.423 * 3e8 / (3.5e9 * r300.t), "{:.1f}")
            rep(f"- {sc}: MPR T=100 {r100.t:.3f} s, gap {1e3 * r100.g:.4f}e-3; SR needs {t1:.2f} s")
        if 600 in mp.index:
            r600 = mp.loc[600]
            t2 = time_to_reach(curves[(sc, "SR")], r600.g)
            num(f"Any{tag}MPRsixTime", r600.t); num(f"Any{tag}MPRsixGap", 1e3 * r600.g, "{:.3f}")
            num(f"Any{tag}SpeedupSix", t2 / r600.t, "{:.0f}")
            rep(f"- {sc}: MPR T=600 {r600.t:.3f} s, gap {1e3 * r600.g:.4f}e-3; SR needs {t2:.2f} s ({t2 / r600.t:.0f}x)")
        # final quality: MPR+SR(10 sweeps) vs SR(20 sweeps)
        ms, s = curves[(sc, "MPR+SR")], curves[(sc, "SR")]
        rep(f"- {sc}: MPR+SR 10 sweeps {ms.t.iloc[-1]:.2f} s gap {1e3 * ms.g.iloc[-1]:.4f}e-3; "
            f"SR 20 sweeps {s.t.iloc[-1]:.2f} s gap {1e3 * s.g.iloc[-1]:.4f}e-3")
    rep("")


# ------------------------------------------------------------------ E11 optimum
def do_optimality(df, out):
    df = df.copy()
    df["loss_dB"] = df.loss_dB.clip(lower=0)          # c64 rounding gives |loss| < 2e-6 dB below 0
    Nmax = df.N.max()
    d0 = df[df.N == Nmax]
    books = [b for b in ("ideal", "practical") if b in set(d0.codebook)] + \
            [b for b in d0.codebook.unique() if b.startswith("measured")]
    widths = [1 if not b.startswith("measured") else 0.35 for b in books]
    fig, axes = plt.subplots(1, len(books), figsize=(W2, 2.3), sharey=True, squeeze=False,
                             gridspec_kw=dict(width_ratios=widths))
    for ax, b in zip(axes[0], books):
        d = d0[d0.codebook == b]
        Ks = sorted(d.K.unique())
        ms = methods_in(d)
        for i, m in enumerate(ms):
            c, mk, ls = style(m)
            st = d[d.method == m].groupby("K").loss_dB
            med, lo, hi = st.median(), st.quantile(.25), st.quantile(.75)
            off = (i - (len(ms) - 1) / 2) * (0.07 if len(Ks) > 1 else 0.12)
            ax.errorbar(np.array(med.index) + off, med.values, yerr=[med - lo, hi - med], fmt=mk,
                        ls=ls if len(Ks) > 1 else "none", color=c, ms=4, capsize=1.5, lw=1, label=m)
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.6)
        ax.set_ylim(-1e-4, 6)
        name = {"ideal": "ideal", "practical": "practical (model)"}.get(b, "measured 3-bit")
        ax.set_title(f"{name}", fontsize=7.5); ax.set_xlabel("bits $K$")
        ax.set_xticks(Ks)
        if len(Ks) == 1:
            ax.set_xlim(Ks[0] - 0.6, Ks[0] + 0.6)
    axes[0][0].set_ylabel(f"loss to global optimum (dB), $N={Nmax}$")
    axes[0][0].legend(ncol=2, fontsize=5.5, loc="upper left")
    save(fig, out, "optimality.pdf")

    rep("## E11 single receiver: loss to the exact optimum (dB)", "")
    for N in sorted(df.N.unique()):
        d = df[df.N == N]
        rep(f"### N={N} median", "", d.groupby(["codebook", "K", "method"]).loss_dB.median().unstack().round(4).to_markdown(), "")
    rep(f"### N={Nmax} max", "", d0.groupby(["codebook", "K", "method"]).loss_dB.max().unstack().round(4).to_markdown(), "")
    ex = d0.assign(ok=d0.loss_dB < 1e-4).groupby(["codebook", "method"]).ok.mean().unstack()
    rep(f"### N={Nmax} fraction of instances with loss < 1e-4 dB", "", ex.round(3).to_markdown(), "")
    mx = df.groupby("method").loss_dB.max()
    num("OptMaxMPRSR", mx["MPR+SR"], "{:.4f}"); num("OptMaxMPR", mx["MPR"], "{:.3f}")
    num("OptMaxSR", mx["SR"], "{:.3f}")
    mxN = d0.groupby("method").loss_dB.max()
    num("OptMaxMPRSRbig", mxN["MPR+SR"], "{:.4f}"); num("OptMaxMPRbig", mxN["MPR"], "{:.3f}")
    num("OptCases", f"{len(df[df.method == 'MPR+SR']):,}".replace(",", "{,}"))
    num("OptMeanMPRSR", d0[d0.method == "MPR+SR"].loss_dB.mean(), "{:.5f}")
    num("OptExactMPRSR", 100 * (d0[d0.method == "MPR+SR"].loss_dB < 1e-4).mean(), "{:.0f}")
    num("OptExactSR", 100 * (d0[d0.method == "SR"].loss_dB < 1e-4).mean(), "{:.0f}")
    pr = d0[d0.codebook == "practical"].groupby(["K", "method"]).loss_dB.median().unstack()
    num("OptCQpracKtwo", pr.loc[2, "CQ"]); num("OptCQpracKfive", pr.loc[5, "CQ"])
    num("OptMOpracKfive", pr.loc[5, "MO"])
    ms = d0[d0.codebook.str.startswith("measured")].groupby("method").loss_dB.median()
    if len(ms):
        num("OptCQmeas", ms["CQ"]); num("OptMOmeas", ms["MO"])
    id_ = d0[d0.codebook == "ideal"].groupby("method").loss_dB.max()
    num("OptCQidealMax", id_["CQ"], "{:.3f}")
    num("OptGAmax", d0[d0.method == "GA"].groupby(["codebook", "K"]).loss_dB.median().max(), "{:.1f}")
    num("OptTimeMPR", d0[d0.method == "MPR"].time_s.mean()); num("OptTimeMPRSR", d0[d0.method == "MPR+SR"].time_s.mean(), "{:.1f}")


# ------------------------------------------------------------------ E6 measured
def do_measured(df, meta, out):
    if meta and "lut" in meta:
        L = meta["lut"]
        fig = plt.figure(figsize=(W1, 2.6))
        ax = fig.add_subplot(111, projection="polar")
        r_all = 10 ** (np.array(L["all_amp_dB"]) / 20)
        ax.scatter(np.deg2rad(L["all_phase_deg"]), r_all, s=14, color="#c3c2b7", label="measured (14 bias values)")
        ax.scatter(np.deg2rad(L["phase_deg"]), 10 ** (np.array(L["amp_dB"]) / 20), s=26,
                   color="#2a78d6", label=f"selected {2 ** L['K']} states")
        for c, ph, a in zip(L["control"], L["phase_deg"], L["amp_dB"]):
            ax.annotate(f"{c:g} V", (np.deg2rad(ph), 10 ** (a / 20)), fontsize=5, color=INK2,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_rmax(1.05); ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.15))
        save(fig, out, "measured_codebook.pdf")
        ph = np.sort(np.array(L["phase_deg"]))
        gaps = np.diff(np.r_[ph, ph[0] + 360])
        num("MeasMinGap", gaps.min(), "{:.1f}"); num("MeasMaxGap", gaps.max(), "{:.0f}")
        num("MeasAmpMin", min(L["amp_dB"]), "{:.1f}"); num("MeasAmpMax", max(L["amp_dB"]), "{:.1f}")
        num("MeasBias", ", ".join(f"{v:g}" for v in sorted(L["control"])))
        num("MeasRows", f"{len(L['all_control'])}")
        rep("## E6 measured LUT codebook", "", f"- {L.get('note', '')}; selected bias (V): "
            f"{sorted(L['control'])}; amplitudes {min(L['amp_dB']):.1f} to {max(L['amp_dB']):.1f} dB; "
            f"phase gaps {np.round(gaps, 1).tolist()} deg", "")
    if df is None:
        return
    df = df[df.get("codebook", pd.Series(dtype=str)).astype(str).str.startswith("measured")] if "codebook" in df else df
    df = gap(df, ["codebook", "N", "seed"])
    # paired loss in mean user power relative to MPR+SR on the same instance
    P = df.pivot_table(index=["codebook", "N", "seed"], columns="method", values="user_mean_dB")
    ms = [m for m in ORDER if m in P.columns and m != "MPR+SR"]
    Ns = sorted(df.N.unique())
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for i, m in enumerate(ms):
        c, mk, ls = style(m)
        loss = (P["MPR+SR"] - P[m]).groupby("N")
        med, lo, hi = loss.median(), loss.quantile(.25), loss.quantile(.75)
        x = np.arange(len(Ns)) + (i - (len(ms) - 1) / 2) * 0.06
        ax.errorbar(x, med.values, yerr=[med - lo, hi - med], fmt=mk, ls=ls, color=c, ms=4,
                    capsize=1.5, lw=1, label=m)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("elements $N$"); ax.set_ylabel("loss in mean user power\nvs. MPR+SR (dB)")
    ax.legend(ncol=3, fontsize=5.5, loc="upper right")
    ax.set_ylim(-0.3, None)
    save(fig, out, "measured_results.pdf")
    Nmax = max(Ns)
    loss = {m: (P["MPR+SR"] - P[m]).xs(Nmax, level="N").mean() for m in ms}
    num("MeasLossIdeal", loss["ideal-design"]); num("MeasLossCQ", loss["CQ"]); num("MeasLossMO", loss["MO"])
    num("MeasLossMPR", loss["MPR"], "{:.3f}")
    num("MeasLossIdealMin", (P["MPR+SR"] - P["ideal-design"]).groupby("N").mean().min(), "{:.1f}")
    num("MeasLossIdealMax", (P["MPR+SR"] - P["ideal-design"]).groupby("N").mean().max(), "{:.1f}")
    num("MeasLossCQMin", (P["MPR+SR"] - P["CQ"]).groupby("N").mean().min(), "{:.1f}")
    num("MeasLossCQMax", (P["MPR+SR"] - P["CQ"]).groupby("N").mean().max(), "{:.1f}")
    G = df[df.N == Nmax].groupby("method").agg(g=("gap", "mean"), t=("time_s", "mean"))
    num("MeasGapMPR", 1e3 * G.g["MPR"], "{:.1f}"); num("MeasTimeMPR", G.t["MPR"])
    num("MeasGapCQ", 1e3 * G.g["CQ"], "{:.0f}")
    rep("### E6 results (20 instances per N): mean over instances", "",
        df.groupby(["N", "method"])[["gap", "user_mean_dB", "user_min_dB", "leak_mean_dB", "time_s"]].mean().round(4).to_markdown(), "",
        "mean paired loss in user power vs MPR+SR (dB):", "",
        pd.DataFrame({m: (P["MPR+SR"] - P[m]).groupby("N").mean() for m in ms}).round(3).to_markdown(), "")


# ------------------------------------------------------------------ E7 variation
def do_variation(df, out):
    df = gap(df, ["scenario", "seed", "phase_std", "amp_std"])
    amps = sorted(df.amp_std.unique())
    a_sel = amps[len(amps) // 2]
    scen = list(df.scenario.unique())
    fig, axes = plt.subplots(1, len(scen), figsize=(W2 * 0.7, 2.3), squeeze=False, sharey=True)
    for ax, sc in zip(axes[0], scen):
        d = df[(df.scenario == sc) & (df.amp_std == a_sel)]
        for m in methods_in(d):
            c, mk, ls = style(m)
            g = d[d.method == m].groupby("phase_std").gap.mean()
            ax.semilogy(g.index, np.maximum(1e3 * g.values, FLOOR), ls, color=c, marker=mk, label=m)
        ax.set_title({"array": "array, ideal 3-bit", "ris": "RIS, practical 3-bit"}.get(sc, sc)
                     + f", $\\sigma_a={a_sel:g}$", fontsize=7.5)
        ax.set_xlabel(r"phase spread $\sigma_\phi$ (deg)")
    axes[0][0].set_ylabel(r"mean normalized gap ($10^{-3}$)"); axes[0][-1].legend(fontsize=5.5, ncol=2)
    save(fig, out, "variation.pdf")
    rep("## E7 element-dependent codebooks (N=1024, K=3, 20 instances): mean gap x1e3", "")
    for sc in scen:
        for a in amps:
            d = df[(df.scenario == sc) & (df.amp_std == a)]
            rep(f"### {sc}, amp_std={a:g}", "", (d.groupby(["phase_std", "method"]).gap.mean().unstack() * 1e3).round(3).to_markdown(), "")
    # objective lost by the nominal design, |y_nominal| vs |y_MPR+SR| in dB
    P = df.pivot_table(index=["scenario", "phase_std", "amp_std", "seed"], columns="method", values="y")
    L = (10 * np.log10(P["MPR+SR"] / P["nominal-design"])).groupby(["scenario", "phase_std", "amp_std"]).median()
    rep("nominal-design objective loss vs MPR+SR, median (dB):", "", L.unstack().round(3).to_markdown(), "")
    G = df.groupby(["scenario", "phase_std", "amp_std", "method"]).gap.mean().unstack() * 1e3
    pmax = df.phase_std.max()
    num("VarMaxPhase", f"{pmax:g}"); num("VarAmp", f"{a_sel:g}")
    num("VarMPRSRmax", G["MPR+SR"].max(), "{:.2f}")
    num("VarMPRarrMax", G.loc[("array", pmax, a_sel), "MPR"], "{:.0f}")
    num("VarMPRrisMax", G.loc[("ris", pmax, a_sel), "MPR"], "{:.0f}")
    num("VarMPRarrTen", G.loc[("array", 10.0, a_sel), "MPR"], "{:.2f}")
    num("VarNomArrMax", G.loc[("array", pmax, a_sel), "nominal-design"], "{:.0f}")
    num("VarNomRisMax", G.loc[("ris", pmax, a_sel), "nominal-design"], "{:.0f}")
    num("VarNomArrTen", G.loc[("array", 10.0, a_sel), "nominal-design"], "{:.0f}")
    num("VarNomLossArr", L.loc[("array", pmax, a_sel)]); num("VarNomLossRis", L.loc[("ris", pmax, a_sel)])
    num("VarCQris", G.loc[("ris", 0.0, 0.0), "CQ"], "{:.0f}")


# ------------------------------------------------------------------ E8 fairness
def do_fairness(df, out):
    def tau_of(m):
        return np.inf if m.startswith("sum") else float(m.split("tau=")[1].split()[0])
    cats = sorted({tau_of(m) for m in df.method}, reverse=True)
    lab = ["sum\npower" if np.isinf(t) else f"$\\tau$={t:g}" for t in cats]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for col, c, name in (("user_mean_dB", "#2a78d6", "mean user"), ("user_min_dB", "#eb6834", "weakest user")):
        for solver, mk, mfc, off in (("MPR+SR", "o", None, -0.1), ("MPR", "^", "white", 0.1)):
            xs, med, lo, hi = [], [], [], []
            for i, t in enumerate(cats):
                e = df[(df.method.map(tau_of) == t) & (df.method.str.endswith(" " + solver))][col]
                if e.empty:
                    continue
                xs.append(i + off); med.append(e.median()); lo.append(e.quantile(.25)); hi.append(e.quantile(.75))
            if not xs:
                continue
            med, lo, hi = map(np.array, (med, lo, hi))
            ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt=mk, color=c, mfc=mfc or c, ms=4.5, lw=0.8,
                        capsize=1.5, label=f"{name}, {solver}")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylabel("user power (dB)"); ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.legend(ncol=2, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "fairness.pdf")
    P = df.pivot_table(index="seed", columns="method", values="user_min_dB")
    Q = df.pivot_table(index="seed", columns="method", values="user_mean_dB")
    base = "sum-power MPR+SR"
    soft = [m for m in P.columns if m.startswith("soft") and m.endswith("MPR+SR")]
    dmin = pd.Series({m: (P[m] - P[base]).median() for m in P.columns if m != base})
    dmean = pd.Series({m: (Q[m] - Q[base]).median() for m in Q.columns if m != base})
    num("FairMinGainLo", dmin[soft].min(), "{:.0f}"); num("FairMinGainHi", dmin[soft].max(), "{:.0f}")
    num("FairMeanCostLo", -dmean[soft].max(), "{:.1f}"); num("FairMeanCostHi", -dmean[soft].min(), "{:.1f}")
    num("FairSpreadSum", (Q[base] - P[base]).median(), "{:.0f}")
    num("FairSpreadSoft", np.median([(Q[m] - P[m]).median() for m in soft]), "{:.0f}")
    t = df.groupby("method").time_s.median()
    num("FairTimeMPR", t[[m for m in t.index if m.startswith("soft") and m.endswith(" MPR")]].mean())
    num("FairTimeSumSR", t[base], "{:.1f}")
    num("FairTimeSoftSR", t[soft].mean(), "{:.1f}")
    rep("## E8 fairness (N=1024, K=3, practical, 20 instances)", "",
        df.groupby("method")[["user_mean_dB", "user_min_dB", "leak_mean_dB", "time_s"]].median().round(2).to_markdown(), "",
        "paired median change vs sum-power MPR+SR (dB):", "",
        pd.DataFrame({"d_min": dmin, "d_mean": dmean}).round(2).to_markdown(), "")


# ------------------------------------------------------------------ E9 sensitivity
def do_sensitivity(df, out):
    df = gap(df, ["scenario", "seed"])
    pars = list(df.param.unique())
    fig, axes = plt.subplots(1, len(pars), figsize=(W2, 1.9), sharey=True, squeeze=False)
    for ax, par in zip(axes[0], pars):
        for sc, c in zip(df.scenario.unique(), ("#2a78d6", "#eb6834")):
            for ref, ls, mfc in ((False, "-", c), (True, "--", "white")):
                g = df[(df.param == par) & (df.scenario == sc) & (df.refine == ref)].groupby("value").gap.median()
                ax.semilogy(g.index, np.maximum(1e3 * g.values, FLOOR), ls, color=c, marker="o", mfc=mfc,
                            label=f"{sc.replace('_practical', '')}{', +SR' if ref else ''}")
        ax.set_xlabel({"lam_max": r"$\lambda_{\max}$", "iters": "$T$", "lr": "learning rate",
                       "init_spread": "init. spread"}.get(par, par))
        if par in ("lr", "lam_max", "iters"):
            ax.set_xscale("log")
            vals = sorted(df[df.param == par].value.unique())
            ax.set_xticks(vals); ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=5.5)
            ax.minorticks_off()
    axes[0][0].set_ylabel(r"median gap ($10^{-3}$)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="upper center", fontsize=6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
    save(fig, out, "sensitivity.pdf")
    G = df.groupby(["scenario", "param", "value", "refine"]).gap.median().unstack() * 1e3
    rep("## E9 sensitivity (N=1024, K=3, 10 instances): median gap x1e3", "", G.round(3).to_markdown(), "")
    num("SensSRmax", G[True].max(), "{:.2f}")
    a = G.loc["array", False]; r = G.loc["ris_practical", False]
    num("SensArrNoSRmaxExclT", a.drop(index=[("iters", 50.0)], errors="ignore").max(), "{:.2f}")
    num("SensRisIterFifty", r.loc[("iters", 50.0)], "{:.0f}"); num("SensRisIterSix", r.loc[("iters", 600.0)], "{:.2f}")
    num("SensRisDefault", r.loc[("lr", 0.05)], "{:.2f}"); num("SensRisLrHigh", r.loc[("lr", 0.2)], "{:.2f}")


# ------------------------------------------------------------------ E10 pattern
def do_pattern(df, out):
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for col, c, ls in (("continuous", "#52514e", "-"), ("CQ", "#e87ba4", ":"),
                       ("MO", "#4a3aa7", "--"), ("MPR+SR", "#2a78d6", "-")):
        if col in df:
            ax.plot(df.u, df[col], ls, color=c, lw=1.0, label=col if col != "continuous" else "continuous phases")
    ax.set_ylim(-60, 2); ax.set_xlabel("$u$ (principal cut)"); ax.set_ylabel("normalized gain (dB)")
    ax.legend(ncol=4, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "pattern.pdf")
    res = {}
    for col in ("continuous", "CQ", "MO", "MPR+SR"):
        y = df[col].values; i = int(np.nanargmax(y)); j = k = i
        while j > 0 and y[j - 1] < y[j]:
            j -= 1
        while k < len(y) - 1 and y[k + 1] < y[k]:
            k += 1
        side = np.r_[y[:j], y[k + 1:]]
        res[col] = (y[i], np.nanmax(side) - y[i])
    num("PatPeakCont", res["continuous"][0]); num("PatPeakQ", res["MPR+SR"][0])
    num("PatPSLcont", res["continuous"][1], "{:.1f}"); num("PatPSLmpr", res["MPR+SR"][1], "{:.1f}")
    num("PatPSLcq", res["CQ"][1], "{:.1f}")
    rep("## E10 pattern (32x32, 2-bit): peak (dB) / PSL in the cut (dB)", "",
        "\n".join(f"- {k}: peak {v[0]:.2f}, PSL {v[1]:.2f}" for k, v in res.items()), "")


# ------------------------------------------------------------------ E4 ablation (old campaign)
def do_ablation(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.1))
    best = df.groupby(["scenario", "K", "seed"])["y"].min()
    d = df[df.P == df.P.max()].join(best.rename("best"), on=["scenario", "K", "seed"])
    d["rel_round"] = 100 * (d.y_round - d.best) / d.best.abs()
    for sc, c in (("array", "#2a78d6"), ("ris", "#eb6834")):
        for mp, ls, mfc in (("gray", "-", c), ("binary", "--", "white")):
            e = d[(d.scenario == sc) & (d.mapping == mp)].groupby("K")["rel_round"].mean()
            axes[0].semilogy(e.index, e.clip(lower=1e-4).values, ls, color=c, marker="o",
                             mfc=mfc, label=f"{'RIS' if sc == 'ris' else 'array'}, {mp}")
    axes[0].set_xlabel("bits per element $K$"); axes[0].set_ylabel("gap after rounding (%)")
    axes[0].set_xticks(sorted(d.K.unique())); axes[0].set_ylim(5e-5, 3e3)
    axes[0].legend(fontsize=5.5, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.8)
    axes[0].set_title("(a) bit-to-state mapping", fontsize=7.5)
    g = df[df.mapping == "gray"].join(best.rename("best"), on=["scenario", "K", "seed"])
    g["rel"] = 100 * (g.y - g.best) / g.best.abs()
    for sc, c, mk in (("array", "#2a78d6", "o"), ("ris", "#eb6834", "s")):
        e = g[g.scenario == sc].groupby("P")["rel"].mean()
        axes[1].loglog(e.index, e.clip(lower=1e-4).values, "-", color=c, marker=mk,
                       label="RIS" if sc == "ris" else "array")
    axes[1].set_xlabel("parallel candidates $P$"); axes[1].set_ylabel("final gap (%)")
    axes[1].legend(fontsize=5.5); axes[1].set_title("(b) batch size", fontsize=7.5)
    for ax in axes:
        ax.axhline(1e-4, color=INK2, lw=0.6, ls=":")
    save(fig, out, "ablation.pdf")
    num("AblSeeds", f"{df.seed.nunique()}")



# ------------------------------------------------------------------ E6/E11 measured 16-state table
F_STYLE = {0: ("#2a78d6", "o", "-"), 1: ("#eb6834", "s", "--")}      # first / second frequency


def fname(b):
    """'measured-4bit-3.6GHz' -> '3.6 GHz'"""
    return b.split("-")[-1].replace("GHz", " GHz")


def do_measured4(df, dual, meta, out):
    """Measured 4-bit prototype of [zhang2025] (digitized), one codebook per frequency."""
    if meta and "table" in meta:
        fig = plt.figure(figsize=(W1, 2.6))
        ax = fig.add_subplot(111, projection="polar")
        for i, (f, T) in enumerate(sorted(meta["table"].items(), key=lambda t: float(t[0]))):
            c, mk, _ = F_STYLE.get(i, (INK2, "o", "-"))
            ax.scatter(np.deg2rad(T["phase_deg"]), 10 ** (np.array(T["amp_dB"]) / 20), s=18, color=c,
                       marker=mk, label=f"{float(f):g} GHz")
            ph = np.sort(np.array(T["phase_deg"]))
            gaps = np.diff(np.r_[ph, ph[0] + 360])
            tag = "A" if i == 0 else "B"
            num(f"Mfour{tag}Freq", f"{float(f):g}")
            num(f"Mfour{tag}AmpMin", min(T["amp_dB"]), "{:.1f}"); num(f"Mfour{tag}AmpMax", max(T["amp_dB"]), "{:.1f}")
            num(f"Mfour{tag}GapMin", gaps.min(), "{:.1f}"); num(f"Mfour{tag}GapMax", gaps.max(), "{:.1f}")
            rep(f"## E6 measured 16-state table, {float(f):g} GHz", "",
                f"- amplitudes {min(T['amp_dB']):.2f} to {max(T['amp_dB']):.2f} dB; phase gaps "
                f"{gaps.min():.1f}-{gaps.max():.1f} deg", "")
        ax.set_rmax(1.05); ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.15))
        save(fig, out, "measured4_codebook.pdf")
    if df is not None:
        df = gap(df, ["codebook", "N", "seed"])
        books = sorted(df.codebook.unique(), key=lambda b: float(fname(b).split()[0]))
        Ns = sorted(df.N.unique())
        fig, axes = plt.subplots(1, len(books), figsize=(W2 * 0.75, 2.2), sharey=True, squeeze=False)
        for ax, b in zip(axes[0], books):
            D = df[df.codebook == b]
            P = D.pivot_table(index=["N", "seed"], columns="method", values="user_mean_dB")
            ms = [m for m in ORDER if m in P.columns and m != "MPR+SR"]
            for i, m in enumerate(ms):
                c, mk, ls = style(m)
                loss = (P["MPR+SR"] - P[m]).groupby("N")
                med, lo, hi = loss.median(), loss.quantile(.25), loss.quantile(.75)
                x = np.arange(len(Ns)) + (i - (len(ms) - 1) / 2) * 0.06
                ax.errorbar(x, med.values, yerr=[med - lo, hi - med], fmt=mk, ls=ls, color=c, ms=4,
                            capsize=1.5, lw=1, label=m)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_xticks(range(len(Ns))); ax.set_xticklabels([str(n) for n in Ns])
            ax.set_xlabel("elements $N$"); ax.set_title(f"measured 4-bit cell, {fname(b)}", fontsize=7.5)
            Nm = max(Ns)
            tag = "A" if b == books[0] else "B"
            for m, key in (("ideal-design", "Ideal"), ("CQ", "CQ"), ("MO", "MO"), ("MPR", "MPR"), ("SR", "SR")):
                if m in P.columns:
                    num(f"Mfour{tag}Loss{key}", (P["MPR+SR"] - P[m]).xs(Nm, level="N").mean(),
                        "{:.3f}" if m in ("MPR", "SR") else "{:.2f}")
            G = D[D.N == Nm].groupby("method").agg(g=("gap", "mean"), t=("time_s", "mean"))
            num(f"Mfour{tag}TimeMPR", G.t["MPR"])
            rep(f"### E6 4-bit table {fname(b)}: mean over instances", "",
                D.groupby(["N", "method"])[["gap", "user_mean_dB", "user_min_dB", "leak_mean_dB", "time_s"]].mean().round(4).to_markdown(), "",
                "mean paired loss in user power vs MPR+SR (dB):", "",
                pd.DataFrame({m: (P["MPR+SR"] - P[m]).groupby("N").mean() for m in ms}).round(3).to_markdown(), "")
        axes[0][0].set_ylabel("loss in mean user power\nvs. MPR+SR (dB)")
        axes[0][0].legend(ncol=2, fontsize=5.5, loc="upper left")
        save(fig, out, "measured4_results.pdf")
    if dual is not None:
        # dual-band: bits shared by both frequencies; per-band designs evaluated in both bands
        P = dual.pivot_table(index=["N", "seed", "band"], columns="design", values="user_mean_dB")
        bands = sorted(dual.band.unique(), key=lambda b: float(b.replace("GHz", "")))
        Nm = dual.N.max()
        rows = {}
        for b in bands:
            Q = P.xs(b, level="band").xs(Nm, level="N")
            rows[b] = {d: (Q["joint MPR+SR"] - Q[d]).mean() for d in Q.columns if d != "joint MPR+SR"}
        R = pd.DataFrame(rows)
        rep(f"### E6 dual-band (N={Nm}): mean user power of the joint design minus that of each "
            "single-band design, evaluated in each band (dB; >0 = joint better)", "", R.round(2).to_markdown(), "",
            "mean user power (dB) by N, design, band:", "",
            dual.groupby(["N", "design", "band"]).user_mean_dB.mean().unstack().round(2).to_markdown(), "")
        b0, b1 = bands[0], bands[-1]
        d0, d1 = f"{b0}-only MPR+SR", f"{b1}-only MPR+SR"
        if d0 in R.index and d1 in R.index:
            num("DualOtherBandLossA", R.loc[d0, b1], "{:.1f}")    # 3.6-only design used at 4.0 GHz
            num("DualOtherBandLossB", R.loc[d1, b0], "{:.1f}")
            num("DualOwnBandCostA", -R.loc[d0, b0], "{:.2f}")      # joint vs dedicated design, same band
            num("DualOwnBandCostB", -R.loc[d1, b1], "{:.2f}")


def do_optimality4(df, out):
    df = df.copy()
    df["loss_dB"] = df.loss_dB.clip(lower=0)
    Nmax = df.N.max()
    d0 = df[df.N == Nmax]
    rep("## E11 single receiver, measured 4-bit table: loss to the exact optimum (dB)", "")
    for N in sorted(df.N.unique()):
        rep(f"### N={N} median / max", "",
            df[df.N == N].groupby(["codebook", "method"]).loss_dB.agg(["median", "max"]).unstack(0).round(4).to_markdown(), "")
    med = d0.groupby("method").loss_dB.median()
    num("OptFourMaxMPRSR", df[df.method == "MPR+SR"].loss_dB.max(), "{:.4f}")
    num("OptFourMaxMPR", d0[d0.method == "MPR"].loss_dB.max(), "{:.3f}")
    num("OptFourCQ", med["CQ"]); num("OptFourMO", med["MO"])
    num("OptFourExactMPRSR", 100 * (d0[d0.method == "MPR+SR"].loss_dB < 1e-4).mean(), "{:.0f}")
    num("OptFourCases", f"{len(df[df.method == 'MPR+SR']):,}".replace(",", "{,}"))


# ------------------------------------------------------------------ E12 BRISC measured channels
def do_brisc(fit, sc, df, out):
    """E12: measured channels (BRISC).  Powers in dB relative to the mean measured
    power of the random configurations of the same TX position."""
    if fit is not None:
        fit = fit.sort_values("pos")
        rep("## E12 BRISC: linear model with bias on held-out configurations", "",
            fit.set_index("pos").drop(columns=["antenna"]).round(3).to_markdown(), "")
        num("BriscPositions", f"{fit.pos.nunique()}")
        num("BriscNMSE", fit.nmse_dB.median(), "{:.1f}")
        num("BriscNMSEmin", fit.nmse_dB.min(), "{:.1f}"); num("BriscNMSEmax", fit.nmse_dB.max(), "{:.1f}")
        num("BriscPerrMed", fit.perr_med_dB.median(), "{:.2f}")
        num("BriscPerrMedMax", fit.perr_med_dB.max(), "{:.2f}")
        num("BriscPerrPninety", fit.perr_p90_dB.median(), "{:.1f}")
        num("BriscPerrBiasMin", -fit.perr_bias_dB.max(), "{:.2f}"); num("BriscPerrBiasMax", -fit.perr_bias_dB.min(), "{:.2f}")
        num("BriscSpearmanMin", fit.spearman.min(), "{:.2f}"); num("BriscSpearmanMax", fit.spearman.max(), "{:.2f}")
        num("BriscNtest", f"{int(fit.n_test.median()):,}".replace(",", "{,}"))
        num("BriscNtrain", f"{int(fit.n_train.median()):,}".replace(",", "{,}"))
        num("BriscFrames", f"{fit.frames.sum() / 1e6:.1f}")
        num("BriscPickTopOneN", f"{int((fit.test_pick_rank == 1).sum())}")
        num("BriscPickTopFourN", f"{int((fit.test_pick_rank <= 4).sum())}")
        pl = fit.test_best_meas_dB - fit.test_pick_meas_dB
        num("BriscPickLossMax", pl[fit.test_pick_rank <= 4].max(), "{:.1f}")
        num("BriscPickRankWorst", f"{int(fit.test_pick_rank.max())}")
        num("BriscPickLossWorst", pl.max(), "{:.1f}")
    if df is None:
        return
    ms = [m for m in ORDER if m in set(df.method)]
    W = df[df.task == "wb-max"]
    per = W[W.method != "best-measured"].groupby(["pos", "method"]).p_dB.median().unstack()
    bm = W[W.method == "best-measured"].groupby("pos")[["p_dB", "p_meas_dB"]].first()
    gain = per.sub(bm.p_meas_dB, axis=0)                     # vs best measured configuration
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1, 1.3, 1.2]))
    # (a) predicted vs measured wideband power on held-out configurations
    ax = axes[0]
    if sc is not None:
        ax.scatter(sc.meas_dB, sc.pred_dB, s=1.2, color="#2a78d6", alpha=0.2, lw=0, rasterized=True)
        lo, hi = np.nanpercentile(np.r_[sc.meas_dB, sc.pred_dB], [0.1, 99.9])
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured (dB)"); ax.set_ylabel("predicted (dB)")
    ax.set_title("(a) unseen configurations", fontsize=7.5)
    # (b) per TX position: optimized (predicted) minus best measured configuration
    ax = axes[1]
    mb = [m for m in ms if m in gain.columns]
    for i, m in enumerate(mb):
        c, mk, ls = style(m)
        x = np.arange(len(gain)) + (i - (len(mb) - 1) / 2) * 0.09
        ax.plot(x, gain[m].values, mk, color=c, ms=3.5, label=m, ls="none")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(gain))); ax.set_xticklabels([str(p) for p in gain.index])
    ax.set_xlabel("TX position"); ax.set_ylabel("gain over best measured\nconfiguration (dB)")
    ax.set_title("(b) wideband power, 242 subcarriers", fontsize=7.5)
    ax.legend(ncol=2, fontsize=5, loc="upper right", handletextpad=0.2, columnspacing=0.6)
    # (c) single subcarrier: loss to the exact optimum of Lemma 1
    ax = axes[2]
    S = df[df.task == "sc-max"].copy()
    S["loss_dB"] = S.loss_dB.clip(lower=0)
    mm = [m for m in ms if m in set(S.method)] + ["best-measured"]
    for i, m in enumerate(mm):
        c, mk, ls = style(m) if m != "best-measured" else (INK, "s", "--")
        v = S[S.method == m].loss_dB
        med, lo_, hi_ = v.median(), v.quantile(.25), v.quantile(.75)
        ax.errorbar(i, med, yerr=[[med - lo_], [hi_ - med]], fmt=mk, color=c, ms=4, capsize=1.5, lw=1)
        ax.plot(i, v.max(), mk, mfc="none", color=c, ms=4)
    ax.set_xticks(range(len(mm))); ax.set_xticklabels(mm, rotation=60, fontsize=5.5, ha="right")
    ax.set_yscale("symlog", linthresh=1e-3, linscale=0.6); ax.set_ylim(-1e-4, 30)
    ax.set_ylabel("loss to global optimum (dB)")
    ax.set_title("(c) one subcarrier", fontsize=7.5)
    save(fig, out, "brisc.pdf")

    rep("### E12 wb-max per TX position: predicted power (dB, median over seeds) and best measured configuration", "",
        per.join(bm.rename(columns={"p_dB": "best-meas (pred)", "p_meas_dB": "best-meas (meas)"})).round(2).to_markdown(), "",
        "gain of each method over the best measured configuration (dB):", "", gain.round(2).to_markdown(), "")
    g = gain["MPR+SR"]
    num("BriscMaxGainMed", g.median(), "{:.1f}"); num("BriscMaxGainMin", g.min(), "{:.1f}")
    num("BriscMaxGainMax", g.max(), "{:.1f}"); num("BriscMaxGainBigN", f"{int((g > 5).sum())}")
    num("BriscMaxPredMin", per["MPR+SR"].min(), "{:.1f}"); num("BriscMaxPredMax", per["MPR+SR"].max(), "{:.1f}")
    num("BriscBestMeasMin", bm.p_meas_dB.min(), "{:.1f}"); num("BriscBestMeasMax", bm.p_meas_dB.max(), "{:.1f}")
    num("BriscBestPredErr", (bm.p_dB - bm.p_meas_dB).abs().median(), "{:.1f}")
    d = per["SR"] - per["MPR+SR"]                               # >0: SR better
    num("BriscMaxTieN", f"{int((d.abs() < 0.01).sum())}")
    num("BriscMaxSRbetter", d.max(), "{:.2f}"); num("BriscMaxSRbetterPos", f"{int(d.idxmax())}")
    v = per["MPR+SR"] - bm.p_dB                                 # vs model value of the best measured config
    num("BriscMaxVsBestPredMin", v.min(), "{:.2f}"); num("BriscMaxVsBestPredPos", f"{int(v.idxmin())}")
    for m, key in (("MPR", "MPR"), ("CQ", "CQ"), ("MO", "MO"), ("GA", "GA")):
        if m in per:
            num(f"BriscMaxLoss{key}", (per["MPR+SR"] - per[m]).median(), "{:.2f}")
            num(f"BriscMaxLoss{key}Max", (per["MPR+SR"] - per[m]).max(), "{:.2f}")
    for m, key in (("MPR", "MPR"), ("MPR+SR", "MPRSR"), ("SR", "SR"), ("GA", "GA"), ("MO", "MO"), ("CQ", "CQ")):
        num(f"BriscTime{key}", W[W.method == m].time_s.mean(), "{:.2f}")

    num("BriscScCases", f"{S[S.method == 'exact-opt'].groupby(['pos', 'sub']).ngroups}")
    num("BriscScSeeds", f"{S[S.method == 'MPR+SR'].seed.nunique()}")
    for m, key in (("MPR+SR", "MPRSR"), ("SR", "SR"), ("MPR", "MPR"), ("GA", "GA"), ("CQ+SR", "CQSR"),
                   ("CQ", "CQ"), ("MO", "MO")):
        v = S[S.method == m].loss_dB
        num(f"BriscSc{key}Exact", 100 * (v < 1e-4).mean(), "{:.0f}")
        num(f"BriscSc{key}Max", v.max(), "{:.2f}")
        num(f"BriscSc{key}Med", v.median(), "{:.3f}")
    num("BriscScBestMeas", S[S.method == "best-measured"].loss_dB.median(), "{:.1f}")
    if "MPR+SR-dedup" in set(S.method):
        v = S[S.method == "MPR+SR-dedup"].loss_dB
        num("BriscScDedupExact", 100 * (v < 1e-4).mean(), "{:.0f}"); num("BriscScDedupMax", v.max(), "{:.2f}")
        hard = S[S.pos.isin(S[(S.method == "MPR+SR") & (S.loss_dB > 0.01)].pos.unique())]
        for m, key in (("MPR+SR", "MPRSR"), ("MPR+SR-dedup", "Dedup"), ("SR", "SR")):
            num(f"BriscScHard{key}Exact", 100 * (hard[hard.method == m].loss_dB < 1e-4).mean(), "{:.0f}")
        num("BriscScHardPos", ", ".join(str(p) for p in sorted(hard.pos.unique())))
        for m, key in (("MPR+SR", "MPRSR"), ("MPR+SR-dedup", "Dedup"), ("SR", "SR")):
            v = S[S.method == m].loss_dB
            num(f"BriscScMiss{key}", f"{int((v > 0.01).sum())}")
            num(f"BriscScMean{key}", v.mean(), "{:.4f}")
        num("BriscScRuns", f"{len(S[S.method == 'MPR+SR'])}")
        dmiss = S[(S.method == "MPR+SR-dedup") & (S.loss_dB > 0.01)]
        num("BriscScDedupMissPos", ", ".join(str(p) for p in sorted(dmiss.pos.unique())) or "--")
        num("BriscTimeDedup", W[W.method == "MPR+SR-dedup"].time_s.mean(), "{:.2f}")
        dd = per["SR"] - per["MPR+SR-dedup"]
        num("BriscMaxDedupVsSR", dd.max(), "{:.2f}"); num("BriscMaxDedupTieN", f"{int((dd.abs() < 0.01).sum())}")
        num("BriscMaxDedupGainMax", (per["MPR+SR-dedup"] - per["MPR+SR"]).max(), "{:.2f}")
        sp = W[W.method == "MPR+SR-dedup"].groupby("pos").p_dB.agg(lambda v: v.max() - v.min())
        num("BriscMaxDedupSpread", sp.max(), "{:.2f}")
        nd = W[W.method == "MPR+SR-dedup"]
        if "n_dup" in nd:
            num("BriscDupMed", f"{int(nd.n_dup.median())}")
    rep("### E12 sc-max: loss to the exact optimum (dB) over positions x subcarriers x seeds", "",
        S[S.method != "exact-opt"].groupby("method").loss_dB.describe(percentiles=[.5, .9]).round(4).to_markdown(), "",
        "MPR+SR cases with loss > 0.01 dB:", "",
        S[(S.method == "MPR+SR") & (S.loss_dB > 0.01)][["pos", "sub", "seed", "loss_dB"]].round(3).to_markdown(index=False), "")

    M = df[df.task == "wb-min"]
    if not M.empty:
        pm = M[M.method != "best-measured"].groupby(["pos", "method"]).p_dB.median().unstack()
        bmin = M[M.method == "best-measured"].groupby("pos")[["p_dB", "p_meas_dB"]].first()
        rep("### E12 wb-min (null towards the receiver) per position -- model accuracy limits this task", "",
            pm.join(bmin.rename(columns={"p_dB": "best-meas (pred)", "p_meas_dB": "best-meas (meas)"})).round(2).to_markdown(), "")
        num("BriscMinDeeperN", f"{int((bmin.p_meas_dB < pm.min(axis=1)).sum())}")
        num("BriscMinPredErr", (bmin.p_dB - bmin.p_meas_dB).median(), "{:.1f}")
        num("BriscMinPredErrMax", (bmin.p_dB - bmin.p_meas_dB).max(), "{:.1f}")
    Pp = df[df.task == "pair"]
    if not Pp.empty:
        X = Pp[Pp.method == "MPR+SR"].groupby(["pos", "null_pos"])[["p_dB", "p2_dB"]].median()
        B = Pp[Pp.method == "best-measured"].groupby(["pos", "null_pos"])[["p_dB", "p2_dB", "p_meas_dB"]].first()
        rep("### E12 pair (serve pos, suppress null_pos): MPR+SR (predicted) vs best measured configuration", "",
            X.join(B, rsuffix=" best-meas").round(2).to_markdown(), "")
        num("BriscPairServe", (X.p_dB - B.p_dB).median(), "{:.1f}")
        num("BriscPairs", f"{len(X)}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES_DEFAULT,
                    help="results folder, or several separated by commas (files are merged)")
    ap.add_argument("--out", default=PAPER_DEFAULT)
    ap.add_argument("--summary", default=None, help="where to write results_summary.md (default: first --res)")
    a = ap.parse_args()
    res = [os.path.abspath(r) for r in a.res.split(",")]
    for sub in ("figs", "tables"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    print("results:", ", ".join(res))
    rep("# Results summary (generated by experiments/make_figures.py)", "",
        "source: " + ", ".join(f"`{os.path.basename(r)}`" for r in res), "")
    for name, fn in (("e1b_scaling", do_scaling), ("e11_optimality", do_optimality),
                     ("e5_anytime", do_anytime), ("e7_variation", do_variation),
                     ("e8_fairness", do_fairness), ("e9_sensitivity", do_sensitivity),
                     ("e10_pattern", do_pattern)):
        df = load(res, name)
        if df is not None:
            fn(df, a.out)
    e2, e3 = load(res, "e2_array"), load(res, "e3_ris")
    if e2 is not None or e3 is not None:
        do_main(e2, e3, a.out)
        if e2 is not None:
            num("NInst", f"{e2.seed.nunique()}")
    mp = find(res, "e6_codebooks.json")
    do_measured(load(res, "e6_measured"), json.load(open(mp)) if mp else None, a.out)
    mp4 = find(res, "e6_codebooks_table.json")
    m4, d4 = load(res, "e6_measured_table"), load(res, "e6_dualband_table")
    if m4 is not None or d4 is not None or mp4:
        do_measured4(m4, d4, json.load(open(mp4)) if mp4 else None, a.out)
    o4 = load(res, "e11_measured4")
    if o4 is not None:
        do_optimality4(o4, a.out)
    bf, bs, bm = load(res, "e12_brisc_fit"), load(res, "e12_brisc_scatter"), load(res, "e12_brisc")
    if bf is not None or bm is not None:
        do_brisc(bf, bs, bm, a.out)
    e4 = load(res, "e4_ablation")
    if e4 is None:
        e4 = load(RES_OLD, "e4_ablation")
    if e4 is not None:
        do_ablation(e4, a.out)
    with open(os.path.join(a.out, "tables", "numbers.tex"), "w") as f:
        f.write("% generated by experiments/make_figures.py -- do not edit by hand\n")
        for k, v in sorted(NUM.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("  numbers.tex:", len(NUM), "macros")
    sp = a.summary or os.path.join(res[0], "results_summary.md")
    with open(sp, "w") as f:
        f.write("\n".join(REPORT) + "\n")
    print("  summary:", sp)


if __name__ == "__main__":
    main()
