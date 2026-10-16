"""E12 (IEEE Access): measurement-calibrated RIS channels (BRISC dataset).

For each TX position the per-configuration mean CSI (242 subcarriers) is fitted
by the linear model with bias of the BRISC paper, h_f(x) = d_f + sum_j beta_fj x_j,
which is the field model (1) with a 1-bit codebook {+1,-1} (mpr/brisc.py).

1) Model check (e12_brisc_fit.csv, e12_brisc_scatter.csv):
   fit on the block configurations + 80% of the random ones, test on the
   remaining 20% (unseen configurations): NMSE of the CSI, error of the
   wideband power in dB, rank correlation; and a purely measured check:
   among the held-out configurations, the one the model ranks first is
   compared with the measured best.
2) Optimisation on the model fitted to ALL measured configurations
   (e12_brisc.csv), powers in dB relative to the mean measured power of the
   random configurations, "best-measured" = best of the ~10 000 measured ones:
     wb-max : maximise the mean power over the 242 subcarriers (M = 242)
     wb-min : minimise it (null towards the receiver)
     sc-max : one subcarrier (M = 1); the exact optimum of Lemma 1 is available,
              so every method's loss to the global optimum is reported
   Methods include MPR+SR-dedup: MPR+SR whose near-duplicate candidates are
   replaced by random configurations before the refinement (optimize.diversify).
     pair   : (only with >= 2 positions) serve TX position i, suppress j
              (rows of both positions, a = -1/M for i, +w/M for j, w = --null_weight;
               each position normalised by its own reference power)

    python e12_brisc.py --data ../data/brisc --positions 1,2,3   # GPU/CPU
    python e12_brisc.py --quick                                    # synthetic smoke test
"""
import itertools
import math
import numpy as np
from common import *
from mpr import brisc


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def build(a_np, B_np, d_np, args):
    cdt, rdt = dtypes(args)
    a = torch.tensor(a_np, dtype=torch.float64)
    B = torch.tensor(B_np, dtype=torch.complex128)
    d = torch.tensor(d_np, dtype=torch.complex128)
    return make_obj(a, B, ideal_phase_codebook(1), d=d, args=args)


def main():
    p = base_parser(__doc__)
    p.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data", "brisc"))
    p.add_argument("--antenna", default="log", choices=["log", "dipole"])
    p.add_argument("--positions", default=None, help="comma list (default: all available)")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--lam", type=float, default=1e-4, help="ridge weight (relative)")
    p.add_argument("--subcarriers", default="0,30,60,90,121,150,180,210,241")
    p.add_argument("--pairs", default="auto", help="'auto' = all ordered pairs of --positions, 'none', or e.g. 1-2,2-1")
    p.add_argument("--max_pairs", type=int, default=12)
    p.add_argument("--null_weight", type=float, default=4.0, help="weight of the suppressed position (as leak_weight in e3/e6)")
    p.add_argument("--only_pairs", action="store_true", help="skip the per-position tasks (pairs only)")
    p.add_argument("--synthetic", action="store_true", help="use a synthetic BRISC-like dataset")
    args = p.parse_args()
    if args.quick:
        args.synthetic = True
    seeds = 1 if args.quick else args.seeds
    it, sw, gens = (60, 2, 40) if args.quick else (300, 10, 500)
    methods = ["CQ", "CQ+SR", "MO", "MO+SR", "GA", "SR", "MPR", "MPR+SR", "MPR+SR-dedup"]
    scs = [int(v) for v in args.subcarriers.split(",")]

    # ---------------------------------------------------------------- data
    data = {}
    if args.synthetic:
        positions = [1, 2] if args.positions is None else [int(v) for v in args.positions.split(",")]
        for k in positions:
            bits, H, _, _ = brisc.synthetic(n_cfg=1500 if args.quick else 10000, seed=k)
            data[k] = (bits, H, np.ones(len(bits), int))
        ant = "synthetic"
    else:
        bp = os.path.join(args.data, "configurations_bits.npy")
        bits_all = np.load(bp) if os.path.exists(bp) else \
            brisc.load_configurations(os.path.join(args.data, "configurations_10000.mat"))
        positions = brisc.available_positions(args.data, args.antenna) if args.positions is None \
            else [int(v) for v in args.positions.split(",")]
        if not positions:
            print(f"no BRISC files in {args.data} (see data/brisc/README.md)"); return
        for k in positions:
            r = brisc.load_position(args.data, args.antenna, k, bits_all)
            data[k] = (bits_all, r["H"], r["count"])
        ant = args.antenna
    scs = [s for s in scs if s < next(iter(data.values()))[1].shape[1]]

    if not args.only_pairs:
        fit_out = CSV("e12_brisc_fit", args)
        sc_out = CSV("e12_brisc_scatter", args)
    out = CSV("e12_brisc", args)
    models = {}
    for k in positions:
        bits, H, cnt = data[k]
        H = H.astype(np.complex128)
        have = np.flatnonzero(cnt > 0)
        rnd = have[have >= brisc.N_BLOCK_CFG]
        blk = have[have < brisc.N_BLOCK_CFG]
        g = np.random.default_rng(1234 + k)
        perm = g.permutation(rnd)
        n_te = int(round(args.test_frac * len(perm)))
        te, tr = perm[:n_te], np.r_[blk, perm[n_te:]]
        P_meas = brisc.wideband_power(H)
        P_ref = P_meas[rnd].mean()                     # 0 dB = mean over random configurations

        if args.only_pairs:
            d, beta = brisc.fit_lmb(H, bits, have, args.lam)
            dp, Bp = brisc.to_pm1(d, beta)
            models[k] = (dp, Bp, P_ref, P_meas, have, H)
            continue
        # ------------------------------------------------ 1) model check
        d0, beta0 = brisc.fit_lmb(H, bits, tr, args.lam)
        Hte = brisc.predict(d0, beta0, bits[te])
        P_te = brisc.wideband_power(Hte)
        err = db(P_te) - db(P_meas[te])
        from scipy.stats import spearmanr
        rho = spearmanr(P_te, P_meas[te]).correlation
        pick = te[np.argmax(P_te)]
        rank = int((P_meas[te] > P_meas[pick]).sum()) + 1
        pick_lo = te[np.argmin(P_te)]
        rank_lo = int((P_meas[te] < P_meas[pick_lo]).sum()) + 1
        row = dict(pos=k, antenna=ant, n_cfg=len(have), n_train=len(tr), n_test=len(te),
                   frames=int(cnt.sum()), nmse_dB=brisc.nmse_db(Hte, H[te]),
                   nmse_train_dB=brisc.nmse_db(brisc.predict(d0, beta0, bits[tr]), H[tr]),
                   perr_med_dB=float(np.median(np.abs(err))), perr_p90_dB=float(np.quantile(np.abs(err), .9)),
                   perr_bias_dB=float(np.mean(err)), spearman=rho,
                   test_best_meas_dB=db(P_meas[te].max() / P_ref), test_pick_meas_dB=db(P_meas[pick] / P_ref),
                   test_pick_rank=rank,
                   test_worst_meas_dB=db(P_meas[te].min() / P_ref), test_pickmin_meas_dB=db(P_meas[pick_lo] / P_ref),
                   test_pickmin_rank=rank_lo,
                   meas_max_dB=db(P_meas[have].max() / P_ref), meas_min_dB=db(P_meas[have].min() / P_ref))
        print(row); fit_out.write(row)
        for i, c in enumerate(te):
            sc_out.write(dict(pos=k, antenna=ant, cfg=int(c), meas_dB=db(P_meas[c] / P_ref),
                              pred_dB=db(P_te[i] / P_ref)))

        # ------------------------------------------------ 2) optimisation on the full fit
        d, beta = brisc.fit_lmb(H, bits, have, args.lam)
        dp, Bp = brisc.to_pm1(d, beta)
        models[k] = (dp, Bp, P_ref, P_meas, have, H)
        M = Bp.shape[0]
        Ppred_all = brisc.wideband_power(brisc.predict(d, beta, bits[have]))
        for task, sign in (("wb-max", -1.0), ("wb-min", +1.0)):
            obj = build(np.full(M, sign / M), Bp, dp, args)
            yr = y_random(obj)
            i_best = have[np.argmax(P_meas[have])] if sign < 0 else have[np.argmin(P_meas[have])]
            p_bm_pred = brisc.wideband_power(brisc.predict(d, beta, bits[i_best:i_best + 1]))[0]
            base = dict(task=task, pos=k, antenna=ant, sub=-1, null_pos=-1, y_rand=yr)
            out.write(dict(base, seed=-1, method="best-measured", y=sign * p_bm_pred,
                           p_dB=db(p_bm_pred / P_ref), p2_dB=np.nan, p_meas_dB=db(P_meas[i_best] / P_ref),
                           loss_dB=np.nan, time_s=np.nan, n_dup=-1))
            for seed in range(args.seed0, args.seed0 + seeds):
                for r in run_methods(obj, args, methods=methods, iters=it, sweeps=sw, ga_gens=gens, seed=seed):
                    pw = sign * r["y"]
                    row = dict(base, seed=seed, method=r["name"], y=r["y"], p_dB=db(pw / P_ref), p2_dB=np.nan,
                               p_meas_dB=np.nan, loss_dB=np.nan, time_s=r["time"], n_dup=r.get("n_dup", -1))
                    print(row); out.write(row)
        # single subcarrier: exact optimum available
        for f in scs:
            obj = build(np.array([-1.0]), Bp[f:f + 1], dp[f:f + 1], args)
            yr = y_random(obj)
            p_opt, _ = single_user_optimum(obj.B[0].to(torch.complex128), obj.V.to(torch.complex128),
                                           obj.d[0].to(torch.complex128))
            Pf_ref = np.mean(np.abs(H[rnd, f]) ** 2)
            i_best = have[np.argmax(np.abs(H[have, f]))]
            p_bm = np.abs(brisc.predict(d[f:f + 1], beta[f:f + 1], bits[i_best:i_best + 1])[0, 0]) ** 2
            base = dict(task="sc-max", pos=k, antenna=ant, sub=f, null_pos=-1, y_rand=yr)
            out.write(dict(base, seed=-1, method="exact-opt", y=-p_opt, p_dB=db(p_opt / Pf_ref), p2_dB=np.nan,
                           p_meas_dB=np.nan, loss_dB=0.0, time_s=np.nan, n_dup=-1))
            out.write(dict(base, seed=-1, method="best-measured", y=-p_bm, p_dB=db(p_bm / Pf_ref), p2_dB=np.nan,
                           p_meas_dB=db(np.abs(H[i_best, f]) ** 2 / Pf_ref), loss_dB=db(p_opt / p_bm), time_s=np.nan, n_dup=-1))
            for seed in range(args.seed0, args.seed0 + seeds):
                for r in run_methods(obj, args, methods=methods, iters=it, sweeps=sw, ga_gens=gens, seed=seed):
                    row = dict(base, seed=seed, method=r["name"], y=r["y"], p_dB=db(-r["y"] / Pf_ref), p2_dB=np.nan,
                               p_meas_dB=np.nan, loss_dB=db(p_opt / max(-r["y"], 1e-30)), time_s=r["time"], n_dup=r.get("n_dup", -1))
                    print(row); out.write(row)

    # ---------------------------------------------------- 3) serve i, suppress j
    if args.pairs == "none" or len(positions) < 2:
        pairs = []
    elif args.pairs == "auto":
        pairs = list(itertools.permutations(positions, 2))[: args.max_pairs]
    else:
        pairs = [tuple(int(v) for v in s.split("-")) for s in args.pairs.split(",")]
    for (i, j) in pairs:
        di, Bi, Pri, Pmi, hi, Hi = models[i]
        dj, Bj, Prj, Pmj, hj, Hj = models[j]
        M = Bi.shape[0]
        # normalise each position by its reference power so both rows count equally
        w = args.null_weight
        a = np.r_[np.full(M, -1.0 / (M * Pri)), np.full(M, w / (M * Prj))]
        obj = build(a, np.r_[Bi, Bj], np.r_[di, dj], args)
        yr = y_random(obj)
        common = np.intersect1d(hi, hj)
        contrast = db(Pmi[common] / Pri) - db(Pmj[common] / Prj)
        meas_obj = -Pmi[common] / Pri + w * Pmj[common] / Prj
        cb = common[np.argmin(meas_obj)]
        base = dict(task="pair", pos=i, antenna=ant, sub=-1, null_pos=j, y_rand=yr)
        out.write(dict(base, seed=-1, method="best-measured", y=float(meas_obj.min()),
                       p_dB=db(Pmi[cb] / Pri), p2_dB=db(Pmj[cb] / Prj), p_meas_dB=float(contrast.max()),
                       loss_dB=np.nan, time_s=np.nan, n_dup=-1))
        for seed in range(args.seed0, args.seed0 + seeds):
            for r in run_methods(obj, args, methods=methods, iters=it, sweeps=sw, ga_gens=gens, seed=seed):
                z = obj.field(obj.X_states(r["s"][None]))[0].to(torch.complex128).cpu().numpy()
                pi_, pj_ = np.mean(np.abs(z[:M]) ** 2), np.mean(np.abs(z[M:]) ** 2)
                row = dict(base, seed=seed, method=r["name"], y=r["y"], p_dB=db(pi_ / Pri), p2_dB=db(pj_ / Prj),
                           p_meas_dB=np.nan, loss_dB=np.nan, time_s=r["time"], n_dup=r.get("n_dup", -1))
                print(row); out.write(row)
    out.close()
    if not args.only_pairs:
        fit_out.close(); sc_out.close()


if __name__ == "__main__":
    main()
