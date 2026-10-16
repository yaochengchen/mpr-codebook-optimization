"""E5 (IEEE Access): quality versus wall-clock budget ("anytime" comparison).
Each method is run with a ladder of budgets (iterations / sweeps / generations)
on the SAME instances; every (time, objective) pair is recorded."""
from common import *

LADDER = {
    "MPR":    [("iters", t) for t in (10, 25, 50, 100, 200, 300, 600)],
    "MPR+SR": [("sweeps", k) for k in (1, 2, 3, 5, 10)],
    "SR":     [("sweeps", k) for k in (1, 2, 3, 5, 10, 20)],
    "CQ":     [("iters", t) for t in (25, 50, 100, 300, 600)],
    "MO":     [("iters", t) for t in (10, 25, 50, 100, 300)],
    "MO+SR":  [("sweeps", k) for k in (1, 3, 10)],
    "GA":     [("gens", g) for g in (50, 100, 200, 500, 1000, 2000)],
}


def run(obj, m, kind, val, P, seed):
    ph = obj.V.angle().to(obj.rdtype)
    if m == "MPR":
        return mpr(obj, P=P, iters=val, refine_sweeps=0, seed=seed)
    if m == "MPR+SR":
        return mpr(obj, P=P, iters=100, refine_sweeps=val, seed=seed)
    if m == "SR":
        return random_sr(obj, P=P, sweeps=val, seed=seed)
    if m == "CQ":
        return cont_quantize(obj, ph, P=P, iters=val, seed=seed)
    if m == "MO":
        return manifold_cq(obj, ph, P=P, iters=val, seed=seed)
    if m == "MO+SR":
        return manifold_cq(obj, ph, P=P, iters=100, refine_sweeps=val, seed=seed)
    if m == "GA":
        return genetic(obj, pop=max(P, 128), gens=val, seed=seed)
    raise ValueError(m)


def main():
    p = base_parser(__doc__)
    p.add_argument("--Nx", type=int, default=45)
    p.add_argument("--K", type=int, default=3)
    args = p.parse_args()
    Nx = 10 if args.quick else args.Nx
    seeds = 1 if args.quick else args.seeds
    out = CSV("e5_anytime", args)
    for seed in range(args.seed0, args.seed0 + seeds):
        a, B, info = array_problem(Nx, Nx, M=150 if args.quick else 500, main_dir=(0.3, 0.2),
                                   side_weight=100.0, seed=seed)
        a2, B2, d2, info2 = ris_problem(Nx, Nx, n_freq=4 if args.quick else 16, leak_weight=4.0, seed=seed)
        for scen, obj in (("array", make_obj(a, B, ideal_phase_codebook(args.K), args=args)),
                          ("ris_practical", make_obj(a2, B2, practical_ris_codebook(args.K), d=d2, args=args))):
            yr = y_random(obj)
            for m, ladder in LADDER.items():
                if args.methods and m not in args.methods.split(","):
                    continue
                for kind, val in (ladder[:2] if args.quick else ladder):
                    r = run(obj, m, kind, 3 if (args.quick and kind == "iters") else val, args.P, seed)
                    row = dict(scenario=scen, N=obj.N, K=args.K, seed=seed, method=m,
                               budget_kind=kind, budget=val, y=r["y"], y_rand=yr, time_s=r["time"])
                    print(row); out.write(row)
    out.close()


if __name__ == "__main__":
    main()
