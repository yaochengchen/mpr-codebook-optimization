"""E9 (IEEE Access): hyper-parameter sensitivity of MPR (one factor at a time
around the defaults lr=0.05, lambda_max=2, T=300, init_spread=1)."""
from common import *

GRID = {"lr": [0.01, 0.02, 0.05, 0.1, 0.2],
        "lam_max": [0.5, 1.0, 2.0, 5.0, 10.0],
        "iters": [50, 100, 200, 300, 600],
        "init_spread": [0.1, 0.5, 1.0]}
DEFAULT = dict(lr=0.05, lam_max=2.0, iters=300, init_spread=1.0)


def main():
    p = base_parser(__doc__)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--Nx", type=int, default=32)
    args = p.parse_args()
    Nx = 8 if args.quick else args.Nx
    seeds = 1 if args.quick else args.seeds
    out = CSV("e9_sensitivity", args)
    for seed in range(args.seed0, args.seed0 + seeds):
        a, B, info = array_problem(Nx, Nx, M=150 if args.quick else 500, main_dir=(0.3, 0.2),
                                   side_weight=100.0, seed=seed)
        a2, B2, d2, _ = ris_problem(Nx, Nx, n_freq=4 if args.quick else 16, leak_weight=4.0, seed=seed)
        for scen, obj in (("array", make_obj(a, B, ideal_phase_codebook(args.K), args=args)),
                          ("ris_practical", make_obj(a2, B2, practical_ris_codebook(args.K), d=d2, args=args))):
            yr = y_random(obj)
            for par, vals in GRID.items():
                for v in (vals[:2] if args.quick else vals):
                    kw = dict(DEFAULT); kw[par] = v
                    if args.quick:
                        kw["iters"] = 20
                    for sw in (0, 10):
                        r = mpr(obj, P=args.P, refine_sweeps=sw if not args.quick else min(sw, 1),
                                seed=seed, **kw)
                        row = dict(scenario=scen, N=obj.N, K=args.K, seed=seed, param=par, value=v,
                                   refine=sw > 0, y=r["y"], y_round=r["y_round"], y_rand=yr,
                                   time_s=r["time"])
                        print(row); out.write(row)
    out.close()


if __name__ == "__main__":
    main()
