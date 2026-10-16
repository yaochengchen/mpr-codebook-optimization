"""E7 (IEEE Access): element-dependent codebooks (fabrication / bias spread).
Each element j gets its own table V_j = nominal * (1 + amp err) * exp(i phase err).
Exact-codebook methods use V_j; 'nominal-design' optimises with the nominal
table and deploys the same state indices on the perturbed hardware."""
from common import *


def main():
    p = base_parser(__doc__)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--phase_std", default="0,5,10,20,30")
    p.add_argument("--amp_std", default="0,0.1,0.2")
    p.add_argument("--Nx", type=int, default=32)
    args = p.parse_args()
    Nx = 8 if args.quick else args.Nx
    seeds = 1 if args.quick else args.seeds
    it, sw = (50, 2) if args.quick else (300, 10)
    phs = [0, 10] if args.quick else [float(v) for v in args.phase_std.split(",")]
    ams = [0.1] if args.quick else [float(v) for v in args.amp_std.split(",")]
    out = CSV("e7_variation", args)
    for seed in range(args.seed0, args.seed0 + seeds):
        scen = {
            "array": (*array_problem(Nx, Nx, M=150 if args.quick else 500, main_dir=(0.3, 0.2),
                                     side_weight=100.0, seed=seed)[:2], None, ideal_phase_codebook(args.K)),
            "ris": (*ris_problem(Nx, Nx, n_freq=4 if args.quick else 16, leak_weight=4.0, seed=seed)[:3],
                    practical_ris_codebook(args.K)),
        }
        for name, (a, B, d, nominal) in scen.items():
            obj_nom = make_obj(a, B, nominal, d=d, args=args)
            r_nom = mpr(obj_nom, P=args.P, iters=it, refine_sweeps=sw, seed=seed)
            for sp in phs:
                for sa in ams:
                    # per-element tables in phase order -> bit-pattern table via Gray map
                    cbN = per_element_variation(nominal, obj_nom.N, sp, sa, seed=1000 + seed)
                    obj = make_obj(a, B, cbN, d=d, args=args)
                    yr = y_random(obj)
                    rows = run_methods(obj, args, methods=["CQ", "MO", "SR", "MPR", "MPR+SR"],
                                       iters=it, sweeps=sw, seed=seed)
                    rows.append(dict(name="nominal-design", y=obj.value_states(r_nom["s"][None])[0].item(),
                                     time=r_nom["time"]))
                    for r in rows:
                        row = dict(scenario=name, N=obj.N, K=args.K, seed=seed, phase_std=sp,
                                   amp_std=sa, method=r["name"], y=r["y"], y_rand=yr, time_s=r["time"])
                        print(row); out.write(row)
    out.close()


if __name__ == "__main__":
    main()
