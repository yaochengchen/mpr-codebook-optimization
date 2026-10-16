"""E4: ablations of the proposed relaxation on the largest instances.
 (a) Gray vs natural-binary bit-to-state mapping (affects the relaxation only;
     exact refinement acts on states and is mapping independent)
 (b) number of parallel candidates P (GPU batching) vs quality and time
 (c) convergence trace of the relaxed objective."""
import json
from common import *


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    Nx = 12 if args.quick else 45
    Ks = [2, 3] if args.quick else [2, 3, 4, 5]
    iters, sweeps = (100, 3) if args.quick else (300, 10)
    out = CSV("e4_ablation", args)
    traces = {}
    for seed in range(args.seed0, args.seed0 + args.seeds):
        a, B, info = array_problem(Nx, Nx, M=150 if args.quick else 500,
                                   main_dir=(0.3, 0.2), side_weight=100.0, seed=seed)
        a2, B2, d2, info2 = ris_problem(Nx, Nx, n_freq=4 if args.quick else 16,
                                        leak_weight=4.0, seed=seed)
        for scen, (aa, BB, dd), cbf in (("array", (a, B, None), ideal_phase_codebook),
                                       ("ris", (a2, B2, d2), practical_ris_codebook)):
            for K in Ks:
                for mapping in ("gray", "binary"):
                    obj = make_obj(aa, BB, cbf(K), mapping=mapping, d=dd, args=args)
                    for P in ([4, 16] if args.quick else [1, 16, 64, 256]):
                        r = mpr(obj, P=P, iters=iters, refine_sweeps=sweeps, seed=seed,
                                return_trace=(P == args.P or P == 16) and seed == 0)
                        row = dict(scenario=scen, N=obj.N, K=K, mapping=mapping, P=P,
                                   seed=seed, y_round=r["y_round"], y=r["y"],
                                   t_relax=r["t_relax"], time_s=r["time"], sweeps=r["sweeps"])
                        print(row); out.write(row)
                        if r["trace"]:
                            traces[f"{scen}_K{K}_{mapping}_P{P}"] = r["trace"]
    out.close()
    with open(os.path.join(RESULTS, f"e4_traces{('_' + args.tag) if args.tag else ''}.json"), "w") as f:
        json.dump(traces, f)


if __name__ == "__main__":
    main()
