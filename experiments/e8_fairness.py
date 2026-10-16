"""E8 (IEEE Access): max-min fairness through a smooth soft-min surrogate.
Sum-power objective (paper eq. (1)) vs soft-min of the per-user powers with a
leakage penalty; the soft-min only changes the row weights in the closed-form
gradient (fields.SoftMinPower).  Reports mean / minimum user power and leakage."""
from common import *


def main():
    p = base_parser(__doc__)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--taus", default="0.3,0.1,0.03,0.01")
    p.add_argument("--Nx", type=int, default=32)
    p.add_argument("--codebook", default="practical", choices=["ideal", "practical"])
    args = p.parse_args()
    Nx = 8 if args.quick else args.Nx
    seeds = 1 if args.quick else args.seeds
    it, sw = (50, 2) if args.quick else (300, 10)
    taus = [0.1] if args.quick else [float(t) for t in args.taus.split(",")]
    cb = practical_ris_codebook(args.K) if args.codebook == "practical" else ideal_phase_codebook(args.K)
    cdt, rdt = dtypes(args)
    out = CSV("e8_fairness", args)
    for seed in range(args.seed0, args.seed0 + seeds):
        a, B, d, info = ris_problem(Nx, Nx, n_freq=4 if args.quick else 16, leak_weight=4.0, seed=seed)
        obj = make_obj(a, B, cb, d=d, args=args)
        results = {"sum-power MPR+SR": mpr(obj, P=args.P, iters=it, refine_sweeps=sw, seed=seed)}
        N = obj.N
        # power scale = mean user power of the sum-power design, so tau is relative
        z0 = obj.field(obj.X_states(results["sum-power MPR+SR"]["s"]))
        um = info["user_mask"].to(z0.device)
        scale = (z0.abs() ** 2)[um].mean().item()
        for tau in taus:
            sm = SoftMinPower([info["user_idx"].to(args.device)], info["n_users"], tau=tau,
                              scale=scale, leak_weight=4.0)
            fo = FieldObjective([obj.B], [obj.c], sm, [obj.d])
            results[f"soft-min tau={tau:g} MPR"] = mpr(fo, P=args.P, iters=it, refine_sweeps=0, seed=seed)
            results[f"soft-min tau={tau:g} MPR+SR"] = mpr(fo, P=args.P, iters=it, refine_sweeps=sw, seed=seed)
        for name, r in results.items():
            z = obj.field(obj.X_states(r["s"])).to(torch.complex128).cpu()
            row = dict(N=N, K=args.K, codebook=args.codebook, seed=seed, method=name,
                       time_s=r["time"], y_sum=obj.value_states(r["s"][None])[0].item(),
                       **ris_metrics(z, info))
            print(row); out.write(row)
    out.close()


if __name__ == "__main__":
    main()
