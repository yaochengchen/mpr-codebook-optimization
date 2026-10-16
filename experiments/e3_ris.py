"""E3: RIS multi-user / multi-subcarrier power shaping with leakage suppression.
Compares ideal K-bit phase codebook vs the practical amplitude-phase-coupled
codebook (Abeywickrama et al.).  Continuous-then-quantise assumes ideal
unit-modulus phases, the bit-level methods use the exact codebook."""
from common import *


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_users", type=int, default=8)
    p.add_argument("--n_leak", type=int, default=8)
    p.add_argument("--n_freq", type=int, default=16)
    p.add_argument("--leak_weight", type=float, default=4.0)
    args = p.parse_args()
    sizes = [(8, 8), (12, 12)] if args.quick else [(16, 16), (32, 32), (45, 45)]
    Ks = [1, 2, 3] if args.quick else [1, 2, 3, 4, 5]
    iters, sweeps, gens = (100, 3, 50) if args.quick else (300, 10, 500)
    nf = 4 if args.quick else args.n_freq
    out = CSV("e3_ris", args)
    for (Nx, Ny) in sizes:
        for seed in range(args.seed0, args.seed0 + args.seeds):
            a, B, d, info = ris_problem(Nx, Ny, n_users=args.n_users, n_leak=args.n_leak,
                                        n_freq=nf, leak_weight=args.leak_weight, seed=seed)
            for K in Ks:
                for cbname, cb in (("ideal", ideal_phase_codebook(K)),
                                   ("practical", practical_ris_codebook(K))):
                    obj = make_obj(a, B, cb, d=d, args=args)
                    yr = y_random(obj)
                    for r in run_methods(obj, args, iters=iters, sweeps=sweeps, ga_gens=gens, seed=seed):
                        z = obj.field(obj.X_states(r["s"])).to(torch.complex128).cpu()
                        met = ris_metrics(z, info)
                        row = dict(N=Nx * Ny, M=obj.M, K=K, codebook=cbname, seed=seed,
                                   method=r["name"], y=r["y"], y_rand=yr, time_s=r["time"], **met)
                        print(row); out.write(row)
    out.close()


if __name__ == "__main__":
    main()
