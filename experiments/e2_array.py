"""E2: phased-array beam synthesis with K-bit phase shifters (UPA).
Sweeps array size and K; random steering direction per instance."""
import math
from common import *


def main():
    p = base_parser(__doc__)
    p.add_argument("--side_weight", type=float, default=100.0)
    p.add_argument("--M", type=int, default=500)
    args = p.parse_args()
    sizes = [(8, 8), (12, 12)] if args.quick else [(16, 16), (32, 32), (45, 45)]
    Ks = [1, 2, 3] if args.quick else [1, 2, 3, 4, 5]
    iters, sweeps, gens = (100, 3, 50) if args.quick else (300, 10, 500)
    M = 150 if args.quick else args.M
    out = CSV("e2_array", args)
    for (Nx, Ny) in sizes:
        for seed in range(args.seed0, args.seed0 + args.seeds):
            g = torch.Generator().manual_seed(1000 + seed)
            th = math.radians(40) * torch.rand(1, generator=g).item()
            ph = 2 * math.pi * torch.rand(1, generator=g).item()
            u0, v0 = math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph)
            a, B, info = array_problem(Nx, Ny, M=M, main_dir=(u0, v0),
                                       side_weight=args.side_weight, seed=seed)
            for K in Ks:
                obj = make_obj(a, B, ideal_phase_codebook(K), args=args)
                yr = y_random(obj)
                for r in run_methods(obj, args, iters=iters, sweeps=sweeps, ga_gens=gens, seed=seed):
                    X = obj.X_states(r["s"]).to(torch.complex128).cpu()
                    met = array_metrics(X, info, grid=128 if args.quick else 512)
                    row = dict(N=Nx * Ny, K=K, seed=seed, method=r["name"], y=r["y"], y_rand=yr,
                               time_s=r["time"], **met)
                    print(row); out.write(row)
    out.close()


if __name__ == "__main__":
    main()
