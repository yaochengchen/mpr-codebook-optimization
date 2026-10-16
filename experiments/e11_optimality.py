"""E11 (IEEE Access): gap to the GLOBAL optimum in the single-receiver case.
For one receive point, max |d + sum_j b_j X_j|^2 over arbitrary complex
codebooks is solved exactly by an angle sweep (optimize.single_user_optimum),
so every method's loss to the true optimum (dB) can be measured.
Codebooks: ideal, practical (model), and optionally a measured LUT (--lut) and
the measured 16-state (4-bit) table, one codebook per frequency (--table).
--only_measured skips ideal/practical and writes e11_measured4*.csv instead of
e11_optimality*.csv (used to add the 4-bit table without rerunning the rest)."""
from common import *


def main():
    p = base_parser(__doc__)
    p.add_argument("--sizes", default="16x16,32x32,45x45")
    p.add_argument("--lut", default=None)
    p.add_argument("--lut_K", type=int, default=3)
    p.add_argument("--table", default=None, help="16-state CSV (code,freq_GHz,amp_dB,phase_deg)")
    p.add_argument("--only_measured", action="store_true")
    args = p.parse_args()
    sizes = [(6, 6)] if args.quick else [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    Ks = [1, 2] if args.quick else [1, 2, 3, 4, 5]
    seeds = 2 if args.quick else args.seeds
    it, sw, gens = (50, 2, 30) if args.quick else (300, 10, 500)
    books = {} if args.only_measured else {"ideal": ideal_phase_codebook, "practical": practical_ris_codebook}
    extra = []                                   # (name, K, codebook) of measured tables
    if args.lut:
        extra.append((f"measured-LUT-{args.lut_K}bit", args.lut_K,
                      select_states(load_lut(args.lut), args.lut_K).codebook))
    if args.table:
        for f in table_frequencies(args.table):
            extra.append((f"measured-4bit-{f:g}GHz", 4,
                          select_states(load_state_table(args.table, f), 4, rule="all").codebook))
    out = CSV("e11_measured4" if args.only_measured else "e11_optimality", args)
    for (Nx, Ny) in sizes:
        for seed in range(args.seed0, args.seed0 + seeds):
            a, B, d, info = ris_problem(Nx, Ny, n_users=1, n_leak=0, n_freq=1, seed=seed)
            jobs = [(name, K, f(K)) for name, f in books.items() for K in Ks]
            jobs += extra
            for name, K, cb in jobs:
                obj = make_obj(a, B, cb, d=d, args=args)
                p_opt, _ = single_user_optimum(obj.B[0].to(torch.complex128),
                                               obj.V.to(torch.complex128),
                                               obj.d[0].to(torch.complex128) if obj.d is not None else None)
                for r in run_methods(obj, args, methods=["CQ", "MO", "CQ+SR", "SR", "GA", "MPR", "MPR+SR"],
                                     iters=it, sweeps=sw, ga_gens=gens, seed=seed):
                    p_m = -r["y"]
                    row = dict(N=Nx * Ny, codebook=name, K=K, seed=seed, method=r["name"],
                               p_opt=p_opt, p=p_m, loss_dB=10 * math.log10(p_opt / max(p_m, 1e-30)),
                               time_s=r["time"])
                    print(row); out.write(row)
    out.close()


if __name__ == "__main__":
    import math
    main()
