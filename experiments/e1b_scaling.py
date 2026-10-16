"""E1b (IEEE Access): scaling to large apertures.
Per N: closed-form gradient vs autograd (one evaluation), one full MPR run
(T iterations, no refinement), one exact SR sweep, and GPU peak memory."""
import time
from common import *


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def bench(f, reps, dev):
    f(); sync(dev)
    t = time.perf_counter()
    for _ in range(reps):
        f()
    sync(dev)
    return (time.perf_counter() - t) / reps


def peak(dev, f):
    if not str(dev).startswith("cuda"):
        f(); return float("nan")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    f(); sync(dev)
    return torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    p = base_parser(__doc__)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--M", type=int, default=500)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--Ns", default="256,512,1024,2048,4096,8192,16384,32768")
    p.add_argument("--autograd_max_N", type=int, default=16384)
    args = p.parse_args()
    Ns = [64, 128, 256] if args.quick else [int(n) for n in args.Ns.split(",")]
    T = 20 if args.quick else args.T
    reps = 3 if args.quick else 10
    cdt, rdt = dtypes(args)
    out = CSV("e1b_scaling", args)
    for N in Ns:
        a, B = random_problem(M=args.M, N=N, seed=0)
        obj = make_obj(a, B, torch.randn(1 << args.K, dtype=torch.complex128), args=args)
        x = torch.rand(args.P, N, args.K, dtype=rdt, device=args.device)
        s = torch.randint(0, obj.S, (args.P, N), device=args.device)
        row = dict(N=N, M=args.M, K=args.K, P=args.P, device=args.device, dtype=args.dtype)
        row["t_grad_ms"] = 1e3 * bench(lambda: obj.value_grad(x), reps, args.device)
        if N <= args.autograd_max_N:
            row["t_autograd_ms"] = 1e3 * bench(lambda: obj.value_grad_autograd(x), reps, args.device)
        else:
            row["t_autograd_ms"] = float("nan")
        row["t_mpr_s"] = bench(lambda: mpr(obj, P=args.P, iters=T, refine_sweeps=0), 1, args.device)
        row["t_sr_sweep_s"] = bench(lambda: successive_refinement(obj, s, max_sweeps=1), 1, args.device)
        row["mem_mpr_MB"] = peak(args.device, lambda: mpr(obj, P=args.P, iters=5, refine_sweeps=0))
        row["mem_B_MB"] = B.numel() * (8 if cdt == torch.complex64 else 16) / 2 ** 20
        print(row); out.write(row)
        del obj, B, x
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
