"""E10 (IEEE Access): beam-pattern example (one UPA instance).
Saves the principal-plane cut through the steering direction for the
continuous-phase optimum (MO before quantisation) and the K-bit designs of
CQ, MO and MPR+SR -> results/e10_pattern.csv (u, dB per design)."""
from common import *


def cut(X, info, n=2001):
    px, py = info["px"].to(X.device), info["py"].to(X.device)
    u = torch.linspace(-1, 1, n, dtype=px.dtype, device=X.device)
    v = torch.full_like(u, info["v0"])
    ok = u ** 2 + v ** 2 <= 1
    A = steering(px, py, u, v).to(X.dtype) @ X
    p = (A.abs() ** 2) / (px.numel() ** 2)
    return u.cpu(), torch.where(ok, 10 * torch.log10(p.clamp_min(1e-12)), torch.nan).cpu()


def main():
    from mpr.problems import steering
    globals()["steering"] = steering
    p = base_parser(__doc__)
    p.add_argument("--Nx", type=int, default=32)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    Nx = 10 if args.quick else args.Nx
    it = 50 if args.quick else 300
    a, B, info = array_problem(Nx, Nx, M=150 if args.quick else 500, main_dir=(0.3, 0.2),
                               side_weight=100.0, seed=args.seed)
    obj = make_obj(a, B, ideal_phase_codebook(args.K), args=args)
    ph = obj.V.angle().to(obj.rdtype)
    r_mo = manifold_cq(obj, ph, P=args.P, iters=it, seed=args.seed)
    designs = {"continuous": r_mo["X_cont"],
               "CQ": obj.X_states(cont_quantize(obj, ph, P=args.P, iters=it, seed=args.seed)["s"]),
               "MO": obj.X_states(r_mo["s"]),
               "MPR+SR": obj.X_states(mpr(obj, P=args.P, iters=it, refine_sweeps=10, seed=args.seed)["s"])}
    cols = {}
    for k, X in designs.items():
        u, db = cut(X.to(torch.complex128), info)
        cols["u"] = u
        cols[k] = db
        cols[k + "_metrics"] = array_metrics(X.to(torch.complex128).cpu(), info, grid=128 if args.quick else 512)
    path = os.path.join(RESULTS, f"e10_pattern{('_' + args.tag) if args.tag else ''}.csv")
    import pandas as pd
    pd.DataFrame({k: v.numpy() for k, v in cols.items() if not k.endswith("_metrics")}).to_csv(path, index=False)
    print("wrote", path, {k: v for k, v in cols.items() if k.endswith("_metrics")})


if __name__ == "__main__":
    main()
