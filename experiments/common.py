import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mpr import *

HERE = os.path.dirname(os.path.abspath(__file__))
# MPR_RESULTS overrides the output folder (used by run_campaign.py)
RESULTS = os.path.abspath(os.environ.get("MPR_RESULTS") or os.path.join(HERE, "results"))
os.makedirs(RESULTS, exist_ok=True)


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="c64", choices=["c64", "c128"],
                   help="complex64 is ~2x faster on GPU; c128 for reference runs")
    p.add_argument("--quick", action="store_true", help="tiny sizes for a CPU smoke test")
    p.add_argument("--seeds", type=int, default=5, help="number of random instances")
    p.add_argument("--seed0", type=int, default=0, help="first instance seed (split runs across GPUs)")
    p.add_argument("--P", type=int, default=64, help="parallel candidates per method")
    p.add_argument("--tag", default="", help="suffix for output csv")
    p.add_argument("--methods", default=None,
                   help="comma-separated subset of methods (default: all)")
    return p


def dtypes(args):
    return (torch.complex64, torch.float32) if args.dtype == "c64" else (torch.complex128, torch.float64)


def make_obj(a, B, codebook, mapping="gray", d=None, args=None):
    cdt, rdt = dtypes(args)
    dev = args.device
    return Objective(a.to(dev, rdt), B.to(dev, cdt),
                     codebook_to_coeffs(codebook.to(torch.complex128), mapping).to(dev, cdt),
                     None if d is None else d.to(dev, cdt))


def y_random(obj, n=256, seed=123):
    """mean objective of uniformly random configurations (normalisation anchor)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, obj.S, (n, obj.N), generator=g).to(obj.device)
    return obj.value_states(s).mean().item()


METHODS = ["CQ", "CQ+SR", "MO", "MO+SR", "GA", "SR", "MPR", "MPR+SR"]


def run_methods(obj, args, methods=None, iters=300, sweeps=10, ga_gens=500, seed=0):
    """Runs each method with the same candidate budget P.  Returns list of dicts."""
    if methods is None:
        methods = args.methods.split(",") if getattr(args, "methods", None) else METHODS
    P = args.P
    ph = obj.V.angle().to(obj.rdtype)          # state phases per bit pattern
    out = []
    for m in methods:
        if m == "CQ":
            r = cont_quantize(obj, ph, P=P, iters=iters, seed=seed)
        elif m == "CQ+SR":
            r = cont_quantize(obj, ph, P=P, iters=iters, refine_sweeps=sweeps, seed=seed)
        elif m == "MO":
            r = manifold_cq(obj, ph, P=P, iters=iters, seed=seed)
        elif m == "MO+SR":
            r = manifold_cq(obj, ph, P=P, iters=iters, refine_sweeps=sweeps, seed=seed)
        elif m == "GA":
            r = genetic(obj, pop=max(P, 128), gens=ga_gens, seed=seed)
        elif m == "SR":
            r = random_sr(obj, P=P, sweeps=sweeps, seed=seed)
        elif m == "MPR":
            r = mpr(obj, P=P, iters=iters, refine_sweeps=0, seed=seed)
        elif m == "MPR+SR":
            r = mpr(obj, P=P, iters=iters, refine_sweeps=sweeps, seed=seed)
        elif m == "MPR+SR-dedup":
            r = mpr(obj, P=P, iters=iters, refine_sweeps=sweeps, seed=seed, dedup=True)
        else:
            raise ValueError(m)
        r["name"] = m
        out.append(r)
    return out


class CSV:
    def __init__(self, name, args):
        self.path = os.path.join(RESULTS, f"{name}{('_' + args.tag) if args.tag else ''}.csv")
        self.f = open(self.path, "w", newline="")
        self.w = None

    def write(self, row):
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row.keys()))
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
        print("wrote", self.path)
