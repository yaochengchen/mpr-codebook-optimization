"""E6 (IEEE Access): measured unit-cell codebooks.

Inputs (see data/codebooks/README.md):
  --lut    measured LUT text file (control ; magnitude [dB] ; phase [deg]),
           e.g. the public varactor-RIS table; 2^K rows are SELECTED (no
           interpolation) by maximal phase coverage -> K-bit hardware codebook.
  --table  16-state CSV (code,freq_GHz,amp_dB,phase_deg) for a native 4-bit
           RIS; every frequency in the file is used as its own codebook, and,
           if two or more are present, a dual-band design shares the bits.

Comparisons on the RIS scenario (users + leakage, several subcarriers):
  ideal-design : MPR+SR designed with the ideal uniform K-bit codebook, then the
                 nearest-phase measured state is applied (what a designer who
                 ignores the measurement would deploy)
  CQ, MO, CQ+SR, MO+SR, SR, MPR, MPR+SR : run on the measured codebook itself
For dual band: per-band designs evaluated on both bands vs the joint design.
"""
import json
import math
from common import *


def nearest_phase_states(X_ideal_states_phase, cb_phase):
    d = torch.remainder(X_ideal_states_phase[:, None] - cb_phase[None, :] + math.pi, 2 * math.pi) - math.pi
    return d.abs().argmin(-1)


def bitpattern_of_state(K, mapping="gray"):
    """inverse map: state q -> bit pattern s (for V[s] indexing)."""
    m = bit_to_state_map(K, mapping)
    inv = torch.empty_like(m); inv[m] = torch.arange(len(m))
    return inv


def eval_states(obj, s):
    return obj.value_states(s[None])[0].item()


def run_single(cb, K, label, args, out, sizes, seeds, nf):
    inv = bitpattern_of_state(K)
    for (Nx, Ny) in sizes:
        for seed in range(args.seed0, args.seed0 + seeds):
            a, B, d, info = ris_problem(Nx, Ny, n_freq=nf, leak_weight=4.0, seed=seed)
            obj = make_obj(a, B, cb, d=d, args=args)
            yr = y_random(obj)
            rows = run_methods(obj, args, methods=["CQ", "MO", "CQ+SR", "MO+SR", "SR", "MPR", "MPR+SR"],
                               seed=seed, iters=100 if args.quick else 300, sweeps=3 if args.quick else 10)
            # ideal-codebook design deployed on the measured hardware
            obj_id = make_obj(a, B, ideal_phase_codebook(K), d=d, args=args)
            r_id = mpr(obj_id, P=args.P, iters=100 if args.quick else 300,
                       refine_sweeps=3 if args.quick else 10, seed=seed)
            q_ideal = bit_to_state_map(K)[r_id["s"].cpu()]                 # ideal state index
            ph_ideal = 2 * math.pi * q_ideal.double() / (1 << K)
            ph_ideal = torch.remainder(ph_ideal + math.pi, 2 * math.pi) - math.pi
            q_meas = nearest_phase_states(ph_ideal, torch.remainder(cb.angle() + math.pi, 2 * math.pi) - math.pi)
            s_meas = inv[q_meas].to(args.device)
            t_id = r_id["time"]
            rows.append(dict(name="ideal-design", y=eval_states(obj, s_meas), s=s_meas, time=t_id))
            for r in rows:
                z = obj.field(obj.X_states(r["s"])).to(torch.complex128).cpu()
                row = dict(codebook=label, N=Nx * Ny, K=K, seed=seed, method=r["name"], y=r["y"],
                           y_rand=yr, time_s=r["time"], **ris_metrics(z, info))
                print(row); out.write(row)


def run_dualband(lut_list, labels, args, out, sizes, seeds):
    """bits shared across bands; band f uses its own measured codebook."""
    K = 4
    f0 = sum(lut.meta["freq_GHz"] for lut in lut_list) / len(lut_list)
    for (Nx, Ny) in sizes:
        for seed in range(args.seed0, args.seed0 + seeds):
            comps, metas = [], []
            for lut in lut_list:
                a, B, d, info = ris_problem(Nx, Ny, leak_weight=4.0, seed=seed,
                                            freq_ratios=[lut.meta["freq_GHz"] / f0])
                info_cb = select_states(lut, K, rule="all")
                cdt, rdt = dtypes(args)
                comps.append((a.to(args.device, rdt), B.to(args.device, cdt),
                              codebook_to_coeffs(info_cb.codebook, "gray").to(args.device, cdt),
                              d.to(args.device, cdt)))
                metas.append(info)
            joint = FieldObjective([c[1] for c in comps], [c[2] for c in comps],
                                   Quadratic([c[0] for c in comps]), [c[3] for c in comps])
            per_band = [FieldObjective([c[1]], [c[2]], Quadratic([c[0]]), [c[3]]) for c in comps]
            it, sw = (100, 3) if args.quick else (300, 10)
            cand = {"joint MPR+SR": mpr(joint, P=args.P, iters=it, refine_sweeps=sw, seed=seed)}
            for lab, ob in zip(labels, per_band):
                cand[f"{lab}-only MPR+SR"] = mpr(ob, P=args.P, iters=it, refine_sweeps=sw, seed=seed)
            for name, r in cand.items():
                s = r["s"][None]
                for lab, ob, info in zip(labels, per_band, metas):
                    z = ob.fields_states(s)[0][0].to(torch.complex128).cpu()
                    row = dict(codebook="dualband", design=name, band=lab, N=Nx * Ny, K=K, seed=seed,
                               y_joint=joint.value_states(s)[0].item(), time_s=r["time"],
                               **ris_metrics(z, info))
                    print(row); out.write(row)


def main():
    p = base_parser(__doc__)
    p.add_argument("--lut", default=None)
    p.add_argument("--lut_K", type=int, default=3)
    p.add_argument("--amp_floor_dB", type=float, default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--sizes", default="20x10,32x32,45x45")
    p.add_argument("--n_freq", type=int, default=16)
    args = p.parse_args()
    sizes = [(8, 8)] if args.quick else [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    seeds = 1 if args.quick else args.seeds
    nf = 4 if args.quick else args.n_freq
    if not args.lut and not args.table:
        print("no measured data given (--lut / --table); see data/codebooks/README.md"); return
    out = CSV("e6_measured", args)
    meta = {}
    if args.lut:
        lut = load_lut(args.lut)
        info = select_states(lut, args.lut_K, amp_floor_dB=args.amp_floor_dB)
        meta["lut"] = dict(file=args.lut, K=args.lut_K, rows_used=info.hw_index,
                           control=info.control, note=info.note,
                           amp_dB=(20 * torch.log10(info.codebook.abs())).tolist(),
                           phase_deg=torch.rad2deg(info.codebook.angle()).tolist(),
                           all_amp_dB=(20 * torch.log10(lut.amp)).tolist(),
                           all_phase_deg=torch.rad2deg(lut.phase).tolist(), all_control=lut.control)
        print("LUT codebook:", info.note)
        run_single(info.codebook, args.lut_K, f"measured-LUT-{args.lut_K}bit", args, out, sizes, seeds, nf)
    if args.table:
        freqs = table_frequencies(args.table)
        luts = [load_state_table(args.table, f) for f in freqs]
        meta["table"] = {}
        for f, lut in zip(freqs, luts):
            info = select_states(lut, 4, rule="all")
            meta["table"][str(f)] = dict(codes=info.control,
                                         amp_dB=(20 * torch.log10(info.codebook.abs())).tolist(),
                                         phase_deg=torch.rad2deg(info.codebook.angle()).tolist())
            run_single(info.codebook, 4, f"measured-4bit-{f:g}GHz", args, out, sizes, seeds, nf)
        if len(luts) >= 2:
            out2 = CSV("e6_dualband", args)
            run_dualband(luts, [f"{f:g}GHz" for f in freqs], args, out2, sizes, seeds)
            out2.close()
    out.close()
    with open(os.path.join(RESULTS, f"e6_codebooks{('_' + args.tag) if args.tag else ''}.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
