"""Balanced experiment campaign: one job queue shared by all GPUs.

Every experiment is cut into small chunks (a few instances / one array size
each).  N workers, one per GPU, pull the next chunk as soon as their GPU is
free (largest estimated chunks first), so all GPUs finish at about the same
time.  Each chunk writes its own  <exp>_part_<id>.csv  in the results folder;
make_figures.py merges the parts automatically.

  python run_campaign.py                      # all GPUs from nvidia-smi
  python run_campaign.py --gpus 0,1,2,3       # subset
  python run_campaign.py --quick --gpus a,b   # CPU smoke test of the whole pipeline

Results folder: $MPR_RESULTS (default experiments/results_v2).
Restarting skips chunks that already finished (markers in <results>/_done);
a chunk that was interrupted is rerun from its start.
"""
import argparse, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUT = os.path.join(ROOT, "data", "codebooks", "unit_cell_look_up_table.txt")
TAB = os.path.join(ROOT, "data", "codebooks", "measured_16state.csv")
BRISC = os.path.join(ROOT, "data", "brisc")

# rough cost per instance in seconds (from the first 5-instance run); used only
# to order the queue (largest first) and for the ETA, not for correctness
COST = {"e2": 160, "e3": 340, "e11_45x45": 190, "e11_32x32": 86, "e11_16x16": 24,
        "e5": 100, "e7": 120, "e8": 15, "e9": 80,
        "e11m_45x45": 60, "e11m_32x32": 20, "e11m_16x16": 5, "e12_pos": 400}


def chunks(n_total, size):
    return [(s, min(size, n_total - s)) for s in range(0, n_total, size)]


def build_jobs(q):
    """returns list of (name, cost, argv).  q = quick mode (CPU smoke test)."""
    S = (lambda n: 2) if q else (lambda n: n)          # instances per experiment
    J = []
    lut = ["--lut", LUT] if os.path.exists(LUT) else []
    tab = ["--table", TAB] if os.path.exists(TAB) else []
    # single jobs (fixed output names expected by make_figures.py)
    J.append(("e1b_scaling", 2400, ["e1b_scaling.py"]))
    J.append(("e10_pattern", 300, ["e10_pattern.py"]))
    if lut:
        J.append(("e6_measured", 1200, ["e6_measured.py", "--seeds", str(S(20))] + lut))
    if tab:   # measured 16-state (4-bit) table: own output names (*_table.csv)
        J.append(("e6_table", 2400, ["e6_measured.py", "--seeds", str(S(20))] + tab))
        for size, per in (("45x45", 10), ("32x32", 25), ("16x16", 50)):
            for s0, k in chunks(S(100), 1 if q else per):
                J.append((f"e11m_{size}_s{s0:03d}", COST[f"e11m_{size}"] * k,
                          ["e11_optimality.py", "--sizes", size, "--seeds", str(k), "--seed0", str(s0),
                           "--only_measured"] + tab))
    # BRISC measured channels (only if downloaded + prepared, see data/brisc/README.md)
    sys.path.insert(0, ROOT)
    from mpr.brisc import available_positions
    bpos = available_positions(BRISC, "log")
    if q:
        J.append(("e12b_pos1", 60, ["e12_brisc.py", "--synthetic", "--positions", "1", "--pairs", "none"]))
    elif bpos:
        for k in bpos:
            # e12b_*: second version with MPR+SR-dedup (same output files as e12_*)
            J.append((f"e12b_pos{k}", COST["e12_pos"], ["e12_brisc.py", "--seeds", "5", "--positions", str(k),
                                                         "--pairs", "none", "--data", BRISC]))
        if len(bpos) >= 2:
            J.append(("e12b_pairs", 600, ["e12_brisc.py", "--seeds", "5", "--only_pairs", "--max_pairs", "12",
                                          "--positions", ",".join(map(str, bpos)), "--data", BRISC]))
    else:
        print("note: no BRISC data in", BRISC, "-> e12 skipped (see data/brisc/README.md)", flush=True)
    # main tables: 100 instances, 4 per chunk
    for exp, script, n, per in (("e2", "e2_array.py", 100, 4), ("e3", "e3_ris.py", 100, 4)):
        for s0, k in chunks(S(n), 1 if q else per):
            J.append((f"{exp}_s{s0:03d}", COST[exp] * k, [script, "--seeds", str(k), "--seed0", str(s0)]))
    # global optimum: 100 instances, chunked per array size
    for size, per in (("45x45", 4), ("32x32", 10), ("16x16", 25)):
        for s0, k in chunks(S(100), 1 if q else per):
            J.append((f"e11_{size}_s{s0:03d}", COST[f"e11_{size}"] * k,
                      ["e11_optimality.py", "--sizes", size, "--seeds", str(k), "--seed0", str(s0)] + lut))
    # secondary experiments
    for exp, script, n, per in (("e5", "e5_anytime.py", 20, 4), ("e7", "e7_variation.py", 20, 4),
                                ("e8", "e8_fairness.py", 20, 10), ("e9", "e9_sensitivity.py", 10, 5)):
        for s0, k in chunks(S(n), 1 if q else per):
            J.append((f"{exp}_s{s0:03d}", COST[exp] * k, [script, "--seeds", str(k), "--seed0", str(s0)]))
    single = {"e1b_scaling", "e10_pattern", "e6_measured"}
    fixed_tag = {"e6_table": "table"}
    out = []
    for name, cost, argv in J:
        if name in fixed_tag:
            tag = ["--tag", fixed_tag[name]]
        else:
            tag = [] if name in single else ["--tag", "part_" + name.split("_", 1)[1]]
        if q:
            argv = argv + ["--quick"]
            if "e11" in name:      # quick mode ignores --sizes; keep one chunk per seed
                if not (name.startswith("e11_45x45") or name.startswith("e11m_45x45")):
                    continue
        out.append((name, cost, argv + tag))
    # e1b first (timing figure: start on an otherwise idle device), then largest first
    out.sort(key=lambda j: (j[0] != "e1b_scaling", -j[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=None, help="comma list; default: all GPUs from nvidia-smi")
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--quick", action="store_true", help="CPU smoke test (tiny sizes, device cpu)")
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args()

    res = os.path.abspath(os.environ.get("MPR_RESULTS") or os.path.join(HERE, "results_v2"))
    os.environ["MPR_RESULTS"] = res
    logs, done_dir = os.path.join(res, "logs"), os.path.join(res, "_done")
    os.makedirs(logs, exist_ok=True); os.makedirs(done_dir, exist_ok=True)

    if a.gpus:
        gpus = a.gpus.split(",")
    else:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        gpus = [g.strip() for g in out.split() if g.strip()]
    if not gpus:
        sys.exit("no GPUs found (use --gpus)")

    jobs = build_jobs(a.quick)
    todo = [j for j in jobs if not os.path.exists(os.path.join(done_dir, j[0]))]
    total_cost = sum(j[1] for j in jobs)
    done_cost = [total_cost - sum(j[1] for j in todo)]
    lock = threading.Lock()
    queue = list(todo)
    failed, t0 = [], time.time()
    status = os.path.join(res, "status.txt")
    tsv = os.path.join(res, "jobs.tsv")
    if not os.path.exists(tsv):
        open(tsv, "w").write("job\tgpu\tstart\tseconds\treturncode\n")
    print(f"results: {res}\nGPUs: {gpus}\njobs: {len(jobs)} total, {len(todo)} to run "
          f"(estimated {sum(j[1] for j in todo) / len(gpus) / 3600:.1f} h on {len(gpus)} GPUs)", flush=True)

    def report():
        el = time.time() - t0
        run_cost = done_cost[0] - (total_cost - sum(j[1] for j in todo))
        eta = el * (total_cost - done_cost[0]) / run_cost if run_cost > 0 else float("nan")
        line = (f"{time.strftime('%H:%M:%S')}  done {len(jobs) - len(queue) - len(running)}/{len(jobs)} "
                f"(est. {100 * done_cost[0] / total_cost:.0f}% of work)  running {len(running)}  "
                f"failed {len(failed)}  elapsed {el / 3600:.2f} h  ETA {eta / 3600:.2f} h")
        print(line, flush=True)
        with open(status, "w") as f:
            f.write(line + "\nrunning: " + ", ".join(f"{g}:{n}" for g, n in sorted(running.items())) + "\n")
            if failed:
                f.write("failed: " + ", ".join(failed) + "\n")

    running = {}

    def worker(gpu):
        env = dict(os.environ)
        if not a.quick:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        while True:
            with lock:
                if not queue:
                    return
                name, cost, argv = queue.pop(0)
                running[gpu] = name
            cmd = [sys.executable, "-u"] + argv + ["--device", "cpu" if a.quick else "cuda", "--P", str(a.P)]
            ts = time.time()
            with open(os.path.join(logs, name + ".log"), "w") as lf:
                lf.write("# " + " ".join(cmd) + "\n"); lf.flush()
                rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
            dt = time.time() - ts
            with lock:
                running.pop(gpu, None)
                with open(tsv, "a") as f:
                    f.write(f"{name}\t{gpu}\t{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}\t{dt:.0f}\t{rc}\n")
                if rc == 0:
                    open(os.path.join(done_dir, name), "w").close()
                    done_cost[0] += cost
                else:
                    failed.append(name)
                report()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for i, t in enumerate(threads):
        t.start()
        if i == 0:
            time.sleep(5)       # let e1b_scaling grab the first GPU before the others start
    for t in threads:
        t.join()
    report()
    if failed:
        print("FAILED jobs (see logs/):", " ".join(failed), flush=True)
    if not a.no_plots:
        print("running make_figures.py ...", flush=True)
        subprocess.call([sys.executable, "make_figures.py"], cwd=HERE)
    print("finished; results in", res, flush=True)


if __name__ == "__main__":
    main()
