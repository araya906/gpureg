"""CPU benchmark of the automatic screening chunk size against fixed chunk sizes.

Usage:  python validation/bench_chunks.py [--quick] [--out validation/results]
Each (model, n, chunk) run is a fresh subprocess so its peak RSS can be measured. Writes
bench_chunks.md / bench_chunks.json. Informational only: not part of the validation gate.
Times are wall-clock on whatever CPU runs it; no GPU timing is measured or implied.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

CHUNKS = ("auto", 16, 64, 256, 1024)
CASES = (("binomial", 5000, 2000), ("binomial", 20000, 1000), ("binomial", 100000, 200),
         ("cox", 5000, 1000), ("cox", 20000, 400), ("cox", 50000, 100),
         ("cox_untied", 5000, 1000), ("cox_untied", 20000, 400), ("cox_untied", 50000, 100))
QUICK = (("binomial", 5000, 300), ("cox", 5000, 200), ("cox_untied", 5000, 200))


def _rss_mb(field):
    """VmRSS / VmHWM of this process in MB (Linux)."""
    line = next(l for l in open("/proc/self/status") if l.startswith(field + ":"))
    return int(line.split()[1]) / 1024


def _simulate(kind, n, m):
    """run_checks data; cox has integer (heavily tied) times, cox_untied jitters them apart."""
    from run_checks import simulate_screen
    C, F, out = simulate_screen(kind.replace("_untied", ""), n, m, seed=0)
    if kind == "cox_untied":
        out = (out[0] - 0.5 * np.random.default_rng(1).random(n), out[1])
    return C, F, out


def _heuristic(kind, n, out):
    """(max ties, k, copies) as glm_screen / cox_screen pass them to auto_chunk_size."""
    if not kind.startswith("cox"):
        return 0, 5, 6
    from gpureg.cox import _cox_setup
    ties = _cox_setup(out[0], out[1], None, None, torch.device("cpu"), torch.float64)[2].max_ties
    return ties, 4, 8 + 2 * ties


def run_one(kind, n, m, chunk):
    """Child process: time one screen and report the peak RSS it added (MB)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from gpureg import cox_screen, glm_screen
    from gpureg._device import auto_chunk_size

    C, F, out = _simulate(kind, n, m)
    ties, k, copies = _heuristic(kind, n, out)
    used = auto_chunk_size(n, k, torch.float64, torch.device("cpu"), copies)
    used = used if chunk == "auto" else int(chunk)
    Path("/proc/self/clear_refs").write_text("5")  # reset the peak-RSS high-water mark
    base, t0 = _rss_mb("VmRSS"), time.perf_counter()
    if kind.startswith("cox"):
        res = cox_screen(F, out[0], out[1], C, chunk_size=used)
    else:
        res = glm_screen(F, out, C, family=kind, chunk_size=used)
    secs = time.perf_counter() - t0
    print(json.dumps({"kind": kind, "n": n, "m": m, "chunk": chunk, "used": used,
                      "max_ties": ties, "seconds": secs, "peak_mb": _rss_mb("VmHWM") - base,
                      "predicted_mb": min(used, m) * n * k * 8 * copies / 2 ** 20,
                      "converged": float(res["converged"].mean())}))


def measure(kind, n, m, chunk):
    cmd = [sys.executable, __file__, "--child", kind, str(n), str(m), str(chunk)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def _table(rows):
    lines = ["| model | n | m | max ties | chunk | models/chunk | seconds | ms/model | "
             "peak MB | heuristic MB |", "|" + "---|" * 10]
    for r in rows:
        lines.append(f"| {r['kind']} | {r['n']} | {r['m']} | {r['max_ties']} | {r['chunk']} | "
                     f"{r['used']} | {r['seconds']:.2f} | {1e3 * r['seconds'] / r['m']:.2f} | "
                     f"{r['peak_mb']:.0f} | {r['predicted_mb']:.0f} |")
    return lines


def write_report(rows, out_dir):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    env = {"time": stamp, "platform": platform.platform(), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "rows": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench_chunks.json").write_text(json.dumps(env, indent=2))
    head = [f"# Chunk-size benchmark {stamp} (CPU, {env['threads']} threads, torch "
            f"{env['torch']})", "", "float64, 3 covariates, 5% missing feature values (validation/run_checks.py "
            "simulate_screen; cox times are integers 1..~300, cox_untied jitters them apart); "
            "peak MB is the RSS high-water mark added by the screen (fresh process per row, "
            "data already in memory); heuristic MB is what auto_chunk_size assumes the chunk "
            "needs (models/chunk x n x k x 8 bytes x copies; CPU default: budget 256e6 bytes, at most 64 models).", ""]
    (out_dir / "bench_chunks.md").write_text("\n".join(head + _table(rows)) + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return run_one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="validation/results")
    args = ap.parse_args()
    rows = []
    for kind, n, m in (QUICK if args.quick else CASES):
        for chunk in CHUNKS:
            if chunk != "auto" and chunk > m:
                continue
            rows.append(measure(kind, n, m, chunk))
            print(_table([rows[-1]])[-1], flush=True)
    write_report(rows, Path(args.out))


if __name__ == "__main__":
    main()
