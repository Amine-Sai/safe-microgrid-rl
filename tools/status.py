import glob
import json
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)


def pytest_status():
    """The unit suite is ~6 s, so run it rather than trust a stale pytest cache."""
    try:
        # No extra -q on the command line: a second -q suppresses the
        # "N passed" summary line this parses.
        r = subprocess.run(["python", "-m", "pytest", "tests/"],
                           capture_output=True, text=True, timeout=300)
        txt = r.stdout + r.stderr
        m = re.search(r"^.*\b(\d+) passed.*$", txt, re.M)
        fail = re.search(r"^.*\b(\d+) failed.*$", txt, re.M)
        unit = (f"{m.group(1)} passed" if m else "unknown")
        if fail:
            unit += f", {fail.group(1)} FAILED"
        if r.returncode not in (0, 5):
            unit += f" (rc={r.returncode})"
    except Exception as e:
        unit = f"error {e}"
    return unit


def variant(r):
    """Which agent a run is: the base agent plus the shield/residual flags."""
    rc = r.get("resolved_config") or {}
    if rc.get("shielded"):
        return "ppo_shielded"
    if rc.get("residual_on_heuristic"):
        return "ppo_residual"
    return r.get("agent") or r.get("method")


def collect():
    """Normalize every run JSON shape we write into one row tuple."""
    rows = []
    for f in sorted(glob.glob("results/runs/*/*.json"), key=os.path.getmtime):
        if os.path.basename(f).endswith("_grid.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if "method" not in d:
            continue
        agent_dir = os.path.basename(os.path.dirname(f))
        curve = d.get("curve_val_cost") or {}
        fc = d.get("final_checkpoint") or {}
        met = fc.get("metrics") if isinstance(fc, dict) else None
        steps = d.get("total_steps") or (max(int(k) for k in curve) if curve else None)
        j = (met or {}).get("total_cost_eur")
        # the run id is the file stem
        rid = os.path.basename(f)[:-5]
        rows.append((
            ("%s/%s" % (agent_dir, rid))[:32],
            variant(d)[:16], steps,
            d.get("seed"), j,
            (met or {}).get("mean_exceedance_kw"),
            (met or {}).get("violation_rate_pct"),
            (d.get("lambda_summary") or {}).get("lambda_end"),
            d.get("wall_clock_sec"),
            os.path.getmtime(f)))
    rows.sort(key=lambda x: x[9])
    return rows


def main():
    out = []
    out.append(f"PYTEST: {pytest_status()}")
    out.append("TEST-LOCK: " + ("PRESENT (the published grid already ran)"
                                if os.path.exists("results/TEST_RUN.lock")
                                else "absent (Test split untouched)"))
    n = sum(len(re.findall(r"""["']test["']""",
                           open(f, encoding="utf-8", errors="replace").read()))
            for f in glob.glob("agents/**/*.py", recursive=True))
    out.append(f"TEST-READS under agents/: {n}  (must be 0 before the published grid runs)")
    out.append("RUNS  agent/run-id | variant | steps | seed | J | exc | V% | lambda | wall")
    fmt = lambda x, k=2: "-" if x is None else (f"{x:,.{k}f}" if isinstance(x, float) else str(x))
    for rid, m, st, sd, j, e, v, lam, w, _ in collect()[-10:]:
        out.append(f"  {rid:32s}{m:16s}{str(st):>6}{str(sd):>4} "
                   f"J={fmt(j):>12} exc={fmt(e, 4):>8} V%={fmt(v, 2):>6} "
                   f"lam={fmt(lam, 3):>7} {fmt(w, 0):>6}s")
    try:
        t = subprocess.run('tasklist /fi "imagename eq python.exe"', shell=True,
                           capture_output=True, text=True, timeout=30).stdout
        n_py = sum(1 for l in t.splitlines() if "python.exe" in l)
    except Exception:
        n_py = "?"
    out.append(f"PY PROCS: {n_py} | results/runs/: {len(glob.glob('results/runs/*/*.json'))} files")
    print("\n".join(out[:20]))


if __name__ == "__main__":
    main()
