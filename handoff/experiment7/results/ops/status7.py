"""Exp 7 progress (read-only). Counts cached evaluations per context and the change since the last run."""
import json, time, subprocess
from pathlib import Path
OUT = Path("/workspace/ducanh/results/experiment7")
SNAP = Path("/workspace/ducanh/lanes/.status_prev.json")
LAB = {"dtd-eurosat-gtsrb-svhn": "C1", "eurosat-gtsrb-mnist-svhn": "C2", "eurosat-gtsrb-sun397-svhn": "C3",
       "stanford_cars-eurosat-gtsrb-mnist": "C4", "stanford_cars-eurosat-gtsrb-resisc45": "C5",
       "stanford_cars-eurosat-mnist-resisc45": "C6", "stanford_cars-eurosat-mnist-sun397": "C7"}
STEPS = ("controls", "method", "entropy", "wide")
def n(p):
    try: return len(json.loads(p.read_text()))
    except Exception: return 0
now = time.time()
cur = {c: {"val": n(OUT / c / "val_cache.json"), "proxy": n(OUT / c / "proxy_cache_n500_s0.json"),
           "test": n(OUT / c / "test_cache.json")} for c in LAB}
prev = json.loads(SNAP.read_text()) if SNAP.exists() else None
SNAP.write_text(json.dumps({"t": now, "cur": cur}))
run = []
for line in subprocess.run(["pgrep", "-af", "run_exp7.py"], capture_output=True, text=True).stdout.split("\n"):
    p = line.split()
    if "lane_harm/run_exp7.py" in p and "--contexts" in p:
        run.append(f"{p[p.index('lane_harm/run_exp7.py') + 1]} {LAB.get(p[p.index('--contexts') + 1], '?')}")
print(time.strftime("%H:%M UTC", time.gmtime(now)), "| running:", ", ".join(sorted(run)) or "none")
dt_h = (now - prev["t"]) / 3600 if prev else 0
print(f"{'ctx':4s} {'val':>5s} {'proxy':>6s} {'test':>5s}   per hour since last run ({dt_h*60:.0f} min)   done")
tot = 0
for c, lab in LAB.items():
    d = cur[c]
    done = [s for s in STEPS if (OUT / c / f"{s}_result.json").exists()]
    if prev and dt_h > 0:
        q = prev["cur"][c]; rv, rp = (d["val"] - q["val"]) / dt_h, (d["proxy"] - q["proxy"]) / dt_h; tot += rv
        rate = f"val {rv:6.1f}/h  proxy {rp:6.1f}/h"
    else:
        rate = "(first run: no speed yet)"
    print(f"{lab:4s} {d['val']:5d} {d['proxy']:6d} {d['test']:5d}   {rate:28s}   {','.join(done) or '-'}")
if prev and dt_h > 0:
    print(f"all contexts: {tot:.0f} new validation evaluations per hour")
