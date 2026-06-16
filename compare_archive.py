"""
Compare experiment folders inside experiments_logs/archive1/ against the
top-level experiments_logs/. For every scenario (e.g. 'mhealth to Pamap2_EXP_HAR')
report which algorithm subfolders are in one but not the other, and which are
in both.
"""
import os
from pathlib import Path

ROOT = Path("/mnt/data/home/tp2474/AdaTime-adatime_v2/experiments_logs")
ARCHIVE = ROOT / "archive1"

def list_scenarios(path: Path):
    return {p.name for p in path.iterdir() if p.is_dir() and " to " in p.name}

def list_algos(path: Path):
    if not path.is_dir():
        return set()
    return {p.name for p in path.iterdir() if p.is_dir()}

def run_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir() and "_run_" in p.name)

live = list_scenarios(ROOT)
arch = list_scenarios(ARCHIVE)

print("=" * 72)
print(f"Scenarios live-only  : {sorted(live - arch)}")
print(f"Scenarios archive-only: {sorted(arch - live)}")
print(f"Scenarios in BOTH    : {sorted(live & arch)}")
print("=" * 72)

for scen in sorted(live & arch):
    live_algos = list_algos(ROOT / scen)
    arch_algos = list_algos(ARCHIVE / scen)

    only_live = live_algos - arch_algos
    only_arch = arch_algos - live_algos
    both = live_algos & arch_algos

    print(f"\n--- {scen} ---")
    print(f"  only in live    ({len(only_live):>2}): {sorted(only_live)}")
    print(f"  only in archive ({len(only_arch):>2}): {sorted(only_arch)}")
    if both:
        print(f"  in BOTH         ({len(both):>2}):")
        for a in sorted(both):
            rc_live = run_count(ROOT / scen / a)
            rc_arch = run_count(ARCHIVE / scen / a)
            flag = "  <-- differ" if rc_live != rc_arch else ""
            print(f"      {a:30s} live={rc_live:>2} runs  archive={rc_arch:>2} runs{flag}")

print("\n" + "=" * 72)
print("Scenarios present only in archive1/ (nothing to merge against):")
for scen in sorted(arch - live):
    print(f"  {scen}  ({len(list_algos(ARCHIVE / scen))} algos)")
