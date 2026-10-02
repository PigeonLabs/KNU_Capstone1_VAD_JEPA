"""Count real normal-fit candidates before choosing the paired extraction density."""
import argparse
import hashlib
import json
from pathlib import Path

from ipad_jepa.representation import capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/setup/representation_capacity.json"))
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())["sequences"]
    checks = []
    for mode in ["offline", "online"]:
        for device in ["R01", "R02", "R03", "R04"]:
            fit = [r for r in rows if r["device"] == device and r.get("split") == "fit"]
            for stride in [4, 1]:
                checks.append({"device": device, **capacity(fit, mode, stride)})
    if not all(r["enough_for_128_per_bin"] for r in checks if r["fit_stride"] == 1):
        raise ValueError("Dense normal fit still lacks observed candidates")
    source = Path(__file__).resolve().parents[1] / "src/ipad_jepa/representation.py"
    result = {"status": "manifest_geometry_verified", "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "sampling_code_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "checks": checks,
              "selected_paired_fit_stride": 1, "phase_bins": 16, "prototypes_per_bin": 128,
              "planned_seed_conditions": 96, "representations": ["patch", "global_mean"],
              "scope": "Actual manifest counts; no encoder execution or accuracy/runtime result"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
