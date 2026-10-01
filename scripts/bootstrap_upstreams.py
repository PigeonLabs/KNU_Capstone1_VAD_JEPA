"""Fetch exact official source revisions into Git-ignored third_party."""
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
spec=json.loads((root/"configs/upstreams.json").read_text())
for name, source in spec.items():
    destination=root/"third_party"/name
    if not destination.exists():
        subprocess.run(["git","clone",source["url"],str(destination)],check=True)
    actual=subprocess.check_output(["git","-C",str(destination),"remote","get-url","origin"]).decode().strip()
    if actual!=source["url"]:
        raise SystemExit(f"Unexpected upstream origin for {name}")
    subprocess.run(["git","-C",str(destination),"checkout","--detach",source["commit"]],check=True)
    actual=subprocess.check_output(["git","-C",str(destination),"rev-parse","HEAD"]).decode().strip()
    if actual!=source["commit"]:
        raise SystemExit(f"Revision mismatch: {name}")
    print(name,actual)

