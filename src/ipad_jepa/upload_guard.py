"""Inspect all Git-tracked files before publishing experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess

BLOCKED = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".mp4", ".avi", ".mov", ".pdf", ".docx", ".pptx", ".jpg", ".jpeg"}
BLOCKED_DIRS = {"data", "IPAD_dataset", "artifacts", "weights", "cache", ".venv", "third_party"}


def inspect(path: Path, relative: Path) -> list[str]:
    errors = []
    if path.is_symlink():
        return [f"Symlink: {relative}"]
    if relative.suffix.lower() in BLOCKED or BLOCKED_DIRS.intersection(relative.parts):
        errors.append(f"Raw data/model/cache file: {relative}")
    if relative.name.startswith(".env") and relative.name != ".env.example":
        errors.append(f"Environment secrets: {relative}")
    if path.stat().st_size > 10 * 1024**2:
        errors.append(f"Exceeds 10 MiB: {relative}")
    if path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".svg"}:
        content = path.read_text(errors="replace")
        if re.search(r"(?:hf_|ghp_)[A-Za-z0-9]{20,}|[?&](?:Signature|X-Amz-Signature|token)=", content):
            errors.append(f"Possible token/signed URL: {relative}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    files = subprocess.check_output(["git", "-C", str(args.repo), "ls-files", "-z"]).decode().split("\0")
    errors = [e for name in files if name for e in inspect(args.repo / name, Path(name))]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Upload guard passed: {sum(bool(n) for n in files)} tracked files; no model/raw-data files.")


if __name__ == "__main__":
    main()
