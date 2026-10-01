"""Download official checkpoints to ignored artifacts; publish hashes, not weights."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

MODELS = {
    "vjepa21-b": ("https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt", "vjepa2_1_vitb_dist_vitG_384.pt"),
    "vjepa21-l": ("https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt", "vjepa2_1_vitl_dist_vitG_384.pt"),
    "dinov3-l": ("https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
}
USER_PROVIDED_DINO = "https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, folder, user_mirror=False):
    url, filename = MODELS[name]
    if user_mirror:
        if name != "dinov3-l":
            raise ValueError("User-approved mirror is only defined for DINOv3-L")
        url = USER_PROVIDED_DINO
    target = folder / filename
    partial = target.with_suffix(target.suffix+".part")
    started = time.monotonic()
    reused = target.exists()
    if not target.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "IPAD-JEPA-Research/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urlopen(Request(url, headers=headers), timeout=120) as response:
            if offset and response.status != 206:
                offset = 0
            content_length = response.headers.get("Content-Length")
            expected = int(content_length) + offset if content_length else None
            with partial.open("ab" if offset else "wb") as stream:
                count = offset
                reported = count // (256*1024**2)
                while chunk := response.read(8*1024**2):
                    stream.write(chunk)
                    count += len(chunk)
                    if count // (256*1024**2) > reported:
                        reported = count // (256*1024**2)
                        print(f"{name}: {count/1024**2:.0f} MiB", flush=True)
            if expected is not None and count != expected:
                raise IOError("Incomplete checkpoint download")
    digest = sha256(target if reused else partial)
    if name == "dinov3-l" and not digest.startswith("8aa4cbdd"):
        raise ValueError("DINOv3 checkpoint SHA256 does not match official filename hash prefix")
    if not reused:
        partial.rename(target)
    return {"model": name, "filename": filename, "source": url, "sha256": digest,
            "source_type": "user_provided_mirror" if user_mirror else "official",
            "official_hash_prefix_match": digest.startswith("8aa4cbdd") if name == "dinov3-l" else None,
            "bytes": target.stat().st_size, "download_seconds": round(time.monotonic()-started, 3),
            "reused_local_file": reused,
            "status": "downloaded_not_yet_validated"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/weights"))
    p.add_argument("--metadata", type=Path, default=Path("results/setup"))
    p.add_argument("--user-mirror", action="store_true", help="Explicitly use the user-provided DINOv3 mirror; verify official hash prefix")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.metadata.mkdir(parents=True, exist_ok=True)
    try:
        result = download(args.model,args.out,args.user_mirror)
    except HTTPError as e:
        # Access-gated official weights require legitimate approval; never mirror-fallback.
        result = {"model": args.model, "status": "http_access_failed", "http_status": e.code,
                  "action": "Use official approved checkpoint URL or local file; do not bypass gating."}
        (args.metadata / f"{args.model}_download_failure.json").write_text(json.dumps(result,indent=2)+"\n")
        raise SystemExit(f"{args.model}: official endpoint returned HTTP {e.code}")
    prior_path = args.metadata / f"{args.model}.json"
    if result["reused_local_file"]:
        if not prior_path.exists():
            raise SystemExit("Existing checkpoint has no source metadata; record its actual provenance before reusing")
        prior = json.loads(prior_path.read_text())
        if prior.get("sha256") != result["sha256"]:
            raise SystemExit("Existing checkpoint differs from recorded source metadata")
        result.update({k:prior[k] for k in ("source","source_type","status")})
    (args.metadata / f"{args.model}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
