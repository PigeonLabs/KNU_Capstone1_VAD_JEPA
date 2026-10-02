"""Acquire the fixed DINOv3-B checkpoint; record actual hosting and full hash, never weights."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

FILENAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
OFFICIAL = f"https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/{FILENAME}"
MIRROR = f"https://huggingface.co/jaychempan/dinov3/resolve/4412679/{FILENAME}"
MIRROR_PAGE = f"https://huggingface.co/jaychempan/dinov3/blob/4412679/{FILENAME}"
MIRROR_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-mirror", action="store_true", help="Explicitly select published third-party weights; no automatic fallback")
    parser.add_argument("--out", type=Path, default=Path("artifacts/weights"))
    parser.add_argument("--metadata", type=Path, default=Path("results/setup/dinov3-b.json"))
    args = parser.parse_args()
    target = args.out / FILENAME
    partial = target.with_suffix(".pth.part")
    if target.exists() or partial.exists() or args.metadata.exists():
        raise ValueError("Fresh download namespace required; never overwrite a live or verified checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    source = MIRROR if args.public_mirror else OFFICIAL
    h = hashlib.sha256(); count = 0
    request = Request(source, headers={"User-Agent": "IPAD-JEPA-Research/0.1"})
    with urlopen(request, timeout=60) as response, partial.open("xb") as output:
        expected = response.headers.get("Content-Length")
        while chunk := response.read(8 * 1024**2):
            output.write(chunk); h.update(chunk); count += len(chunk)
        if expected is not None and count != int(expected):
            raise ValueError("Incomplete downloaded checkpoint")
    sha = h.hexdigest()
    if not sha.startswith("73cec8be") or args.public_mirror and sha != MIRROR_SHA256:
        raise ValueError("Checkpoint differs from official filename prefix or published mirror full hash")
    partial.replace(target)
    result = {"model": "dinov3-b", "filename": FILENAME, "bytes": count, "sha256": sha,
              "source": source, "source_type": "public_third_party_mirror" if args.public_mirror else "official",
              "official_hash_prefix_match": True, "status": "downloaded_not_yet_strict_loaded",
              "official_reference": "Fixed upstream dinov3.hub.backbones.dinov3_vitb16; default hash 73cec8be",
              "scope": "Full local hash; official filename prefix; independent full Meta hash verification is not claimed"}
    if args.public_mirror:
        result.update(mirror_file_page=MIRROR_PAGE, mirror_published_sha256=MIRROR_SHA256,
                      mirror_full_hash_matches=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
