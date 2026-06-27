#!/usr/bin/env python3
"""Download FlashLabs/Chroma-4B with explicit gate and disk checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_ID = "FlashLabs/Chroma-4B"
MODEL_URL = "https://huggingface.co/FlashLabs/Chroma-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", default="models/chroma-4b")
    parser.add_argument("--min-free-gib", type=float, default=80.0)
    parser.add_argument("--skip-weights", action="store_true", help="Download model code/config/tokenizer assets without safetensors shards.")
    parser.add_argument("--no-disk-check", action="store_true")
    return parser.parse_args()


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except Exception as exc:  # pragma: no cover - only used before dependency install
        raise SystemExit(
            "Missing dependency 'huggingface_hub'. Install with:\n"
            "  pip install -r requirements-convert.txt\n"
            f"Original import error: {exc}"
        ) from exc

    return HfApi, hf_hub_download, snapshot_download, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def format_gib(num_bytes: int | None) -> str:
    if not num_bytes:
        return "unknown"
    return f"{num_bytes / 1024**3:.2f} GiB"


def sibling_size_report(api, model_id: str, revision: str, token: str | None) -> tuple[int, list[tuple[str, int | None]]]:
    info = api.model_info(model_id, revision=revision, files_metadata=True, token=token)
    rows: list[tuple[str, int | None]] = []
    total = 0
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        rows.append((sibling.rfilename, size))
        if size:
            total += size
    return total, rows


def ensure_access(model_id: str, revision: str, token: str | None) -> None:
    _, hf_hub_download, _, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError = require_huggingface_hub()
    try:
        hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=revision,
            token=token,
            local_files_only=False,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            f"{model_id} is gated. Accept the conditions at {MODEL_URL}, then set HF_TOKEN "
            "or run `huggingface-cli login` and retry."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise SystemExit(f"Could not find model repo {model_id!r}.") from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            raise SystemExit(
                f"Unauthorized for {model_id}. Accept the gate at {MODEL_URL} and set HF_TOKEN."
            ) from exc
        raise


def ignore_patterns(skip_weights: bool) -> Iterable[str] | None:
    if not skip_weights:
        return None
    return ("*.safetensors", "model-*.safetensors", "*.bin", "*.pt", "*.pth")


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    token = os.environ.get("HF_TOKEN")

    HfApi, _, snapshot_download, *_ = require_huggingface_hub()
    api = HfApi()

    print(f"Model: {args.model_id}@{args.revision}")
    print(f"Destination: {local_dir}")
    if token:
        print("HF_TOKEN: present")
    else:
        print("HF_TOKEN: missing; relying on any cached Hugging Face login")

    total, rows = sibling_size_report(api, args.model_id, args.revision, token)
    print(f"Repository file total: {format_gib(total)}")
    for name, size in rows:
        if size and size > 256 * 1024**2:
            print(f"  {format_gib(size):>10}  {name}")

    if not args.no_disk_check:
        available = free_gib(local_dir.parent)
        print(f"Free space near destination: {available:.2f} GiB")
        if available < args.min_free_gib:
            raise SystemExit(
                f"Not enough free space. Need at least {args.min_free_gib:.1f} GiB by request, "
                f"found {available:.1f} GiB. Free disk space or pass --min-free-gib/--no-disk-check."
            )

    ensure_access(args.model_id, args.revision, token)

    start = time.time()
    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        token=token,
        local_dir=str(local_dir),
        ignore_patterns=ignore_patterns(args.skip_weights),
    )
    elapsed = time.time() - start

    manifest = {
        "model_id": args.model_id,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "snapshot_path": str(path),
        "skip_weights": args.skip_weights,
        "downloaded_at_unix": int(time.time()),
    }
    manifest_path = local_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Downloaded to: {path}")
    print(f"Wrote: {manifest_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
