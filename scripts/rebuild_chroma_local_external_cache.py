from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the ORT 1.27-friendly local-external ONNX cache for the "
            "merged Chroma S2S graph. The cache hardlinks Hugging Face "
            "safetensor shards into the cache directory and rewrites ONNX "
            "external_data locations to point at those local shard names."
        )
    )
    parser.add_argument("--model-dir", default="models/chroma-4b")
    parser.add_argument("--bundle-dir", default="onnx/chroma-s2s-full-v2")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Output cache directory. Defaults to "
            "<bundle-dir>/ort-cache-ort-local-external."
        ),
    )
    parser.add_argument("--graph", default="s2s_merged")
    parser.add_argument("--provider", default="cuda")
    parser.add_argument("--memory-profile", default="quality-safe")
    parser.add_argument(
        "--copy-if-hardlink-fails",
        action="store_true",
        help=(
            "Copy safetensor shards if hardlink creation fails. This uses extra "
            "disk space and is not the preferred shared-weight setup."
        ),
    )
    return parser.parse_args()


def hardlink_or_copy(source: Path, target: Path, copy_if_hardlink_fails: bool) -> str:
    if target.exists():
        try:
            if os.path.samefile(source, target):
                return "existing-hardlink"
        except OSError:
            pass
        raise FileExistsError(f"{target} already exists and is not the same file as {source}.")

    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        if not copy_if_hardlink_fails:
            raise
        shutil.copy2(source, target)
        return "copy"


def main() -> int:
    args = parse_args()

    import onnx

    model_dir = Path(args.model_dir).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else bundle_dir / "ort-cache-ort-local-external"
    manifest_path = bundle_dir / "shared_weights_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph_info = manifest["graphs"].get(args.graph)
    if graph_info is None:
        raise KeyError(f"Graph {args.graph!r} was not found in {manifest_path}.")

    graph_path = Path(graph_info["path"])
    if not graph_path.is_file():
        graph_path = bundle_dir / graph_info["path"]
    graph_path = graph_path.resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f"ONNX graph not found: {graph_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    graph_entries = [
        entry for entry in manifest["initializers"] if entry["graph"] == args.graph
    ]
    entries_by_initializer = {
        entry["onnx_initializer"]: entry for entry in graph_entries
    }

    linked_shards = {}
    for source_shard in sorted({entry["source_shard"] for entry in graph_entries}):
        source_path = (model_dir / source_shard).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Safetensor shard not found: {source_path}")
        target_path = cache_dir / source_path.name
        mode = hardlink_or_copy(source_path, target_path, args.copy_if_hardlink_fails)
        linked_shards[source_path.name] = {
            "source": str(source_path),
            "target": str(target_path),
            "mode": mode,
            "bytes": target_path.stat().st_size,
        }

    model = onnx.load(str(graph_path), load_external_data=False)
    rewritten = 0
    missing_manifest_entries = []

    for initializer in model.graph.initializer:
        entry = entries_by_initializer.get(initializer.name)
        if entry is None:
            if initializer.external_data:
                missing_manifest_entries.append(initializer.name)
            continue

        source_shard = entry["source_shard"]
        target_path = cache_dir / source_shard
        if not target_path.is_file():
            raise FileNotFoundError(f"Linked shard not found: {target_path}")

        del initializer.external_data[:]
        initializer.data_location = onnx.TensorProto.EXTERNAL
        for key, value in (
            ("location", source_shard),
            ("offset", str(entry["source_offset"])),
            ("length", str(entry["byte_length"])),
        ):
            item = initializer.external_data.add()
            item.key = key
            item.value = value
        rewritten += 1

    if missing_manifest_entries:
        preview = ", ".join(missing_manifest_entries[:10])
        raise RuntimeError(
            "Some external initializers were not present in the manifest: "
            f"{preview}"
        )

    local_external_path = cache_dir / "chroma_s2s_merged.local_external.onnx"
    cache_key_path = cache_dir / (
        f"{args.graph}.{args.provider}.{args.memory_profile}.optimized.onnx"
    )
    onnx.save_model(model, str(local_external_path), save_as_external_data=False)
    shutil.copy2(local_external_path, cache_key_path)

    report = {
        "bundleDir": str(bundle_dir),
        "modelDir": str(model_dir),
        "cacheDir": str(cache_dir),
        "sourceGraph": str(graph_path),
        "localExternalGraph": str(local_external_path),
        "cacheKeyGraph": str(cache_key_path),
        "graph": args.graph,
        "provider": args.provider,
        "memoryProfile": args.memory_profile,
        "rewrittenInitializers": rewritten,
        "linkedShards": linked_shards,
    }
    (cache_dir / "local_external_cache_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
