#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


FP8_MAX = 448.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _quantized_name(name: str) -> bool:
    return (name.endswith(".weight") and
            (".self_attn." in name or ".mlp." in name))


def _scale_name(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(name)
    return name[:-len(".weight")] + ".weight_scale"


def _expected_keys(source_keys: set[str]) -> set[str]:
    return source_keys | {_scale_name(name) for name in source_keys
                          if _quantized_name(name)}


def _valid_shard(source: Path, target: Path) -> bool:
    if not target.is_file():
        return False
    from safetensors import safe_open
    try:
        with safe_open(source, framework="pt", device="cpu") as handle:
            source_keys = set(handle.keys())
        with safe_open(target, framework="pt", device="cpu") as handle:
            target_keys = set(handle.keys())
            if target_keys != _expected_keys(source_keys):
                return False
            for name in source_keys:
                dtype = handle.get_slice(name).get_dtype()
                if _quantized_name(name) and dtype not in {
                        "F8_E4M3", "F8_E4M3FN"}:
                    return False
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _convert_shard(source: Path, target: Path, device_index: int) -> None:
    if _valid_shard(source, target):
        print(f"reuse {target.name}", flush=True)
        return
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    torch.cuda.set_device(device_index)
    tensors = load_file(source, device="cpu")
    output: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not _quantized_name(name):
            output[name] = tensor
            continue
        weight = tensor.to(device=f"cuda:{device_index}")
        amax = weight.abs().max().float()
        scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = (weight.float() / scale).clamp(
            -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        output[name] = quantized.cpu()
        output[_scale_name(name)] = scale.cpu().reshape(())
        del weight, quantized, scale, amax
    metadata = None
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    save_file(output, partial, metadata=metadata)
    os.replace(partial, target)
    if not _valid_shard(source, target):
        raise RuntimeError(f"converted shard validation failed: {target}")
    print(f"converted gpu={device_index} {source.name} -> {target.name}",
          flush=True)


def _worker(source_root: str, target_root: str,
            names: list[str], device_index: int) -> None:
    source = Path(source_root)
    target = Path(target_root)
    for name in names:
        _convert_shard(source / name, target / name, device_index)


def _copy_metadata(source: Path, target: Path) -> None:
    for path in source.iterdir():
        if (not path.is_file() or path.name == "config.json" or
                path.name == "model.safetensors.index.json" or
                path.suffix == ".safetensors"):
            continue
        shutil.copy2(path, target / path.name)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "ignored_layers": ["lm_head"],
    }
    (target / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _write_index(source: Path, target: Path) -> None:
    index = json.loads((source / "model.safetensors.index.json").read_text(
        encoding="utf-8"))
    weight_map = dict(index["weight_map"])
    for name, shard in list(weight_map.items()):
        if _quantized_name(name):
            weight_map[_scale_name(name)] = shard
    index["weight_map"] = dict(sorted(weight_map.items()))
    index.setdefault("metadata", {})["total_size"] = sum(
        path.stat().st_size for path in target.glob("*.safetensors"))
    (target / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _complete(target: Path, source_config_sha256: str) -> bool:
    manifest = target / "fp8_conversion_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    files = value.get("files", {})
    return (value.get("source_config_sha256") == source_config_sha256 and
            files and all((target / name).is_file() and
                          (target / name).stat().st_size == size
                          for name, size in files.items()))


def convert(source: Path, target: Path, workers: int) -> Path:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("source and target must differ")
    source_index = source / "model.safetensors.index.json"
    if not source_index.is_file():
        raise FileNotFoundError(source_index)
    source_config_sha256 = _sha256(source / "config.json")
    if _complete(target, source_config_sha256):
        print(f"complete FP8 checkpoint already exists: {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    _copy_metadata(source, target)
    index = json.loads(source_index.read_text(encoding="utf-8"))
    names = sorted(set(index["weight_map"].values()))
    assignments = [names[index::workers] for index in range(workers)]
    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=_worker, args=(
        str(source), str(target), assigned, device))
        for device, assigned in enumerate(assignments) if assigned]
    started = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes
                if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"FP8 conversion workers failed: {failures}")
    _write_index(source, target)
    files = {path.name: path.stat().st_size
             for path in sorted(target.glob("*.safetensors"))}
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "target": str(target),
        "source_config_sha256": source_config_sha256,
        "quantization": "fp8_e4m3fn_per_tensor_dynamic_activation",
        "ignored_layers": ["lm_head"],
        "workers": workers,
        "files": files,
        "total_size": sum(files.values()),
        "completed_at": time.time(),
        "elapsed_s": time.time() - started,
    }
    (target / "fp8_conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be in 1..4")
    print(convert(args.source, args.target, args.workers))


if __name__ == "__main__":
    main()
