#!/usr/bin/env python3
"""Build a TensorRT engine with dynamic batch and fixed spatial dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, required=True)
    parser.add_argument("--max-batch", type=int, required=True)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--task", choices=["detect", "pose"], required=True)
    parser.add_argument("--names-json", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def default_metadata(args: argparse.Namespace) -> dict:
    names = None
    if args.names_json and args.names_json.exists():
        names = json.loads(args.names_json.read_text(encoding="utf-8"))
    if names is None:
        names = {"0": "phone"} if args.task == "detect" else {"0": "person"}
    metadata = {
        "description": "JianKong TensorRT batch-dynamic fixed-spatial engine",
        "author": "Ultralytics",
        "version": "8.4.62",
        "task": args.task,
        "batch": args.max_batch,
        "imgsz": [args.height, args.width],
        "names": names,
        "stride": 32,
        "dynamic": True,
    }
    if args.task == "pose":
        metadata["kpt_shape"] = [17, 3]
    return metadata


def main() -> None:
    import tensorrt as trt

    args = parse_args()
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"failed to parse ONNX: {args.onnx}")

    config = builder.create_builder_config()
    config.max_workspace_size = int(args.workspace_gb * (1 << 30))
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        min_shape = (args.min_batch, 3, args.height, args.width)
        opt_shape = (args.opt_batch, 3, args.height, args.width)
        max_shape = (args.max_batch, 3, args.height, args.width)
        print(f'input {inp.name} network_shape={tuple(inp.shape)} profile min={min_shape} opt={opt_shape} max={max_shape}')
        profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f'output {out.name} shape={tuple(out.shape)} dtype={out.dtype}')

    print(f"building {args.output}")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8")) if args.metadata_json else default_metadata(args)
    meta = json.dumps(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        f.write(meta.encode())
        f.write(engine.serialize())
    print(f"saved {args.output} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
