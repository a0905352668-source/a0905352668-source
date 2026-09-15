# JianKong Conservative DeepStream Custom Pipeline

Operational commands, acceptance checks, failure triage, and rollback are in
[`RUNBOOK.md`](RUNBOOK.md). Recorded qualification and benchmark evidence is in
[`RESULTS_20260714.md`](RESULTS_20260714.md).

This candidate preserves the production algorithm and uses DeepStream only for
video transport:

```text
nvurisrcbin x7
  -> per-source PTS phase gate at 8 FPS
  -> nvstreammux, 7 x 2560x1440, CUDA device memory
  -> nvvideoconvert RGBA NVMM
  -> appsink
  -> existing TensorRT pose and phone runners
  -> existing decode, tracking, screen rules, static suppression and events
```

Live mode rejects any inference rate other than exactly 8 FPS. The canonical
camera01 calibration explicitly contains zero screens; that source still runs
transport, pose, tracking and throughput accounting, but it does not construct
Person ROIs or invoke phone inference and cannot enter the alarm path.

It does not replace the production source or the experimental
`deepstream/app`. The copied algorithm source remains independently reversible.

## Why the mux stays at 2560x1440

The screen calibration polygons and all rule geometry use original camera
coordinates. A 960x960 mux would silently change that coordinate system. The
pose engine still receives a 960x960 letterboxed tensor from the CUDA kernel.

## Local CPU verification

```bash
cmake -S deepstream/custom_pipeline -B /tmp/jiankong-custom-cpu
cmake --build /tmp/jiankong-custom-cpu --config Release
ctest --test-dir /tmp/jiankong-custom-cpu -C Release --output-on-failure
```

The CPU golden test verifies RGBA channel order, pitched rows, symmetric
letterbox padding and cropped ROI addressing. It does not validate NVIDIA
headers, CUDA execution or a live RTSP pipeline.

## DeepStream 7.0 container build

On Ubuntu 50.2, use the host wrapper. It builds the local derived image from
`nvcr.io/nvidia/deepstream:7.0-samples-multiarch`, mounts the canonical
`/media/boshi/Data/JianKong` inputs, and writes the binary to the persistent
staging build directory:

The derived image installs `cuda-nvcc-12-2` because the live candidate builds
CUDA translation units; the container audit verifies `nvcc` before compilation.

```bash
bash deepstream/custom_pipeline/scripts/build_container_50p2.sh
```

Run the seven-stream benchmark with:

First generate the private runtime sidecar after building the exact binary:

```bash
cd /media/boshi/Data/JianKong/00_staging/deepstream_7x8_20260714/source
PYTHONPATH=. python3 - <<'PY'
from live_operator.inference import write_live_runtime_manifest
write_live_runtime_manifest(
    "/media/boshi/Data/JianKong/00_staging/deepstream_7x8_20260714/build/custom_pipeline/jiankong_custom_pipeline",
    "deepstream/custom_pipeline/src/jiankong_custom_pipeline.cu",
)
PY
```

This leaves `CURRENT_INFERENCE_VERSION.json` unchanged, atomically writes the
mode `0600` `02_configs/runtime/live_runtime_manifest.json`, and backs up the
previous sidecar as `.json.bak`. It is runtime binding, not version promotion.

```bash
DURATION_SEC=300 bash deepstream/custom_pipeline/scripts/run_container_50p2.sh
```

Without explicit binary variables, the wrapper authenticates the exact binary
from that sidecar and checks its active version against the approved registry.
It never falls back to an unauthenticated staging binary.

The host run wrapper is the only layer that acquires
`/tmp/jiankong_gpu0.lock`. `scripts/run_7x8.sh` is container-internal and must
not acquire the same lock. It validates both engines, the writable output
directory, and all seven v21 calibration files before launching. Camera 01 is
accepted only with an explicit empty `screens` array; the other six views use
their normal screen calibrations from
`/media/boshi/Data/JianKong/02_configs/surveillance`.

`live_summary.json` reports aggregate `completed_fps`, each stream's processed
FPS, phase-gate drops and target attainment. Reconnect counts are explicitly
marked unavailable until the reader has source-specific bus error accounting.

The launch script uses the production view order required by the calibration
router: `dianqi1`, `dianqi2`, `jixie1`, `jixie2`, `ruanjian1`, `ruanjian2`,
and `zoulang` for `camera01` through `camera07` respectively.

Current metric boundaries are intentional: `captured` means admitted by the
per-source 8 FPS PTS gate, `dropped` counts only phase-gate drops, and latency
starts when the appsink batch is received. Appsink overload drops, reconnect
counts and sender-to-result end-to-end latency are not yet observable.

## Required container verification

1. Compile against actual DeepStream 7.0 and TensorRT 8.6 headers.
2. Confirm `NvBufSurface` is `CUDA_DEVICE`, pitch-linear RGBA.
3. Compare pose and phone detections against the backed-up production binary on
   identical frames.
4. Run 30-second smoke, 60-second stability and 300-second formal benchmarks.
5. Do not call the candidate complete until all seven streams reach at least
   8 FPS and aggregate completed FPS reaches 56 without appsink overload drops.
6. Exercise RTSP disconnect/reconnect and confirm one failed source does not
   terminate the other six streams before promoting the candidate.
