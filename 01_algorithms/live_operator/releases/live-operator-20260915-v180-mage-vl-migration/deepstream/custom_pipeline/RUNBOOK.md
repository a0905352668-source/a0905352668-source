# DeepStream 7x8 Operator Runbook

This runbook covers the isolated DeepStream 7.0 candidate on Ubuntu 50.2. It
does not replace the current production executable or dashboard.

## Fixed inputs

- Project root: `/media/boshi/Data/JianKong`
- Staging root: `/media/boshi/Data/JianKong/00_staging/deepstream_7x8_20260714`
- Image: `jiankong/deepstream:7.0-samples-dev`
- Pose engine: `06_training_runs/raw_trt_plans_20260706_114432/pose960_static_b7.plan`
- Phone engine: `06_training_runs/cpp_group01_group02_latest_model_static_20260710_132056/phone_b16_img512_fp16_trt86.engine`
- RTSP sources: `rtsp://192.168.50.3:8554/camera01` through `camera07`
- GPU lock: `/tmp/jiankong_gpu0.lock`

Camera 01 deliberately uses the v21 calibration with an empty `screens`
array. The other six camera calibrations must contain active screens.

## Build and audit

```bash
cd /media/boshi/Data/JianKong/00_staging/deepstream_7x8_20260714/source
bash deepstream/custom_pipeline/scripts/build_container_50p2.sh
bash scripts/run_deepstream_container_audit.sh
```

The audit must report DeepStream 7.0, CUDA 12.2, TensorRT 8.6.1,
`PLUGINS_OK`, and successful deserialization of both engines.

## Runtime sidecar gate

Before any GPU run, call `write_live_runtime_manifest()` as shown in
`README.md` with the exact built DeepStream binary and source, then run the
CPU-only `resolve_current_inference_version()`. The generator does not modify
`CURRENT_INFERENCE_VERSION.json`; it atomically writes the mode `0600`
`/media/boshi/Data/JianKong/02_configs/runtime/live_runtime_manifest.json` and
backs up the previous file as `.json.bak`. The sidecar active version must
match the approved registry. This operation is not version promotion.

## Stream preflight

Before a formal run, verify all seven sources from 50.2. Do not start the GPU
benchmark if any source is unavailable or has a loop-boundary stall.

```bash
for n in 01 02 03 04 05 06 07; do
  timeout 15 ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate \
    -of default=noprint_wrappers=1 \
    "rtsp://192.168.50.3:8554/camera${n}"
done
```

For file-backed simulation, inspect at least one complete source loop. A common
gap across all seven publishers invalidates the formal throughput result even
when DeepStream reports zero source errors and zero reconnects.

## Diagnostic and formal runs

The host wrapper is the only process that acquires the GPU lock. Never wrap it
in a second `flock`, and never start a second benchmark while it is running.

```bash
cd /media/boshi/Data/JianKong/00_staging/deepstream_7x8_20260714/source

RUN_STAMP=diagnostic_60s \
OUTPUT_DIR=/media/boshi/Data/JianKong/06_training_runs/deepstream_7x8_diagnostic_60s \
DURATION_SEC=60 \
bash deepstream/custom_pipeline/scripts/run_container_50p2.sh

RUN_STAMP=formal_300s \
OUTPUT_DIR=/media/boshi/Data/JianKong/06_training_runs/deepstream_7x8_formal_300s \
DURATION_SEC=300 \
bash deepstream/custom_pipeline/scripts/run_container_50p2.sh
```

Acceptance requires all of the following from `live_summary.json`:

- `completed_fps >= 56.0`
- every stream `processed_fps >= 8.0`
- every stream `latency_p95_ms <= 250`
- `backlog_loss_upper_bound_total == 0`
- `source_errors_total == 0`
- `reconnects_total == 0`
- no OOM, pipeline stall, or engine fallback in the run log

Resource sampling may run outside the GPU lock because it is read-only:

```bash
nvidia-smi dmon -s pucvmet -d 5 -o DT > nvidia_smi_dmon.log
docker stats --format '{{json .}}' CONTAINER_NAME > docker_stats.jsonl
```

## Failure triage

1. Preserve the output directory, run log, `live_summary.json`, and resource
   samples. Never overwrite a failed formal result.
2. If all streams lose frames at the same timestamp while GPU utilization is
   low, inspect the 50.3 file publisher for a non-seamless loop boundary.
3. If `backlog_loss_upper_bound_total` grows while admitted FPS remains at the
   target, profile the 50.2 appsink and inference stages.
4. If only one source reports errors or reconnects, repair that publisher or
   network path before repeating the formal run.
5. Re-run the 60-second diagnostic after any code, image, engine, calibration,
   or publisher change. Run the 300-second test only after the diagnostic
   passes.

## Rollback

### Spatial static phone controls

`run_7x8.sh` enables the spatial static phone policy by default with these
values: observation `3.0` seconds, pending `0.75` seconds, long confirmation
`6.0` seconds, minimum detection ratio `0.60`, maximum gap `0.75` seconds,
position radius ratio `0.03`, and hotspot support enabled. The matching
environment variables are documented in `config/rtsp_7x8.env.example`.

Run with the approved defaults:

```bash
ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION=1 \
STATIC_OBSERVATION_SECONDS=3.0 STATIC_PENDING_SECONDS=0.75 \
STATIC_LONG_CONFIRM_SECONDS=6.0 STATIC_MIN_DETECTION_RATIO=0.60 \
STATIC_MAX_GAP_SECONDS=0.75 STATIC_POSITION_RADIUS_RATIO=0.03 \
STATIC_HOTSPOT_ENABLED=1 \
bash deepstream/custom_pipeline/scripts/run_container_50p2.sh
```

One-switch rollback bypasses the spatial policy, shadow buffer, and hotspot
state while preserving the released candidate/risk/S0-S4 path:

```bash
ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION=0 \
bash deepstream/custom_pipeline/scripts/run_container_50p2.sh
```

The equivalent direct binary switch is
`--disable-spatial-static-phone-suppression`. JSONL phone candidates and
selected person summaries contain `static_cluster_samples`,
`static_detection_ratio`, `static_center_spread_px`,
`static_bbox_iou_median`, `static_pending`, `static_pending_duration`,
`static_hotspot_score`, `static_context_reason`, `static_shadow_hits`,
`static_suppressed`, and `static_exit_reason`.

The candidate is isolated and is not promoted automatically. To roll back:

1. Stop only the candidate container, if present:

   ```bash
   docker ps --filter 'name=jk-ds70-7x8-' --format '{{.Names}}' \
     | xargs -r docker stop
   ```

2. Confirm `/tmp/jiankong_gpu0.lock` is free.
3. Resume the existing OpenCV/TensorRT production launcher unchanged.
4. Use the preserved baseline at
   `/media/boshi/Data/JianKong/00_staging/backups/rtsp_live_before_deepstream_20260714_085140`
   if a staging file must be restored.
5. Do not delete failed benchmark evidence, the derived image, model engines,
   calibrations, CVAT containers, or existing dashboards during rollback.

For sidecar-only rollback, stop the candidate, verify the GPU lock is free,
atomically restore `live_runtime_manifest.json.bak` with mode `0600`, and run
`resolve_current_inference_version()` again. Do not change the approved
registry or its `active_version`.

```bash
cd /media/boshi/Data/JianKong/00_staging/deepstream_7x8_20260714/source
PYTHONPATH=. python3 - <<'PY'
from live_operator.inference import (
    resolve_current_inference_version,
    restore_live_runtime_manifest,
)

restore_live_runtime_manifest()
version = resolve_current_inference_version()
print(version.active_version, version.deepstream_binary_sha256)
PY
```
