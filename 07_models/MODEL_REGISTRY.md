# JianKong Model Registry

## Active Inference Stack: JK-INFER-20260708-SUSPECTTRACK-v1

- Purpose: surveillance anti-phone-recording inference with Person ROI phone detection, fixed screen calibration, and suspect-track output.
- Runtime: C++ TensorRT pipeline, `10 FPS`, pose input `960`, phone input `512`.
- C++ binary: `/media/boshi/Data/JianKong/01_algorithms/tools/cpp_full_pipeline_bench_gpu_novideo`
- C++ binary version: `JK-CPP-SUSPECTTRACK-20260708-170048`
- C++ binary sha256: `d79c452fa5d1a3e08b869bde44d7b789a7aa992f0ca470e3dd886e8571172acf`

## Pose Model

- Version: `JK-POSE960-TRT-B7-20260706-114432`
- TensorRT plan: `/media/boshi/Data/JianKong/06_training_runs/raw_trt_plans_20260706_114432/pose960_static_b7.plan`
- Batch: static `7`
- sha256: `dd3fc8c45281355dc29c29156f2c35a862ead3400a7be9fbe7d21c748427b044`

## Phone Model

- Version: `JK-PHONE512-EFFECTIVENET07-TRT-B16-20260708-084750`
- Source PT: `/media/boshi/Data/JianKong/07_models/phone_models/PERSON_ROI_PHONE_yolo11s_512_ALL_effectiveNet07_lowLR_weakaug_phone0_20260707_best.pt`
- Source ONNX: `/media/boshi/Data/JianKong/07_models/phone_models/PERSON_ROI_PHONE_yolo11s_512_ALL_effectiveNet07_lowLR_weakaug_phone0_20260707_best.onnx`
- TensorRT plan: `/media/boshi/Data/JianKong/06_training_runs/raw_trt_plans_20260708_phone512_effectiveNet07/phone512_effectiveNet07_static_b16.plan`
- Batch: static `16`
- TensorRT: `8.6.1`
- Precision: FP16
- sha256: `ed43b62446940a4c7f56c8e8385e6675ff494d344e34c640c21b58718adafbb9`

## Calibration

- Version: `JK-SCREEN-CALIB-V21-20260706-103319`
- Directory: `/media/boshi/Data/JianKong/02_configs/surveillance/recalibration_20260706/generated_v21_from_labelme_20260706_103319`
- Rule: every view must use its matching `camera_*_screen_calibration_v21.json`.

## v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5
- date: 2026-07-09
- task: Person ROI phone detection
- model: YOLO11s detect, phone-only class, continued from v20260708_02 best.pt
- training_host: 192.168.50.5 build, dual RTX 3080
- dataset: ALL own data weight 1.0 + network PersonROI_Phone deterministic 0.7 sample, network class 0 phone only
- canonical_path: /media/boshi/Data/JianKong/07_models/phone_models/versions/v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5
- best_pt: /media/boshi/Data/JianKong/07_models/phone_models/versions/v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5/best.pt
- last_pt: /media/boshi/Data/JianKong/07_models/phone_models/versions/v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5/last.pt
- epochs: 78
- best_mAP50_epoch: 37
- best_mAP50: 0.89543
- best_mAP50_precision: 0.91235
- best_mAP50_recall: 0.82422
- best_mAP50_95_epoch: 43
- best_mAP50_95: 0.65301
- last_epoch: 78
- last_precision: 0.8885
- last_recall: 0.84195
- last_mAP50: 0.88369
- last_mAP50_95: 0.64908
