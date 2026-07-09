# JianKong 模型注册表

本文件用于记录当前监控防拍项目可复现、可交接的模型与推理链路版本。这里只记录版本、路径、指标和来源，不保存模型权重本体。

## 当前推理主链路：`JK-INFER-20260708-SUSPECTTRACK-v1`

- 用途：监控防拍推理，采用 Person ROI 手机检测、固定屏幕标定和嫌疑人跟踪输出。
- 运行方式：C++ TensorRT 推理管线，目标 `10 FPS`，姿态输入 `960`，手机输入 `512`。
- C++ 可执行文件：`/media/boshi/Data/JianKong/01_algorithms/tools/cpp_full_pipeline_bench_gpu_novideo`
- C++ 版本号：`JK-CPP-SUSPECTTRACK-20260708-170048`
- C++ 文件 sha256：`d79c452fa5d1a3e08b869bde44d7b789a7aa992f0ca470e3dd886e8571172acf`

## 姿态模型

- 版本号：`JK-POSE960-TRT-B7-20260706-114432`
- TensorRT plan：`/media/boshi/Data/JianKong/06_training_runs/raw_trt_plans_20260706_114432/pose960_static_b7.plan`
- Batch：静态 `7`
- sha256：`dd3fc8c45281355dc29c29156f2c35a862ead3400a7be9fbe7d21c748427b044`

## 手机检测模型

- 版本号：`JK-PHONE512-EFFECTIVENET07-TRT-B16-20260708-084750`
- 源 PT：`/media/boshi/Data/JianKong/07_models/phone_models/PERSON_ROI_PHONE_yolo11s_512_ALL_effectiveNet07_lowLR_weakaug_phone0_20260707_best.pt`
- 源 ONNX：`/media/boshi/Data/JianKong/07_models/phone_models/PERSON_ROI_PHONE_yolo11s_512_ALL_effectiveNet07_lowLR_weakaug_phone0_20260707_best.onnx`
- TensorRT plan：`/media/boshi/Data/JianKong/06_training_runs/raw_trt_plans_20260708_phone512_effectiveNet07/phone512_effectiveNet07_static_b16.plan`
- Batch：静态 `16`
- TensorRT：`8.6.1`
- 精度：FP16
- sha256：`ed43b62446940a4c7f56c8e8385e6675ff494d344e34c640c21b58718adafbb9`

## 屏幕标定

- 版本号：`JK-SCREEN-CALIB-V21-20260706-103319`
- 配置目录：`/media/boshi/Data/JianKong/02_configs/surveillance/recalibration_20260706/generated_v21_from_labelme_20260706_103319`
- 使用规则：每个摄像头视角必须使用自己对应的 `camera_*_screen_calibration_v21.json`，不能跨视角复用。

## `v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5`

- 日期（date）：2026-07-09
- 任务（task）：Person ROI 内手机检测。
- 模型（model）：YOLO11s detect，单类别 phone，从 `v20260708_02 best.pt` 继续训练。
- 训练主机（training_host）：`192.168.50.5`，双 RTX 3080。
- 数据集（dataset）：自有 `ALL` 数据权重 1.0 + 网络 `PersonROI_Phone` 数据确定性采样 0.7；网络数据仅使用 class 0 phone。
- 规范路径（canonical_path）：`/media/boshi/Data/JianKong/07_models/phone_models/versions/v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5`
- best.pt（best_pt）：`/media/boshi/Data/JianKong/07_models/phone_models/versions/v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5/best.pt`
- last.pt（last_pt）：`/media/boshi/Data/JianKong/07_models/phone_models/versions/v20260709_02_yolo11s512_net0p7_own1_continue_v02_lowFP_50p5/last.pt`
- 训练轮数（epochs）：78
- mAP50 最佳轮次（best_mAP50_epoch）：37
- 最佳 mAP50（best_mAP50）：0.89543
- 最佳 mAP50 对应 precision（best_mAP50_precision）：0.91235
- 最佳 mAP50 对应 recall（best_mAP50_recall）：0.82422
- mAP50-95 最佳轮次（best_mAP50_95_epoch）：43
- 最佳 mAP50-95（best_mAP50_95）：0.65301
- 最后一轮（last_epoch）：78
- 最后一轮 precision（last_precision）：0.8885
- 最后一轮 recall（last_recall）：0.84195
- 最后一轮 mAP50（last_mAP50）：0.88369
- 最后一轮 mAP50-95（last_mAP50_95）：0.64908