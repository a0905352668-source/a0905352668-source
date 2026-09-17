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

## 离线拍屏诊断后备：`QWEN38-27B-Q4KM-20260917`

- 状态：仅离线实验；未替换生产Mage-VL，未启用二阶段报警过滤。推理结果见`01_algorithms/CAPTURE_OFFLINE_VALIDATION_20260916.md`的后续记录，不以安装成功代替效果验证。
- 主机：`192.168.104.54`，RTX3080 10GB；采用CPU／GPU混合卸载，不宣称全权重可装入10GB显存。
- 来源：[Qwen原模型](https://huggingface.co/Qwen/Qwen3.8-27B)、[ggml-org量化文件冻结版本](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/0669b98607d47046c7c2b3f801011d54a08cfccf)，Apache2.0。
- 仓库提交：`0669b98607d47046c7c2b3f801011d54a08cfccf`。
- SSD模型目录：`/opt/jiankong-offline/qwen3.8-27b-q4_k_m-0669b986-20260917`。
- 主权重：`Qwen3.8-27B-Q4_K_M.gguf`，18,973,870,432字节；实际SHA256复核通过：`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`。
- 视觉组件：`mmproj-Qwen3.8-27B-BF16.gguf`，931,145,888字节；实际SHA256复核通过：`de2a49866988ea272c43dc2a43ee2662c25725a717ca5b9059ea420a97f53fe8`。
- 运行时：llama.cpp提交`83078fec0db82d6b5a00d9599062c38c39145755`，CUDA11.8／sm86／MTMD_VIDEO=ON；路径`/home/zty/YL/JianKong/08_envs/llama-cpp-83078fec/build/bin/llama-server`。
- 本机二进制SHA256：`45dd78b118d27ece11c70168f7602a287fa3c1faf5e4bb0673458f743701f3c4`。
- 离线服务仅绑定`127.0.0.1:18879`；公众网页仍`192.168.104.53:8767`，前台未改为Qwen接口。
- 首轮参数：22层GPU、16384上下文、单并发、8线程、batch256／ubatch128、image-max-tokens256、video-fps0（沿用输入视频帧率）、时间标记168ms、输出192token、temperature0、enable_thinking=false。
- 输入：与Mage小外扩实验相同的实际30帧FFV1视频，三类定位框；不改为多张独立图片请求。保存输入文件哈希、原始响应、生成终止原因与原生解码帧审计。
- 运维：共享资源试验时暂停Mage复核以释放内存；独立systemd单元900秒硬超时，ExecStopPost自动请求恢复Mage，采集／网页不停止。首轮恢复请求被root所有的两个Python缓存阻断，定位并仅修正缓存所有者后重新启动；最终必须以签名健康及生产任务确认恢复，不以systemctl start成功代替。特权离线导入须预先禁用字节码写入。无额外对外端口、DNS或防火墙修改。
- 权重、编译产物、视频与实验JSON保留服务器／隔离数据目录，不纳入Git。
