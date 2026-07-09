# JianKong 监控防拍项目代码仓库

本仓库用于管理 JianKong 监控视角防拍系统的源码、配置文件、测试代码和项目说明文档。

## 仓库管理范围

当前纳入 Git 版本管理的内容包括：

- `01_algorithms/`：Python 推理脚本、TensorRT/C++ 推理管线、实时复盘页面、预标注工具、训练辅助脚本和测试代码。
- `02_configs/`：各摄像头视角对应的屏幕标定配置文件。
- `00_README/`：项目资料整理说明和交接说明。
- `07_models/MODEL_REGISTRY.md`：模型版本注册表，仅记录模型来源、路径、版本和指标，不保存模型权重本体。

## 不纳入 Git 的内容

以下内容体积大、变化频繁，或不适合上传到 GitHub，因此保留在 Ubuntu 服务器本地：

- 原始监控视频、抽帧图片和复盘视频；
- LabelMe 标注数据和 YOLO 训练数据集；
- 训练输出、推理输出、前端 dashboard 生成结果；
- `.pt`、`.onnx`、TensorRT `engine/plan` 等模型文件；
- 编译后的 C++/CUDA 可执行文件；
- 日志、CSV、PID、缓存文件和临时备份文件。

服务器上的完整运行数据仍位于：

```text
/media/boshi/Data/JianKong/
```

## 当前主线说明

当前系统主线是：

```text
YOLO11s-pose 人体姿态检测
→ person tracking
→ Person ROI 内手机检测
→ phone 映射回原图
→ 固定屏幕标定 + 手腕关键点 + 姿态 + 时间连续性规则
→ 事件复盘页面展示
```

当前默认工程约定：

- 推理默认使用 TensorRT/C++ 主链路；
- 常规目标帧率按 8~10 FPS 设计；
- 姿态模型输入通常使用 960；
- 手机检测模型输入通常使用 512 或 640，具体以模型注册表为准；
- 更换摄像头视角时必须使用对应的 `camera_*_screen_calibration_v21.json`；
- GitHub 仓库只管理代码和配置，不直接保存数据集和模型权重。

## 常用位置

```text
01_algorithms/                                  核心算法代码
01_algorithms/realtime_sim/                    实时模拟与复盘页面服务
01_algorithms/tools/                           数据处理、预标注、C++ 推理工具
02_configs/surveillance/                       屏幕标定配置
07_models/MODEL_REGISTRY.md                    模型版本注册表
```

## 版本管理要求

后续代码、配置、说明文档发生修改后，应及时提交并推送到 GitHub，便于回滚和追踪。提交前需要确认：

1. 不提交原始视频、图片、标注数据、训练输出和模型权重。
2. 不提交 GitHub token、服务器密码、个人密钥等敏感信息。
3. 修改说明要写清楚本次变更目的。