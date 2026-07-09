# JianKong 监控视角防拍项目资料整理

整理时间: 2026-06-29
目标目录: `/media/boshi/Data/TrainData/JianKong/`

这个目录集中保存“监控视角防拍检测”的数据、算法、模型、配置、训练结果和推理结果。其他 Agent 接手时优先从这里看。

## 目录结构

```text
00_README/               总说明、目录清单、迁移记录
01_algorithms/           监控视角相关算法/训练/推理脚本
02_configs/              固定屏幕位置标定配置
03_raw_videos_and_frames/ 原始监控视频、抽帧、LabelMe 临时数据
04_labelme_datasets/     监控相关 LabelMe 标注数据
05_yolo_datasets/        监控相关 YOLO txt 训练数据
06_training_runs/        监控模型训练输出 runs
07_models/               当前可直接使用的 phone/pose 模型副本
08_inference_results/    已有监控推理结果视频和 summary
09_handoff_docs/         FacePhoneDet 项目交接文档副本
```

## 当前核心算法

主脚本:

```text
01_algorithms/predict_surveillance_phone_pose_fixed_screens.py
```

功能:

```text
监控视角 phone 检测 + YOLO pose 人体姿态 + 固定屏幕标定框
```

当前规则参数:

```text
phone_conf = 0.25
pose_conf = 0.25
kp_conf = 0.35
angle_thresh = 140.0
screen_expand = 0.20
screen_near_scale = 1.00
alert_frames = 5
occluded_alert_frames = 12
```

## 固定屏幕配置

```text
02_configs/surveillance/camera_01_screen_calibration.json
02_configs/surveillance/camera_02_screen_calibration.json
```

视角含义:

```text
camera_01: 09840 视频视角
camera_02: 112135/65512392 等同类视频视角
```

## 当前模型

phone-only 模型:

```text
07_models/phone_models/SURV_PHONE_ONLY_yolo11s_TrainData_plus_things_681_20260624_best.pt
```

pose 模型:

```text
07_models/pose_models/yolo11m-pose.pt
```

## 重要说明

1. 监控视角当前不再训练/检测 screen，screen 使用固定标定框。
2. 当前 phone 模型只检测监控视角下的 phone。
3. 监控原始视频常是 MPEG-PS 伪 MP4，播放器可能不能拖进度。
4. 推理脚本里有灰帧替换逻辑，summary 中记录 `gray_frames_replaced`。
5. 如果要继续调远距离站立拍摄，优先看 `screen_near_scale` 和连续帧策略。
6. 不建议继续只靠 2D angle 规则，角度在监控俯视视角下不稳定。

## 常用推理命令

视角 1:

```bash
/home/boshi/miniconda3/envs/yolo/bin/python \
  /media/boshi/Data/TrainData/JianKong/01_algorithms/predict_surveillance_phone_pose_fixed_screens.py \
  --pose-model /media/boshi/Data/TrainData/JianKong/07_models/pose_models/yolo11m-pose.pt \
  --phone-model /media/boshi/Data/TrainData/JianKong/07_models/phone_models/SURV_PHONE_ONLY_yolo11s_TrainData_plus_things_681_20260624_best.pt \
  --screen-config /media/boshi/Data/TrainData/JianKong/02_configs/surveillance/camera_01_screen_calibration.json \
  --video /path/to/view1.mp4 \
  --output /path/to/output_view1.mp4 \
  --pose-imgsz 1280 --phone-imgsz 1280 --pose-conf 0.25 --phone-conf 0.25 --device 0
```

视角 2:

```bash
/home/boshi/miniconda3/envs/yolo/bin/python \
  /media/boshi/Data/TrainData/JianKong/01_algorithms/predict_surveillance_phone_pose_fixed_screens.py \
  --pose-model /media/boshi/Data/TrainData/JianKong/07_models/pose_models/yolo11m-pose.pt \
  --phone-model /media/boshi/Data/TrainData/JianKong/07_models/phone_models/SURV_PHONE_ONLY_yolo11s_TrainData_plus_things_681_20260624_best.pt \
  --screen-config /media/boshi/Data/TrainData/JianKong/02_configs/surveillance/camera_02_screen_calibration.json \
  --video /path/to/view2.mp4 \
  --output /path/to/output_view2.mp4 \
  --pose-imgsz 1280 --phone-imgsz 1280 --pose-conf 0.25 --phone-conf 0.25 --device 0
```

## 原始位置备注

这些资料是从以下位置汇总而来:

```text
/home/boshi/资料整理_20260625/08_项目数据_视频图片/视频数据/监控
/home/boshi/AI_programming/FacePhoneDet/configs/surveillance
/home/boshi/AI_programming/FacePhoneDet/datasets/*monitor*
/home/boshi/AI_programming/FacePhoneDet/datasets/*surveillance*
/home/boshi/AI_programming/FacePhoneDet/runs/PhoneDet_finetune
/media/boshi/Data/TrainData/FacePhoneTrainData/runs/surveillance
/media/boshi/Data/FacePhoneDet_agent_handoff_20260629
```


## 2026-06-29 新增 raw video

用户今天新录制的视频已归档到:

```text
03_raw_videos_and_frames/raw_videos_20260629/
```

包含:

```text
20260629-机械1.mp4
20260629-机械2.mp4
20260629-电气2.mp4
20260629-软件1.mp4
20260629-软件2.mp4
```

辅助文件:

```text
manifest.txt
ffprobe_summary.txt
```

原始来源:

```text
/home/boshi/视频/
```

本次操作为复制归档，未删除 `/home/boshi/视频/` 下原文件。
