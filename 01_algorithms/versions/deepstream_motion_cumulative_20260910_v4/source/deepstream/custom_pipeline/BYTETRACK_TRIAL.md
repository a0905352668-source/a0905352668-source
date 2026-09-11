# ByteTrack 隔离试验记录（2026-09-09）

状态：候选实现，不是线上版本。默认仍为 legacy；未修改运行 manifest、部署指针或服务。
基线：05d11ba60195d0cf091b7bd447ba1717ae40003f（移动过滤 v2）。

## 改动范围

- `--person-tracker bytetrack` 启用逐摄像头 Kalman + LAPJV + 高低分两阶段关联。
- 不增加外观/ReID 网络，不改变 Pose/Phone 引擎、摄像头数量和推理帧率。
- 保留高分 Pose 门槛；低分框单独 NMS，只续接身份，不进入 Phone ROI 或报警证据。
- 关联结果直接携带原始检测索引，不再用另一次最近距离匹配绑定报警。
- 某帧缺少已确认高分关联时，清理对应业务历史和报警；身份可以短暂保留。
- 首帧之后新增轨迹需要下一帧确认。短暂丢失后须重新积累报警证据；这是保守取舍，可能延迟短事件。
- 未承诺完全遮挡/相同轨迹交错时不会换 ID；无外观算法仍存在此限制。

## 隔离构建

服务器目录 `/var/tmp/jk-bytetrack-20260909.STNxyq`；容器
`jiankong/deepstream:7.0-samples-dev`，限制 2 CPU / 4GB，不挂 GPU。

```sh
cmake -S candidate -B build -DEIGEN3_INCLUDE_DIR=/work/eigen-3.4.0 -DJIANKONG_BUILD_DEEPSTREAM_PIPELINE=ON
cmake --build build -j2
./build/jiankong_custom_pipeline --self-test-rules
./build/person_byte_tracker_test
ctest --test-dir build --output-on-failure
```

只编译跟踪测试可使用 `-DJIANKONG_BUILD_BYTETRACK_TESTS=ON`。
普通默认 policy-only 构建不新增 OpenCV/Eigen 依赖。

## 验证及局限

已在隔离环境执行完整候选编译及规则自测。
跟踪测试覆盖低分续接、低分禁止新建、摄像头隔离、过期身份、精确检测索引、
交错运动与检测重排、无检测时不得输出预测框、异常坐标、丢失后的报警证据重置。

一次合成 CPU 耗时：10 人 p50 0.040ms / p95 0.042ms；
50 人 p50 0.220ms / p95 0.225ms；100 人 p50 0.552ms / p95 0.743ms。
仅为跟踪调用，不含模型、额外低分解码或视频处理；不是现场准确率或整体 FPS 增益。

完整 CTest 有一项环境缺失：pipeline_contract_test 依赖
`/tools/validate_static_phone_regression.py`，本隔离快照不含该文件。
不将其标记为通过；最终完整编译及规则自测成功，其余 9 项 CTest 全部通过。
末轮 CPU 测试 p50/p95：10 人 0.040/0.042ms；50 人 0.219/0.400ms；
100 人 0.684/1.023ms，体现共享主机调度波动。

待补：用户提供跟错人事件编号后的现场回放对比；未进行生产流量 A/B，
未测长期整体资源变化，不能据合成测试声称换人问题已解决。
线上试用前需确认发布窗口及可回滚版本；本次没有上线。
