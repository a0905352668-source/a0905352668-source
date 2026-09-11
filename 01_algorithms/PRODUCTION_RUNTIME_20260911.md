# 2026-09-11 生产运行代码快照

本文件记录 2026-09-11 在 50.2 上实际运行的防拍系统代码来源。Git 中只保存可复现源码、测试和启动脚本；模型、TensorRT 产物、摄像头凭据、运行日志、视频和编译二进制不纳入版本管理。

## 当前运行组合

- 运行目录：`/media/boshi/Data/JianKong/08_inference_results/live_20260911_070542`
- Python `PYTHONPATH`：`/media/boshi/Data/00_active_projects/JianKong/01_algorithms/live_operator/releases/live-operator-20260909-v178-hot-metadata`
- Git 内 Python 快照：`01_algorithms/live_operator/releases/live-operator-20260909-v178-hot-metadata`
- DeepStream 运行二进制：`/media/boshi/Data/JianKong/01_algorithms/versions/deepstream_motion_cumulative_20260910_v4/bin/jiankong_custom_pipeline`
- Git 内 DeepStream 源码：`01_algorithms/versions/deepstream_motion_cumulative_20260910_v4/source`
- 容器镜像：`jiankong/deepstream:7.0-samples-dev`
- 容器镜像 ID：`sha256:e2739c816593a50b8c3a88533066e089c67d5efbc8a2a07eee5d2aea32f338b2`
- 推理帧率：10 FPS

## 关键文件与运行产物校验

| 对象 | SHA256 |
|---|---|
| `live_operator/cli.py` | `e986e0f12067f285b1ff823a29fd14c1760e9a80da7a50a254c24a5a090e61e8` |
| `live_operator/dashboard.py` | `7a700a06239005aa2b46e87805cc3935f31b93319e6526c302cff04e7282e9b6` |
| `live_operator/vlm_state.py` | `2c009d526d537e7847625c388109784abb5c7acb9ada9e5ea996be8acd5dbcca` |
| `live_operator/static/app.js` | `31a96f5296aeedaf42064839eec24b7fd2da2ae7d415cd483be5afff95a38014` |
| v4 `src/jiankong_custom_pipeline.cu` | `6f528437b2c0f0b93463dcaaf7a7eb13900fe35043d8c6bc85b7bc0a7f76dec3` |
| 当前 DeepStream 二进制（不提交） | `8dc78e1a97736716e6f53ac5a3a92eb2b1e5b8221aa513e1f27aff889b281cab` |
| 当前 pose plan（不提交） | `1222798f3ee58499474a52d87276179dc1a970c05a570cde7e85280501ad8cda` |
| 当前 phone engine（不提交） | `33fa0d269679985e454d5dec09e3579b32ec531570e67e422282d7018c64464c` |
| 当前运行配置（只记录哈希，不提交内容） | `e198f4c3beaf54ebff8d08647d49aa93bc879d228a50387d2e36ecc7d940b5cf` |

## 同步边界

已同步：

- 当前 v178 Python 服务、前端静态资源、单元测试和启动脚本；
- v178 发布包中的 DeepStream 源码快照；
- 当前 v4 二进制对应的 DeepStream/CUDA 源码、第三方 ByteTrack 源码、测试和两项静态手机回归辅助工具；
- v4 的升级前来源清单 `manifest-before.json`。

明确排除：

- `bin/jiankong_custom_pipeline` 等编译产物；
- `.plan`、`.engine`、`.pt` 等模型文件；
- `02_configs/runtime/live_operator.json`，因为它包含摄像头访问凭据；
- `__pycache__`、`.pytest_cache`、macOS `._*` 文件和发布目录内的临时备份；
- 推理输出、录像、事件素材和其他运行数据。

## 已发现的版本债务

- `01_algorithms/CURRENT_INFERENCE_VERSION.json` 仍描述 2026-08-24 的动态批次版本，与当前实际运行的 v178 Python + 2026-09-10 v4 DeepStream 组合不一致，因此本次不把它作为生产事实提交。
- 当前 v4 源码包最初缺少合同测试引用的两个 `tools` 文件；本快照采用演进链中 2026-08-22 及多个后续副本哈希一致的版本恢复该依赖。
- v178 的 Python 测试与当前生产配置存在漂移：在 Python 3.13 + pytest 8.4.2 下共 316 项，271 项通过、45 项失败。失败集中在旧的 7/10 路摄像头假设、绝对生产配置路径、缺失安装脚本以及依赖真实 FFmpeg/标定环境的测试。本次只归档真实运行代码，不改变线上行为来掩盖这些失败。

## 验证结果

- 原仓库基线：37 项标准库单元测试通过；
- v4 DeepStream：10 个 CMake/CTest 目标全部通过；
- v4 管线合同：34 项全部通过；
- Python 生产源码：`compileall` 通过；
- 同步后目录与来源目录的 rsync 校验无差异；
- 敏感配置、模型和编译产物未进入 Git 同步范围。
