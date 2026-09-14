# 2026-09-14 视频下载上线记录

2026-09-14 16:56（北京时间）上线后验证完成。生产 `live_operator/current` 已指向 `releases/live-operator-20260914-v179-boxed-download`，安装内容来自提交 `d4b2f647300e16002f572a58c25b3d682972e41b`（84 个文件逐一匹配 Git blob）。候选 release 内的说明是发布前快照；本记录确认其实际上线状态。

按钮显示“↓ 下载视频”，位于播放器右上角全屏按钮左侧，同一行。下载的 H.264 MP4 仍只烧录人物框（含风险人物）和手机框，不包含屏幕框、射线、检测文字标签。摄像头自带时间和水印保留。导出使用 CPU，不重新推理，不覆盖原视频或播放副本。

## 发布范围

仅重载 `_services`（网页/事件服务），并原子切换 current。没有运行 full restart，没有重启 Docker 推理、GPU、模型、摄像头或视频转发。

- 当前 run：`live_20260914_070043`。
- mediamtx PID 11641、deepstream launcher PID 11699 从发布前到验证后不变。
- 新 services PID：767653；状态更新仅替换 services 的 owned identity。
- watchdog 已恢复 active，HTTP health 为 healthy。
- 实测 8/8 路在线，总计约 80 FPS，所有摄像头 source_errors 为 0。
- 生产配置 SHA256：`e198f4c3beaf54ebff8d08647d49aa93bc879d228a50387d2e36ecc7d940b5cf`。
- 生产 v4 二进制 SHA256：`8dc78e1a97736716e6f53ac5a3a92eb2b1e5b8221aa513e1f27aff889b281cab`。

配置和二进制均未改变。原 `LIVE_RUNTIME_MANIFEST_20260911.json` 仍用于推理二进制/配置的来源追溯；UI/services 的 current 版本以本记录及实际 symlink 为准。不要用 v179 继承的历史 CUDA 源码重建生产 v4。

## 验证证据

- 下载专项：服务器真实 FFmpeg 19 项通过；前端 Node 4 项通过；相关 Range/路径安全/前端独立性回归 11 项通过。未声称继承版本的整个历史测试集通过。
- 实际生产事件：POST 返回 202 queued，轮询 ready；文件 HEAD 200、attachment；Range 206，MP4 文件头正确。
- 实测导出：H.264、1920×1080、20 秒、4,338,925 字节；导出前后原 MP4 与 browser 播放副本 SHA256 相同。
- 浏览器真实点击按钮触发 download 事件；完成后按钮恢复“↓ 下载视频”且可点击。DOM 布局实测下载按钮在全屏左侧且同一行。

## 发布过程中发现的原有服务生命周期约束

早期发布尝试失败，造成网页/事件服务短暂中断；始终保留原推理和转发身份，失败恢复回到 v178 services 后再重试，最终切换成功。

1. `stop_process` 的身份检查只比较 start token，可能将 zombie 视为存活；自己的新子进程必须保留 Popen 并 poll/wait 回收，所有停止路径都需有界等待身份消失。独立真实测试复现 stop 后 state Z / identity true，poll 回收后 identity false。
2. `probe_port` 使用不带 SO_REUSEADDR 的独占 bind；HTTP TIME_WAIT 可以使没有 listener 的端口仍被拒绝。最终流程等待原探针最多 75 秒，不绕过端口或未知进程门禁。
3. 原 worker 收到 stop marker 后最多 8 秒退出协调线程。撤销 marker 不能恢复已经退出的线程；一旦发出 block 请求，异常恢复必须只停止/回收当前 owned services 并启动 v178，再确认 readiness。
4. drain 的 8 秒超时仅在其明确 LifecycleError 分支额外等待本次 fresh 握手最多 30 秒：worker owner 匹配、accepting false、时间戳有限且不早于本次请求、active 为非负整数；等待期间持续核对三组件身份。未确认则恢复旧服务，不发布。

watchdog 单元必须硬核对 `KillMode=process`、ExecStop 为空。执行身份保持 boshi、原 Python 3.13、原配置及基本/VLM 环境；只有安装、精确 symlink 切换、systemctl 使用 sudo。外层 EXIT trap 始终恢复 watchdog。现有生产脏修改和本地旧项目未回滚或清理。
