# 带框视频下载候选版本 v179

基于生产 v178 复制，现有推理、规则、配置和原视频均不改变。本文件不表示该版本已经上线。

播放器右上角，全屏按钮左侧增加“下载视频”。导出的 H.264 MP4 只有人物（包含 risk 人物）和手机轮廓框，无屏幕框、射线或文字标签。原始摄像头自带时间/水印不移除。下载不受网页标注复选框影响。

当前事件 `POST /api/events/<event_id>/download`，历史事件 `POST /api/runs/<run_id>/events/<event_id>/download` 请求生成；相同地址 GET/HEAD 只查询，不生成。返回 queued/running/ready/error；ready 时返回 `/download/file`。文件路由支持 GET、HEAD、Range 和 attachment，不更改原播放路由。

只读取已录制视频和已对齐到片段时间的 JSON 标注，不连接摄像头、不重新推理。优先使用已有浏览器播放副本，按保存的原始 frame_width/frame_height 缩放；最近一组标注最多保持 0.75 秒，下一组会替换上一组。

需要服务器现有 ffmpeg/ffprobe、libass 的 ass filter、libx264。仅 CPU，编码/解码各限 2 线程、滤镜 1 线程，nice 10；单 worker，最多 4 个排队及执行中的任务；单视频 ≤60 秒、≤3840×2160；标注 ≤32 MiB、≤12000 个时间样本，每样本 ≤64 个目标，全片 ≤24000 个轮廓，展开 ASS ≤8 MiB；FFmpeg 超时 180 秒。任务失败可重试。临时目录自动清理，成功且时长完整后原子发布 MP4。

缓存放在原媒体 `dashboard/clips/boxed_downloads`，以源视频/标注的路径、inode、大小、纳秒修改/变更时间及导出版本构成键。遵守冻结的 RunStorage 权威路径，拒绝 symlink，worker 启动和发布前复验目录及源文件。每 run 缓存最多 2 GiB，单输出预算 128 MiB（编码限速及文件大小限制），生成前及执行时至少保留 20 GiB 磁盘空间，排队任务预留预算。缓存属于运行数据，不纳入 Git；当前随 run 的媒体保留策略处理，不主动删除历史缓存。容量不足时拒绝新增导出，已缓存文件仍可下载。

验证：服务器真实 FFmpeg 导出、像素位置/颜色/过期框/无屏幕及文字、跨零片头标注及部分出画人物、当前/历史路由、Range/HEAD、原视频哈希、缓存/祖先 symlink、排队后路径/磁盘/源文件变化、磁盘及缓存容量、标注展开上限、四任务并发上限、损坏标注失败及恢复，19 项通过；Node 检查轮询、切换事件取消旧下载、失败及超时、目的地址约束，4 项通过。网页最多等待约 120 秒，超时提示稍后重试，后台排队/生成继续并可复用缓存，不重新创建同一任务。旧 v178 的前端筛选结构断言已有失败，本次没有修复该历史测试。

```sh
PYTHONPATH=<pytest依赖目录>:<本release绝对目录> python3 -m pytest -q <本release绝对目录>/live_operator/tests/test_video_download.py
node --test <本release绝对目录>/live_operator/tests/video_download_frontend.test.cjs
```

部署前必须确认生产服务的事件协调器与 UI 生命周期。新增 Python 路由需要重新载入 Web 服务，不能仅替换静态文件；不允许通过重启推理容器/模型/摄像头来实现此更新。切换前保留 v178 和生产 v4 二进制、配置路径，先取得上线确认。不要把候选目录内继承的 CUDA 源码当作当前 v4 二进制的重建依据；权威运行清单仍见 `01_algorithms/LIVE_RUNTIME_MANIFEST_20260911.json`。
