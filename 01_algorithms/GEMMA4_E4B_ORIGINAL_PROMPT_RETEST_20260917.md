# Gemma E4B：Mage线上原提示词复测（2026-09-17）

## 结果

用户要求Gemma使用Mage线上原提示词重新测试。直接静态读取.54当前生产源码中的NATIVE_PROMPT、FOCUS_PROMPT、EARLY_RESCUE_PROMPT，逐字比较冻结输入manifest：三者完全相同，组合SHA256为600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993。没有使用此前“通话中”变体或后来讨论的简洁版。

计划18例，仅完整完成5例、13次请求；再次因前台/api/status返回unknown／0在线／0FPS触发保护停止，不计未完成13例为通过。结果如下：

| 样本 | 最终判定 | 人物20／手部20／早期8帧 |
| --- | --- | --- |
| p07，用户确认的非通话手机使用 | FILTER，误删 | FILTER／FILTER／FILTER |
| n01，非手机对照 | KEEP，仍保留 | KEEP／未触发／未触发 |
| n02，非手机对照 | KEEP，仍保留 | FILTER／FILTER／KEEP |
| n03，非手机对照 | KEEP，仍保留 | FILTER／FILTER／KEEP |
| n07，非手机对照 | FILTER，排除 | FILTER／FILTER／FILTER |

这5例的各轮标签与[最初同提示词的完整18例试验](GEMMA4_E4B_ALIGNED_PROBE_20260917.md)均一致。原18例本就使用同一套Mage提示词，手机12例保留11例、误删p07，非手机6例排除2例；这是之前完整试验的数据，不能冒充本次完成18例。此前“通话中”离线变体把p07、n07均救回为UNCERTAIN，本次回到线上原条款后两者均恢复FILTER。

用户确认p07为普通手机使用的真值不变，没有重标或剔除难例。业务输出只有LABEL，不能从FILTER反推模型究竟误认手机、无使用还是通话。统一提示词不足以消除Gemma误删；本批小样本仍不支持替换Mage或上线自动过滤，不是模型通用排名。

## 输入和运行

复用同18例冻结视频与人工标签，native20／focus20／early8、448×448、4FPS，原生input_video请求，不拆成独立图片、不新增检测框。后台检测坐标只用于人物裁剪及手部候选区域放大，传入像素没有绘制框。p07优先第一，其余开发／回归及编号顺序不变；请求顺序与最初完整试验不同，需作为运行差异记录。

沿用Google官方Gemma E4B QAT Q4_0主权重及mmproj、固定llama.cpp binary，context16384、parallel1、fit off、ngl99、图像min/max560、video-fps0、timestamp interval200ms、内置Jinja、reasoning off、temperature0、max_tokens24、cache_prompt=false。不重新下载或升级运行依赖；只核对既有verified.json及权重尺寸，不声称重复完整权重SHA。

13个完整响应均正常stop、严格LABEL合法，完整响应对应212帧均与20／20／8预期一致，位置警告记录为0。接下来的p02 native请求已启动但被保护取消，没有最终响应、不算完成；服务器解码总计232帧包含这次取消请求的20帧。

模型加载到HTTP健康49.09秒，p07三轮约6.977／6.402／2.805秒。56条成功监督均8在线、FPS79.4699–80.3287、alerts=[]，GPU总显存采样最高5139MiB。失败的unknown检查导致退出，不在成功JSONL中；不能声称所有检查正常。无框裁剪及其他模型输入保持不变，没有变更Mage生产提示词、源码、标定或人工标签。

## 保护与恢复

14:41:52停止Mage，14:43:55请求重启，监督控制124.02秒；finally与外部systemd ExecStopPost双重恢复。试验单元因保护断言失败退出，不是正常全批完成。Mage复核暂停还包括冷加载，网页／采集未重启，地址仍192.168.104.53:8767。临时Gemma18880监听关闭，生产签名8879恢复。

本轮未修改保护逻辑，没有为完成测试忽略unknown状态或反复切换GPU。只读检查发现/api/status读取status.json失败或存储不可用时可缺失原始状态，但这次unknown具体原因未证实，不能认定真实采集停止，也不能直接认定只是文件写入竞争。随后状态读取恢复running／8在线，不替代对瞬时unknown的根因排查。

恢复签名健康检查Mage ready=true、第一阶段及拍屏第二阶段提示和证据修订保持原值、内存压力false；第一次恢复检查production_admitted=0，尚无新生产请求完成。14:45:55再次验收已接纳3条新production任务，完成推理累计51.608秒、最近21.386秒，另有一条正在处理，production_errors=0，确认已恢复真实生产复核；网页running／8在线、79.7016FPS、alerts=[]。更后一次签名检查详情保存在私有restoration-verification.json；不把验收时间当作精确首次恢复时间。

复用的请求构造／严格解析／三轮补救3项契约测试在.54通过，没有把复测称为新的TDD功能实现。输入、有效提示、完整响应、日志、监督及临时代码私有保存于 /Users/a1/Documents/监控/output/gemma4-original-retest-20260917 和.54的00_staging/gemma4_original_retest_20260917，Git仅保存本文。
