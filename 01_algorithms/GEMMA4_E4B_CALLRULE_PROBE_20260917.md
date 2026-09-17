# Gemma E4B：将“靠耳姿态”改为“通话中”的离线对照

## 结论与边界

用户观看p07原归档视频后明确确认：这是普通手机使用，不是通话，之前过滤确实错误。保留原confirmed/phone_use标签，不写人工复核库，不以重新标注改善结果。

本轮只改离线Gemma三种提示词的通话条件：多帧一致地支持正在通话才可按通话过滤；手机靠近脸、耳朵、耳机，或透视、头发、遮挡造成的二维重叠，不足以认定通话；看不清通话证据则UNCERTAIN并保留。任何清晰支持的离耳观看、点击、持机瞄准或拍摄仍保留。非手机／无主动使用的条件及解析、补救规则不改。

p07最终从FILTER_FALSE_POSITIVE变为UNCERTAIN，按既有保守规则保留。但是人物20帧和手部20帧仍FILTER，靠早期8帧UNCERTAIN救回；不能说模型已经正确识别其普通使用动作，也不能仅凭业务LABEL确认旧版本的错误原因一定是误认通话。

原计划18例，只完成6例、12次请求。前台状态接口一次返回unknown／0在线／0FPS，监督立即停止试验并双重恢复Mage。随后重新读取接口为running、8在线；此次不绕过保护或再次切换GPU。未完成的12例不能计入成绩。

已完成部分的结果：

| 样本 | 旧提示最终 | 新提示最终 | 新提示各轮 |
| --- | --- | --- | --- |
| p07，普通手机使用 | FILTER | UNCERTAIN，保留 | FILTER / FILTER / UNCERTAIN |
| p02，手机使用 | KEEP | KEEP | KEEP |
| n01，非手机对照 | KEEP | KEEP | KEEP |
| n02，非手机对照 | KEEP | KEEP | FILTER / FILTER / KEEP |
| n03，非手机对照 | KEEP | KEEP | KEEP |
| n07，非手机对照 | FILTER | UNCERTAIN，保留 | FILTER / FILTER / UNCERTAIN |

因此两个手机正样本均保留，但n07的非手机排除收益也丢失。未完成全批回归、未加入经确认的真实通话对照，不证明新提示能全面兼顾误删、误报或真实通话过滤。目前仅为离线候选，未改生产Mage提示词、规则、源码、标定、事件标签或网站地址。

## 输入与唯一预期实验变量

复用[上一轮对齐试验](GEMMA4_E4B_ALIGNED_PROBE_20260917.md)冻结的18例、54个FFV1 AVI及采样索引，三种视频仍native20／focus20／early8，448×448、4FPS。每个实际发送的视频校验SHA，不换帧、不改裁剪、不新增框、不把视频拆成图片请求。

沿用同一Google官方QAT Q4_0主权重、mmproj、llama.cpp binary和启动参数：context16384、parallel1、fit off、ngl99、image min/max560、video-fps0、timestamp interval200ms、Jinja、reasoning off、greedy、max_tokens24、cache_prompt=false。权重此前完整SHA核验，本轮仅复用verified.json并核对尺寸，没有重复全文件哈希。

只替换原提示中四处通话条款，三种模式全部应用；其余文字不变。离线提示集合SHA256为dd262406340f19b3cdbdee8234e863d7dc71851749f9ef7b0ec42a5cf2c6b800，源生产提示修订600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993保持原值。为优先验证用户纠正的关键样本，p07移至请求顺序第一；其余保持开发／回归、编号顺序。请求顺序存在这一差异，不能宣称所有运行条件逐项完全相同。

三种提示的新增通话标准强调：

> A call requires consistent evidence across multiple frames of holding a genuine phone to the ear while listening or speaking, with no visible non-call use.

这只是视觉上支持通话的证据标准，不能验证电话是否真实接通；证据不足按不确定保留，不猜通话。

## 审计、暂停与恢复

6个完整事件的12次响应均stop且严格LABEL合法，累计204帧与20／20／8预期一致，记录的位置警告为0。后续p03 native请求已开始但被保护取消，没有最终响应，不能记为通过。服务器总解码日志224帧包含该取消请求的20帧，与完整响应的204帧分开统计。

新提示总prompt tokens分别12117、12193、5167，均在16384上下文内，无完整响应截断。57条成功前台监督记录8在线、FPS79.5305–80.5834、alerts=[]，GPU总显存采样最高5145MiB；下一次unknown状态导致中止，该失败检查不在成功JSONL中，必须结合单元journal看，不能宣称整个试验期间所有检查正常。

14:12:06停止Mage，14:14:12请求重启，监督控制126.06秒，试验单元因保护断言退出失败而非正常完成。finally与systemd ExecStopPost均恢复jiankong-mage-vl-54.service，临时Gemma18880监听已关闭，生产签名8879恢复。复核暂停还包含Mage冷加载，不能只说暂停126秒；网页及采集未重启。一次unknown的根因未在本轮查明，不能认定是真实采集停止，也不能未经诊断弱化保护。

14:16:38的恢复签名健康检查ready=true、内存压力false，源第一阶段及拍屏第二阶段提示、证据修订保持原值；网站192.168.104.53:8767恢复running／8在线、79.8623FPS、alerts=[]。恢复后3条新production任务均已完成，推理累计71.598秒、最近20.954秒，active_kind=null、production_errors=0，确认真实生产推理恢复，而非仅检查unit active。签名检查及时间、监听端口保存在私有restoration-verification.json；未连续测量首次恢复ready时间，不把验收时间当作精确停机时长。

临时补丁工具先观察3项契约测试失败，再实现最小替换及完整模式／唯一源条款校验；与视频请求、保守解析、三轮聚合既有测试合计6项，本地及.54均通过。这不替代模型语义回归，代码为丢弃式离线spike，不纳入生产实现。

私有输入、完整响应、有效提示、日志、监督、可重放脚本及恢复验收保存在 /Users/a1/Documents/监控/output/gemma4-callrule-20260917 和 .54 的00_staging/gemma4_callrule_20260917。Git只保存此说明，不上传监控媒体、权重、服务密码或密钥。
