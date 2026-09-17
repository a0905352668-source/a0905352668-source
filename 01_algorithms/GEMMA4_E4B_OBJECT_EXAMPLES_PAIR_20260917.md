# Gemma E4B：增加非手机物品例子的中英文配对试验

## 结果

同一冻结18事件，中英文各测一次，共36个语言-事件结果，不是36个独立事件。单次加载Gemma，每事件中文→英文配对执行，p07优先。用户要求的中文逐字使用，英文同步翻译，不改线上Mage。

| 本轮版本 | 真实非通话手机使用12例 | 非手机工程对照6例 | p07 |
| --- | --- | --- | --- |
| 中文 | 全部KEEP，误删0 | 排除1（n09），保留5 | native KEEP |
| 英文 | 全部KEEP，误删0 | 排除0，保留6 | native KEEP |

中文n01、n02、n03、n07最终KEEP，n04为UNCERTAIN，n09为FILTER。n07虽native FILTER，但focus KEEP，按补救规则最终保留，不能计为减少误报；n09的native、focus、early三次全部FILTER，才最终排除。英文n01、n02、n03为KEEP，n04、n07、n09为UNCERTAIN。UNCERTAIN继续保留，不算排除成功。

相比[上一轮仅“通话姿态”中英文配对](GEMMA4_E4B_CALLING_POSTURE_PAIR_20260917.md)，中文n09从不确定改为最终过滤，其余最终保留状态不变；英文n04从KEEP改为UNCERTAIN，仍全部保留。加具体物品例子对本批中文有一点改善，但不足以解决非手机误报，不能证明中文普遍优于英文，也不据此替换Mage。

p07沿用用户确认的普通手机使用真值，不改人工复核标签。本批没有经确认的真实通话对照，不能声称已验证通话过滤能力。6个非手机样例沿用工程对照分类，原人工记录仅false_positive/model_misdetect、无具体物品类别；n04尤其有画质歧义，不能当作独立盲评金标准。

## 实际提示词

```text
结合目标人物的连续视频画面，核实物体及使用行为。

保留：明确看到人物使用真实手机，包括查看、点击、游戏、发消息、持机瞄准或拍摄。
过滤：人物手中拿着的物体不是手机，而是纸张、水杯、键盘等物品；没有看到主动使用手机；或者呈现通话姿态。
不确定：画面或时序证据不足，不要猜测。
```

```text
Using the continuous video frames of the target person, verify the object and the person's usage behavior.

KEEP: The person is clearly seen using a real mobile phone, including viewing, tapping, gaming, sending messages, holding/aiming the phone, or taking photos or videos.
FILTER: The object held in the person's hand is not a mobile phone, but paper, a drinking cup, a keyboard, or another item; no active phone use is visible; or the person displays a calling posture.
UNCERTAIN: The image or temporal evidence is insufficient. Do not guess.
```

每语言只追加与上轮一致的严格单行LABEL协议，native/focus/early使用同一提示。保留输出LABEL=KEEP_NON_CALL_PHONE_USE；过滤输出LABEL=FILTER_FALSE_POSITIVE；不确定输出LABEL=UNCERTAIN。除非手机物品条款外业务条件不变；按用户排版在引言后增加空行，不能严格把变化归因于某几个词。

中文提示集合SHA256为3ab6838425563a2284cdc9e66b19d3ae56b36d57ab17eb986e3febfe31eb4bcd；英文为cad38505759c40216043f4bdaf6292f3fd2f2d8ec65543746ee42eae6ab28e4d。源生产组合提示仍为600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993。

## 输入与完整核验

18事件、54个冻结FFV1 AVI、采样索引、裁剪像素和源记录与上一轮逐字段一致，归档后54个视频SHA256再次核对通过。原生input_video，不拆成独立图片、不绘制框，448×448、4FPS；native20帧先判，只有FILTER才focus20，再FILTER才early8。任一次KEEP/UNCERTAIN保留，三次全部FILTER才过滤。

沿用Google官方Gemma E4B QAT Q4_0及mmproj、固定llama.cpp binary与内置模板；context16384、parallel1、fit off、ngl99、image min/max560（实际每帧视觉embedding576）、video-fps0、timestamp interval200ms、reasoning off、temperature0、max_tokens24、cache_prompt=false。模型/依赖不升级，不声称本轮重新计算完整权重SHA。

全部36结果完整且语言-事件键唯一：中文21请求，英文18请求，共39请求。39响应均正常stop、严格LABEL合法，无截断；768/768预期帧解码，服务器lazy bitmap回调总数也为768，位置警告0。中文native/focus为11882 prompt tokens、early为4838，英文为11918，均未超context。native18次平均中文6.373秒、英文6.265秒，不含加载/恢复，不据此比较Mage缓存HTTP速度。

复用上一轮配对调度与保守解析，未改补救逻辑。原5项契约测试本地与.54通过，归档后本地重新执行仍5项通过；契约测试不替代模型语义评测。

## 前台保护与恢复

单次Gemma加载到健康87.18秒。等待签名调度空闲后，15:27:55停止Mage，15:33:28请求重新启动；控制333.69秒，PROBE_COMPLETE36、systemd实验单元Succeeded。沿用finally及ExecStopPost双重恢复，脚本600秒/外部720秒、MemoryMax20GiB、CPUQuota800%，不放宽前台中止保护。

151条监督均running、8路在线、alerts=[]，FPS79.3521–80.4917，GPU占用采样最高5139MiB。网页及采集未重启，地址仍192.168.104.53:8767。Mage复核暂停包含模型切换及后续冷启动，不能称复核完全不中断，也不把最终验收时间当作首次ready时间。

15:35:49最终验收归档：签名model_ready=true、内存压力false、生产/拍屏提示与证据修订保持原值；已接纳3个新production任务，完成推理累计32.895秒、最近9.067秒，另一任务正在处理，production_errors=0。网页running、8在线、80.1372FPS、alerts=[]；生产8879监听，临时Gemma18880关闭。不只检查systemd active。线上Mage提示、规则、源码、标定、人工标签均未更改。

实际视频、有效提示manifest、完整响应、服务器日志、监督、恢复验收及可重放脚本仅私有保存于 /Users/a1/Documents/监控/output/gemma4-objectexamples-pair-20260917 和.54的00_staging/gemma4_objectexamples_pair_20260917。Git仅保存本说明，不上传监控媒体、权重、凭证。
