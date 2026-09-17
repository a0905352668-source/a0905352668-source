# Gemma E4B：仅“通话姿态”条款的中英文配对试验

## 完整结果

同一冻结18事件，中英文各评测一次，共36个语言-事件结果，不是36个独立事件。只加载一次Gemma，按每事件中文→英文顺序成对执行，p07优先。

| 本轮版本 | 真实非通话手机使用12例 | 非手机工程对照6例 | p07 |
| --- | --- | --- | --- |
| 中文 | 全部KEEP、误删0 | 排除0、保留6 | native KEEP |
| 英文 | 全部KEEP、误删0 | 排除0、保留6 | native KEEP |

两版最终保留／过滤结果一致，全部18事件保留。中文非手机n01、n02、n03、n07为KEEP，n04、n09为UNCERTAIN；英文n01、n02、n03、n04为KEEP，n07、n09为UNCERTAIN。UNCERTAIN按既有保守规则保留，不算排除成功。

相比[之前含“真实手机贴耳”条件的简洁中文试验](GEMMA4_E4B_USER_PROMPT_PROBE_20260917.md)，本轮中文p07由FILTER改为第一轮KEEP，其余17例最终LABEL不变。相比[上一轮同条件英文翻译](GEMMA4_E4B_ENGLISH_TRANSLATION_PROBE_20260917.md)，本轮英文只有n07由KEEP改为UNCERTAIN，最终仍保留。脚本核对每语言提示文件，除指定的通话条款外其余核心文字逐字不变。

因此删除“真实手机贴耳”限定后，这批中文也救回p07，但中英文均未解决非手机误报。没有经确认的真实通话对照，不能证明新条款还能正确过滤真实通话；不凭这18例宣称总体准确率或直接替换Mage。原人工标签与用户确认p07为普通手机使用的真值保持不变，不更改复核库。

## 实际提示词

中文核心逐字使用用户文本：

```text
结合目标人物的连续视频画面，核实物体及使用行为。
保留：明确看到人物使用真实手机，包括查看、点击、游戏、发消息、持机瞄准或拍摄。
过滤：物体不是手机；没有看到主动使用手机；或者呈现通话姿态。
不确定：画面或时序证据不足，不要猜测。
```

英文核心保留前轮翻译措辞，只改变对应通话条款：

```text
Using the continuous video frames of the target person, verify the object and the person's usage behavior.
KEEP: The person is clearly seen using a real mobile phone, including viewing, tapping, gaming, sending messages, holding/aiming the phone, or taking photos or videos.
FILTER: The object is not a mobile phone; no active phone use is visible; or the person displays a calling posture.
UNCERTAIN: The image or temporal evidence is insufficient. Do not guess.
```

各自只追加与前轮同语言一致的严格单行LABEL输出协议，无额外业务条款。每语言native／focus／early配置同一有效提示；中文集合SHA256 db675b66f1116d3b42595953372ebfdf98ecf386f80878eef3491fe07722795e，英文集合SHA256 51299bd2a4af86766f31cc3ecf8f6acae038962a2cc65d084d9020aa98c4a6c6。源生产组合提示600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993不变。

本轮不加入“必须离耳才保留”、多帧确认通话中、手机正反面或拍屏机会判断。输出只有业务LABEL，不能从结果猜测p07此前误删的具体内部原因。自然语言表达、token序列及配对执行顺序仍有差异，不能把结果当作普遍语言优劣证明。

## 输入、审计与保护

同一18事件、54个冻结FFV1 AVI、采样索引和裁剪像素不变；发送视频逐次SHA校验。原生input_video，不拆成独立图片、不绘制框，448×448、4FPS。native20先判，只有FILTER进入focus20，再次FILTER才early8；任何KEEP或UNCERTAIN保留，三轮全部FILTER才过滤。本轮36次均native即结束，未触发两轮补看，不能称实际做了36次完整三轮评估。

沿用同Google官方Gemma E4B QAT Q4_0及mmproj、固定llama.cpp binary和内置模板：context16384、parallel1、fit off、ngl99、image min/max560、video-fps0、timestamp interval200ms、reasoning off、temperature0、max_tokens24、cache_prompt=false。不重新下载或升级依赖，不宣称重复完整权重SHA核验。

36/36响应正常stop、严格LABEL合法，720/720个预期帧解码，服务器懒回调总数也为720，位置警告记录0、无完整响应截断。中文prompt11867tokens、英文11897tokens，均小于16384。中文native18次平均6.352秒、英文18次平均6.324秒，不含冷加载或Mage恢复，不据此计算对Mage缓存HTTP返回的速度倍数。

模型加载到HTTP健康78.29秒。15:06:29停止Mage、15:11:40请求启动，监督控制311.56秒，PROBE_COMPLETE36、systemd单元Succeeded。仅切换一次GPU、等待签名调度空闲后停；finally及ExecStopPost双重恢复、脚本600秒／外部720秒、MemoryMax20GiB、CPUQuota800%，前台中止保护不改。138条监督均8在线、FPS79.5990–80.4635、alerts=[]，GPU总占用采样最高5139MiB。Mage复核暂停还包含冷启动，不能称复核完全不中断；网页／采集未重启，地址不变192.168.104.53:8767。

线上Mage提示词、规则、源码、标定及人工标签不改。第一次恢复验收60秒窗口内仍连接拒绝，随后只读日志显示模型加载／预热继续进行，没有反复重启或更改配置。15:14:23最终签名验收ready=true、内存压力false、生产及拍屏提示／证据修订保持原值；已接纳3条新production任务，有完成推理累计37.822秒、最近17.045秒，另一条正在处理，production_errors=0。网页running／8在线、79.9837FPS、alerts=[]，生产8879监听、临时Gemma18880关闭。验收保存于私有restoration-verification.json，不只检查systemd active；首次ready时间未连续测量，不把验收时间当作精确停机时长。

配对调度先观察2项测试失败，再实现确保每事件双语言、p07优先、输入不变及拒绝空／重复版本；与原视频请求／保守解析／聚合测试合计5项，本地及.54通过。这是离线临时代码，不是新生产功能，测试不能替代模型语义评估。

输入、有效提示manifest、完整响应、props、日志、监督及可重放脚本私有保存于 /Users/a1/Documents/监控/output/gemma4-callposture-pair-20260917 与.54的00_staging/gemma4_callposture_pair_20260917。Git仅保存本说明，不上传监控媒体、权重或凭证。
