# Gemma E4B：用户简洁中文提示词完整18例对照

## 结论

按用户指定提示词完成18例、20次原生短视频请求，没有新增通话判断条款。12个真实手机使用事件保留11个，p07仍三轮FILTER而被误删；6个非手机工程对照全部保留，其中n04、n09为UNCERTAIN。

| 同一冻结18例 | 原Mage英文提示词的Gemma完整试验 | 本轮简洁中文提示词的Gemma |
| --- | --- | --- |
| 手机正样本12例 | 保留11，误删p07 | 保留11，误删p07 |
| 非手机对照6例 | 保留4，排除n07、n09 | 保留6，排除0 |

本轮没有救回p07，同时丢失原来n07、n09的排除收益，不支持改用本提示词替换生产Mage或上线自动过滤。该结论只针对固定小批样本，不证明简洁提示或中文提示普遍较差。p07按用户确认的非通话手机使用真值不变；业务LABEL没有理由，不能断言其误删一定因为“通话”而不是物体／行为判断错误。

对照来源为[最初完整18例同输入试验](GEMMA4_E4B_ALIGNED_PROBE_20260917.md)，不是把[上轮保护中止的5例复测](GEMMA4_E4B_ORIGINAL_PROMPT_RETEST_20260917.md)冒充完整基线。Mage原18例基线复用了既有缓存，未在本轮重新生成推理。

## 实际提示词

人物、手部及早期补看三轮均逐字使用同一段用户文本：

```text
结合目标人物的连续视频画面，核实物体及使用行为。
保留：明确看到人物使用真实手机，包括查看、点击、游戏、发消息、持机瞄准或拍摄。
过滤：物体不是手机；没有看到主动使用手机；或者真实手机被贴在耳边，呈现通话姿态。
不确定：画面或时序证据不足，不要猜测。
```

只在末尾增加输出协议，不增加判断标准：

```text
仅输出一行，不要解释：保留输出LABEL=KEEP_NON_CALL_PHONE_USE；过滤输出LABEL=FILTER_FALSE_POSITIVE；不确定输出LABEL=UNCERTAIN。
```

源生产组合提示SHA256为600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993，保持不变。离线三模式有效提示集合SHA256为32737394f86d95ae9cda1c18e24a765eb06c8acc47f62ddd3e8bd1eea44885f4；三种提示完全相同。user_prompt.txt含末尾换行的SHA256为76d5207289de4cc948dc773e123560d2d57622fca183747e2c20462e96c2edf5，构造请求时仅去掉该文件末尾换行。

与原版相比，语言、长度、离耳使用条件及三轮上下文说明一并改变；本轮测试的是整段用户提示，不能把结果归因于某一个短语。没有偷偷给focus增加双视图说明或给early添加原版早期补救提示。

## 分样本结果与输入

手机正样本p01、p02、p03、p04、p05、p06、p08、p09、p10、p11、p12均native第一轮KEEP；p07为native FILTER／focus FILTER／early FILTER。

非手机n01、n02、n03、n07均native KEEP；n04、n09均native UNCERTAIN，按保守规则保留。只有p07触发两轮补看，因此实际请求比原版36次减少为20次；请求减少不代表准确率改善。

复用同一冻结18例及54个FFV1视频，native20／focus20／early8、448×448、4FPS、源采样索引与像素不变。p07优先第一，随后开发／回归及编号顺序，与刚才原提示复测一致、与最初完整试验存在顺序差异。每次发送核对视频SHA。原生input_video接口，不拆成图片、不新增绘制框；候选坐标只用于人物裁剪及手部放大。任一KEEP或UNCERTAIN即保留，三轮全部FILTER才过滤，规则不变。

沿用Google官方Gemma E4B QAT Q4_0主权重及mmproj、固定llama.cpp binary及模型模板；context16384、parallel1、fit off、ngl99、image min/max560、video-fps0、timestamp interval200ms、reasoning off、temperature0、max_tokens24、cache_prompt=false。不下载新模型、不变更生产依赖，不宣称重复全文件权重SHA核验。

## 审计与保护

20/20个响应正常stop、严格LABEL合法，无完整响应截断；388/388个预期帧解码，服务器懒回调总数也为388；记录的位置警告为0。native和focus总prompt tokens均11875，early4831，均小于16384上下文。

加载到HTTP健康45.08秒，native18次平均6.445秒，focus1次6.390秒，early1次2.728秒；不含Mage恢复，不以Mage缓存HTTP时间计算推理速度优势。

只切换一次GPU，等待签名Mage调度空闲后停止。14:51:13停止Mage，14:54:06请求启动，控制173.69秒，systemd单元Succeeded、完整PROBE_COMPLETE18。finally和ExecStopPost双重恢复、外部720秒、脚本600秒、MemoryMax20GiB、CPUQuota800%，未弱化前台保护。81条监督全部8在线、FPS79.5806–80.6094、alerts=[]，GPU总占用采样最高5139MiB。Mage复核暂停还包括冷加载，不能称线上复核完全不中断；网页／采集未重启。

线上Mage提示、规则、源码、标定和人工标签不改，网站地址保持192.168.104.53:8767。14:56:18最终恢复签名检查ready=true、内存压力false，第一／第二阶段提示及拍屏证据修订保持原值；已接纳2条新production任务，完成一条30.617秒、另一条正在运行，production_errors=0，确认真实线上复核恢复。网页running／8在线、80.3089FPS、alerts=[]，签名8879监听、临时Gemma18880关闭。健康及生产推理验收另存私有restoration-verification.json，不只依赖unit active；没有连续测量首次ready时间，不把该验收时刻当作精确停机时长。

输入、完整响应、有效提示manifest、props、日志、前台监督与临时代码私有保存于 /Users/a1/Documents/监控/output/gemma4-userprompt-20260917 和.54的00_staging/gemma4_userprompt_20260917；Git仅保存本文。复用的视频请求／严格解析／聚合3项契约测试本地及.54通过，不能替代模型语义回归或称作新的生产实现。
