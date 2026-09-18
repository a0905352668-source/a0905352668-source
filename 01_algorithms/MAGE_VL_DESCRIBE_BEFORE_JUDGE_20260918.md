# Mage-VL：同次回答先描述再判断（2026-09-18）

## 结论

用户确认后完成5例配对离线试验，共10个事件级结果、14次实际视频生成。原提示与描述版的最终结果完全相同：n03杯子、n09纸张均仍保留，p01/p03/p07真实手机均保留。没有增加真实手机误过滤，但也没有减少这两个非手机误报；不扩展18例、不部署这一版提示。

描述版出现物体幻觉、跨补看自相矛盾及描述与标签不一致。它并未实现可信的独立观察；不能因为原始描述写得详细或某一次给FILTER，就当作安全改进。事件级离线均值从原版11.288秒增加到64.454秒，约5.7倍（不含模型加载）。此小样本、特定CPU/GPU offload耗时不是线上吞吐预测。

## 预设设计和不变项

本轮是已获确认的离线spike，先做5例，只有有收益且保留真实手机才扩大18例。使用既有冻结证据，不重标标签，不修改线上代码、提示、解析或事件数据。固定样例为n03、n09、p01、p03、p07，按该顺序分别执行baseline和describe。

两版均使用同一Mage权重与生产加载方式：压缩AWQ检查点解压后BF16 CPU/GPU dispatch，GPU权重上限3800MiB、CPU映射上限24GiB；不是另装一个模型，也不是完整GPU INT4内核推理。生产模块私有快照SHA256为 `83ec549bb9209018bfde16ccbf03f424a1c49000f83881a19122e99b6e5bfede`，手机提示修订为 `600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993`。

native20连续视频、约5秒、448²人物ROI、无绘制框；native判FILTER才补focus20，两次FILTER才补early8。任何KEEP保留，UNCERTAIN也不排除，只有三次合法FILTER才过滤。业务标准仍是“真实手机的非通话主动使用”，不是“只报警拍屏”；focus上下文/候选局部布局及early救回短暂手机的规则都保持原样。

对照输入来自原 `gemma4_objectexamples_pair_20260917/evidence/selected.json`，每次解码都检查AVI SHA、逐帧RGB SHA和帧数，处理后的像素张量SHA也记录。两臂共同执行的同一模式张量SHA全部一致。完整帧列表通过processor的video参数传入，grid帧数运行时断言通过。Mage原有合成序号时间戳问题在两臂均保持，不在本试验中修正；不声称这些是正确真实秒值，也不声称本轮另做了视觉注入hook审计。

## 提示与解析差异

baseline完整沿用生产native/focus/early三条提示和严格LABEL fullmatch，24个生成token上限。describe保留各提示正文的角色、业务保留/过滤/不确定标准，只删去原末尾“仅输出一行”的要求；前置以下文字：

```text
First describe only the target person's visibly held object, its visible features, and the interaction across the chronological video. Do not assume a phone or an alarm, and do not invent missing details. Then decide using the unchanged rules below. If the object or action remains unresolved in your description, choose UNCERTAIN rather than a definite filter.
```

正文之后追加：

```text
Write one short visible-evidence description sentence of at most 25 words. Then end with exactly one line: LABEL=KEEP_NON_CALL_PHONE_USE, LABEL=FILTER_FALSE_POSITIVE, or LABEL=UNCERTAIN.
```

describe生成上限112token，仍greedy。新解析只用于离线：要求正常EOS、实质描述、唯一且位于末尾的合法LABEL；接受空格/换行及字面量反斜杠n作为分隔。截断、多标签、无观察文字均UNCERTAIN。25词是提示要求而非解析硬门槛，模型多次超出；没有事后以长度改写标签。

这是“同一次回答先观察再分类”，并非独立的纯中性第一轮再交给另一轮：模型一开始仍能看到报警角色和业务类别。生成预算及离线解析格式也随之改变，不能把它称为只有一个变量完全隔离的实验。完整两臂六条提示留存在私有manifest。

## 逐例结果

K=KEEP_NON_CALL_PHONE_USE，F=FILTER_FALSE_POSITIVE；模式依次为native/focus/early。

| 样例 | 已知组别 | 原版逐次／最终 | 描述版逐次／最终 | 原版／描述版事件耗时 |
| --- | --- | --- | --- | --- |
| n03 | 杯子工程对照 | K／K | F→F→K／K | 8.836／113.441秒 |
| n09 | 纸张工程对照 | K／K | K／K | 8.181／78.143秒 |
| p01 | 真实手机 | F→F→K／K | K／K | 22.983／60.733秒 |
| p03 | 真实手机 | K／K | K／K | 8.227／41.360秒 |
| p07 | 真实手机、非通话使用 | K／K | K／K | 8.211／28.593秒 |

14/14生成均正常EOS且符合各自解析协议；总输入256帧，加载32.313秒。不存在把截断/非法输出当作成功过滤的问题，但格式合规不代表语义正确。

### 关键原始回答与含义

n03描述native称黑色矩形、有屏、手指交互，又说不具有典型手机形状，因此F；focus变成黑色圆柱贴耳、拇指上下手势，因此F；early再称矩形有屏且拇指点击，是离耳手机，因此K。最终杯子误报未减少，且描述没有稳定识别喝水行为。

n09描述版把白色矩形物体直接认作smartphone，声称点击/滚动，最终K，仍未排除纸张。

p01描述版native直接认出手机使用，不必再走补看，但原版通过early也正确保留；只是分支变化，不是最终准确性增益。

p03描述为粉色手机、桌旁主动使用，给K。

p07原始回答为：

```text
A woman in a yellow patterned top is holding a purple smartphone to her ear, actively using it for a call. LABEL=KEEP_NON_CALL_PHONE_USE
```

该样例已由用户确认非通话手机使用。描述“贴耳通话”与已知行为不符；即使仅依它自己的文字，按照未改变的业务规则也应该F而非K。本轮最终K没有误过滤，但这不能证明模型真的理解了通话排除标准；尤其不能直接把自由描述中“call”提取成过滤规则。

上述两例非手机是固定工程对照，不是新的独立盲评集；3例真手机也不足以估计广泛漏报率。结论仅覆盖本轮冻结材料。

## 启动失败、修正和保护

首轮a在子脚本argparse启动阶段失败：复用控制器会传 `--trial trial-describe-20260918-a`，新probe只接收 `--root`。0次模型生成、未加载离线模型，但已经请求暂停Mage，随后finally/ExecStopPost恢复。保留a原脚本、控制记录、日志，不把它计作模型结果。

按systematic-debugging追查控制器实参到CLI入口，先增加真实CLI契约测试并复现同样SystemExit，再补required trial参数和受限目录选择。与5项描述/合并测试一起6项通过；远端help也确认接口。恢复验收轮询曾因exec共享全局变量覆盖source而TypeError；改为已编译代码加独立namespace读取，未改变健康保护函数。a恢复后签名ready且真实生产最近10.783秒、0错误，8在线79.867FPS，才以全新b路径重跑。

b保持严格8在线、FPS≥76、alerts为空的保护，等当前生产复核完成再切换，使用同一GPU锁；600秒子进程/720秒systemd上限、20GiB内存及双重恢复未放宽。b正常完成，systemd Result=success、ExecMainStatus=0；199条监督记录全部合格，FPS79.1799—80.4953，GPU总占用采样最高5745MiB，最后监督415.54秒。08:56:39请求停Mage，09:03:36请求恢复，控制周期417.56秒，另含恢复冷加载。没有重启采集、检测或网页；Mage复核有测试切换及恢复冷加载暂停，不声称全程无中断。

最终新鲜签名恢复验收（09:04:12 CST）：ready=true、真实生产已完成且最近10.839秒、production_errors=0；8在线80.2369FPS、alerts空、memory_pressure=false。生产核心SHA、手机提示/证据修订及拍屏提示/证据修订均未变。GPU唯一计算进程PID2930274与生产服务MainPID一致。对外仍 `192.168.104.53:8767`。

## 留存和后续边界

完整原始文字、prompt、EOS/token数、视频/像素校验、逐次耗时、失败a、保护及恢复记录私有保存在 `/Users/a1/Documents/监控/output/mage-describe-gate-20260918`；远端使用既有 `00_staging/mage_frame30_pair_20260917` 内的日期脚本和独立trial-describe-20260918-b。Git只保存本报告，不上传媒体、权重、日志或秘密。

本轮只否定这份“同次回答附描述”的具体提示，不等于证明Mage不能从独立中性观察中获益。后续若继续，值得区分“完全不出现报警/分类标准的纯观察”与“重新看视频做业务判断”，并检查观察是否支持判定；那是不同方案，需要另设对照及真实手机安全门槛。不能直接部署未经验证的两轮过滤，也不采用事后改阈值掩盖当前结果。

brainstorming限定本轮为已确认的离线spike；TDD保护解析/合并和启动接口；systematic-debugging定位失败；verification-before-completion要求实际推理、原始结果和生产恢复证据后再下结论。没有生产实现变更。
