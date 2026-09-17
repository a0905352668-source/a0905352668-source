# Gemma 4 E4B：对齐生产证据的离线补测（2026-09-17）

## 结论与前次纠正

已完成18事件、36次实际短视频请求，按与Mage相同的20帧人物／20帧手部双视图／早期8帧补救规则判定。Gemma最终排除6个非手机工程对照中的2个；12个既有手机使用正样本保留11个，p07被过滤，作为误删风险处理。Mage本批12正样本全保留，但6个非手机对照也全保留。

所以前次单帧／4帧小试验不足以支持“Mage全面更强”；输入及规则对齐后Gemma确有部分排除收益，但同时出现正样本过滤风险，目前不直接替换或上线过滤。没有从2/6推出总体误报削减率，也没有从1/12推出总体漏报率。

本轮是用户确认的离线可行性spike，按脑暴技能保持试验与生产隔离；不保留为生产实现。网站、采集、源码、提示词、标定及人工标签未改，地址仍192.168.104.53:8767。权重与媒体不上传Git。

## 对齐了什么，哪些仍有差异

核实实际Mage第一阶段处理器输入为videos=[frames]：native20先判；只有FILTER_FALSE_POSITIVE才进入focus20；focus仍FILTER才看native最早8帧。任一补救KEEP即保留，任一UNCERTAIN也保留，三轮全部FILTER才过滤。Gemma完全复用这个分支规则，不强迫Mage执行生产不会触发的补看。

复用同18事件及原人工标签，开发8个、既有回归10个，不按本轮输出换样本。正样本12、非手机工程对照6；n04细小白纸标注仍有不确定性，部分正样本来自同人邻近事件，本批不是随机独立盲测。

从冻结生产包重新解码原归档event.mp4及overlay，核对review未变、原视频及overlay/review SHA、采样索引。逐像素验证每事件native20、context_focus20、native[:8]等于既有FFV1 AVI，再复制到隔离目录。54个输入AVI全部核对SHA，潜在864证据帧均核对与Mage原解码画面一致，没有选4帧或只选有手机提议的帧。

Gemma输入为原生input_video，一次请求接收一个实际FFV1短视频，不拼图、不拆成独立图片请求。容器448×448、4FPS、chronological；video-fps=0保留全部帧，timestamp interval200ms。20/20/8个源索引与像素对齐；两模型内部时间／位置编码与视觉张量不同，不能宣称内部张量完全一致。Mage直接接收帧列表，Gemma通过AVI加时间标记，两者有这个接口差异。

三种提示逐字复用当前生产NATIVE_PROMPT、FOCUS_PROMPT、EARLY_RESCUE_PROMPT，提示词修订600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993。问的是“真实非通话手机使用”业务标签，不是拍屏机会或3D关系。严格单行LABEL、greedy/temperature0、最多24输出token、无thinking。无效格式／截断保守为UNCERTAIN。

Mage基线经签名classification-only接口读取相同原归档18事件，没有写事件或人工标签；返回reviewed_at均为先前已有记录，即本轮复用了缓存，不声称重新生成18次Mage推理。模型、提示及证据版本保持当前原值，缓存结果用于标签对照，不用0.1–2.6秒HTTP返回计算速度倍数。

## Gemma运行及实际视觉预算

沿用[Google官方QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)，修订4b4a2c1d584be7264f87aac328a1bc739ce81b6c。此前已完整核验主权重和mmproj的SHA256，本次沿用未变文件及verified.json、启动前核对尺寸，没有重新下载或声称重复全文件SHA核验。

复用既有llama.cpp源码目录修订83078fec0db82d6b5a00d9599062c38c39145755的binary，未重建或更改生产依赖。模型内置规范Jinja模板，reasoning off，fit off、ngl99、context16384、parallel1、threads6、batch256、ubatch128、flash-attn on、no-warmup，仅监听127.0.0.1:18880。

image-min-tokens和image-max-tokens都设560，日志实际预处理1152×1152并输出[2560,576,1]视觉嵌入，共612次，576是取整后的实际数，不把配置560当成精确上限。native每次总prompt11992、focus12068、early5085，均小于16384上下文。此次不是只设置1120最大上限却没有确认实际预算。

日志显示43/43层卸载GPU，但仍有CPU_Mapped2730MiB及CUDA0主模型2696MiB缓冲；没有宣称所有权重都在GPU。图像数据由ffmpeg解码，懒回调审计每个请求20/20/8帧，36次共612帧。实际视觉嵌入调用数量另存私有审计摘要。

36/36响应正常stop、严格LABEL合法，无输出截断或OOM，612/612预期帧全部解码，non-consecutive token position警告为0。这比前次Qwen运行少了已知警告，仍不能用“无警告”证明模型内部语义或时序理解正确。

## 最终结果与补救作用

| 分组 | Mage保留／过滤 | Gemma保留／过滤 |
| --- | --- | --- |
| 12个既有手机使用正样本 | 12／0 | 11／1（p07） |
| 6个非手机工程对照 | 6／0 | 4／2（n07、n09） |

Gemma第一轮native在6个非手机中判FILTER5个；补看后只有n07纸质材料及n09纸片对照最终FILTER。n02包装、n03饮杯、n04细小白纸在native及focus都FILTER，但early又KEEP，因此最后必须保留，不将前两轮数字当作误报已减少。

真手机p03、p01、p09也在native及focus被否定，却在early救回。这说明补救步骤同时阻止了真事件误删和一些误报排除，不能只为了增加过滤数删掉早期补救。前次4帧误判绿色包装的p01本轮已最终保留。

p07三轮全部FILTER。其原人工类别为confirmed/phone_use，Mage为KEEP，预览可见主体手持粉色手机且靠近脸部。保留它的原正样本身份，按正样本过滤风险记录，不测试后改标签提高成绩。由于输出只有业务LABEL，不能确定Gemma是误认手机本体、误判无使用，还是把近脸姿态视作通话；通话／非通话姿态应另做事实诊断，不凭本轮LABEL杜撰原因。这足以阻止当前版本直接替换，但不是总体准确率证明。

## 延迟、保护与恢复

模型加载到HTTP健康34.18秒。native18次平均6.515秒，focus9次平均6.419秒，early9次平均2.801秒。比较对象是相同证据类型的已驻留Gemma请求，不含下载、模型冷加载与Mage恢复；Mage本轮缓存HTTP时间不具备模型速度可比性。输出预算和任务与前次单帧／4帧小试不同，不从0.5秒变化推出模型退化。

只切换一次GPU，等待签名health active_kind为空后停止Mage，检查与stop非原子仍有小竞态。独立root监督禁字节码、不导入生产包，模型由zty运行；脚本600秒、systemd外部720秒、MemoryMax20GiB、CPUQuota800%、KillMode=mixed，finally及ExecStopPost双重恢复Mage。

13:28:32停Mage，13:32:32请求启动，控制周期240.43秒，单元成功退出；实际复核暂停还含机械盘冷启动，不能称复核只中断4分钟。111次监督均8路在线、FPS79.3127–80.5201、alerts=[]，GPU总占用采样最高5139MiB，不保证绝对峰值。网页／采集未重启；未测试并行加载Mage、更高预算、更多事件或30帧。

13:35签名恢复复查Mage ready=true、已接纳6条新production请求，已完成累计73.351秒、最近9.064秒，production_errors=0、offline_errors=0、内存压力false，仍有一条production运行。第一阶段提示600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993、二阶段提示04d3ac7bf9e04bc1ebd909e38decda96f2b50bdcb27292dc96be8dc53ed2420b、证据person-nearby-screens-clean-video-timed-5s30-10s60-jpeg92-v5保持原值；网页running、8在线、79.9707FPS、alerts=[]、原地址不变，8879正常监听，Gemma18880关闭。恢复验收有真实完成推理，不只systemd active。由停Mage到本次已完成请求验收约7分钟，采集／网页持续运行；首次ready时间并非精确停机时长。

新请求构造在TDD下先观察遗漏原生视频请求的断言失败、实现后通过；复用已有保守解析／聚合并做针对性回归，3项测试本地及 .54 通过，不是全生产测试。所有54个AVI、冻结manifest、Mage缓存基线、完整Gemma响应、props、日志、前台监督、审计摘要及可重放临时代码私有保存在 /Users/a1/Documents/监控/output/gemma4-aligned-20260917 和 .54 的00_staging/gemma4_aligned_20260917，Git只保存本说明。

当前结论是“小样本中Gemma有排除收益，也有正样本过滤风险”，不是任何模型通用排名。本轮同时对齐了采样、提示、视觉预算、补救与输出长度，不能将变化归因于某个单一变量。后续可先对p07进行持机／通话事实诊断及独立正负样本回归；本轮没有执行这些额外推理或修改过滤规则。
