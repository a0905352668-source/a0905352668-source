# VideoChat3-4B：手机使用复核离线对照（2026-09-17）

## 结论

完成同一批18个冻结事件、18次实际视频推理。沿用Mage线上三份英文提示词、人物20帧约5秒及条件补救策略，本轮没有减少非手机误报，暂不替换Mage。

| 分组 | Mage既有缓存基线 | VideoChat3本轮严格最终结果 |
| --- | --- | --- |
| 12个真实手机使用样例 | 12 KEEP、0过滤 | 10 KEEP、2 UNCERTAIN、0过滤 |
| 6个非手机工程对照 | 6 KEEP、0过滤 | 5 KEEP、1 UNCERTAIN、0过滤 |

VideoChat3有3次漏写协议要求的 `LABEL=` 前缀：p07原始回答 `KEEP_NON_CALL_PHONE_USE`，n09同样回答KEEP，p03原始回答 `FILTER_FALSE_POSITIVE`。都正常EOS结束、没有耗尽24token，但严格解析转UNCERTAIN。p03是既有真实手机正样本，其原始回答偏向过滤，作为误判风险记录；不能因解析器兜底声称模型正确识别了全部12个真实手机，也不能把这个非法回答统计成已经发生的最终误过滤。

所有严格标签都是KEEP，没有合法FILTER，所以18例均只执行native20，实际没有触发focus20或early8补看。p03因协议异常而非合法FILTER未进入补救。本轮没有测试放宽解析后其补看能否救回。6个非手机最终过滤0，不将“不确定”当作成功排除。

第一请求p07为18.399秒，余17次平均2.787秒；18次请求平均3.654秒，包含处理器预处理和生成、不含外部视频解码，事件平均3.878秒另含解码、校验与清理。驻留推理耗时有吸引力，但缺乏非手机排除收益及协议稳定性，不能仅凭速度替换。Mage基线来自先前签名classification-only接口缓存记录，不是本轮重新推理18次，不能拿缓存HTTP时间计算加速倍数。

## 样本与输入对齐

原18例、原标签不变：p01—p12为真实手机，n01/n02/n03/n04/n07/n09为非手机工程对照；开发8、既有回归10。p07此前用户明确确认普通非通话手机使用。部分正样本来自邻近事件；非手机工程类别不是独立盲评金标准，n04仍有小物体画质歧义，本轮没有专门的真实通话负样本，不推出总体漏报率或通话泛化成绩。

复用 .54 的 `00_staging/gemma4_objectexamples_pair_20260917/evidence`。准备阶段核验54个FFV1 AVI文件SHA及864个RGB证据帧哈希，覆盖native20/focus20/early8。模型处理器全部54份视图CPU预检通过，不使用压缩浏览预览替代模型输入。

输入448×448、无绘制框、不改变裁剪、无二次缩放；视频解码为单个4D视频张量并经 `videos=[video]` 传入，不把20帧拆成20个独立图片问答。显式关闭默认2FPS采样，实际18次共360帧；每次 `video_grid_thw=[[20,32,32]]`，5组×256=1280视觉token，总prompt1550token。early8预检为2组×256=512视觉token，但没有进行实际early生成。

三份提示逐字复用Mage生产修订 `600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993`，greedy、24输出token。合法FILTER才补看，任何KEEP或UNCERTAIN保留，三轮全FILTER才过滤。协议必须完整单行LABEL及EOS，本轮18/18 EOS、15/18协议合法，3/18非法格式，没有截断。

VideoChat3使用冻结源25FPS索引和真实source_times计算各4帧组的时间标记，未重复、删减或补造帧。Mage现部署的预解码处理器仍使用合成帧序号秒数，两模型内部时间编码及视觉token预算不同；本轮对齐源像素、帧数、提示和分支，不声称内部张量或时间编码完全一致，也不把差异单独归因于权重。

后台UNCERTAIN保留不等于默认网页展示：默认列表还要求Mage pass或人工confirmed。本批既有人工标签未改、离线结果未写回，不声称实际事件从网页消失。

## 权重、环境和运行适配

使用[官方VideoChat3-4B](https://huggingface.co/MCG-NJU/VideoChat3-4B)，固定修订 `37fa901ec5913f84bc31108ebc1e60ad1903634c`，模型页标注Apache-2.0。实际约44.72亿BF16参数，没有量化。三分片完整SHA256分别为：

- 4252181336字节：`4c173466ec6a2c5e1544859021fbeac825067ccd54e80b887372d83b1c281654`
- 4288942328字节：`322af91f1ba81f290cee20d66bf0841be4826091e6e942688243b38bbc26e597`
- 403726088字节：`1c16eb013f0224c018db06148e28ee60bc1e2b47b1fffde0e311ed9debd63b90`

校验28个文件，LFS文件哈希及字节数匹配官方metadata。官方自定义代码先检查再加载，推理禁联网。权重保存在 .54 私有staging，不上传监控媒体到外部。

现有Transformers5.x在CPU处理器预检触发旧接口/严格类型校验不兼容。实验目录独立安装Transformers4.57.6、tokenizers0.22.2、huggingface-hub0.36.2，通过启动器优先加载；不升级或改写Mage既有环境。沿用Torch2.11.0+cu128，CPU线程6，权重GPU预算7GiB/CPU16GiB：视觉编码器、词嵌入、语言层0—26在GPU，27—35层及末端norm/rotary在CPU，通过Accelerate offload。验证输出头与词嵌入确实共享权重；checkpoint缺少独立lm_head是共享权重保存行为，不使用随机输出头评测。

首次显卡尝试trial-a加载45.711秒后，在官方SDPA回退路径尝试分配25GiB注意力中间量，OOM中止，完成0个分类，不算准确性结果。根因是全部20帧的打包序列用3D SDPA及大块对角mask，回退到全长密集计算，不是4B权重参数本身需要25GB。

实验性适配仅在模型进程内替换视觉attention函数：按官方4帧分组边界分别做4D SDPA、不使用跨组mask、CUDA限定Flash/efficient内核，保持全部20帧及原分组语义。CPU测试先失败，再实现后通过，验证分组不串帧及与原全长块对角数学结果一致（含不足4帧的末组）。未修改官方checkpoint源码或生产源码；不能把此适配验证当成对全部模型实现的完整正确性证明。

最终trial-c加载134.599秒，18/18推理完成，无再次OOM。GPU总占用采样最高8907MiB；PyTorch单请求峰值allocated8236646400字节、reserved8845787136字节，统计口径不同。不能推广为任意分辨率、帧数或全GPU部署都能在10GB运行。

## 前台保护、中止与恢复

三个有上限的独立尝试，非自动重试循环：

| 尝试 | 请求停Mage／恢复Mage（CST） | 结果 |
| --- | --- | --- |
| a | 17:57:15／17:58:22 | 68.50秒控制周期；模型OOM、0完成 |
| b | 18:01:16／18:01:35 | 19.79秒；网页状态unknown触发保护，尚未进入模型生成 |
| c | 18:03:56／18:07:29 | 213.21秒；完成18例、systemd Succeeded |

恢复请求时间不是首次模型ready时间，实际复核暂停还包含冷加载。第一次恢复后验证了真实生产完成31.285秒；第二次中止后复查ready及连续6次网页正常，再做一次有上限的c尝试，保护阈值未放宽。

b的保护断言是state=unknown、online=0、FPS0，不将其当作模型失败或证明摄像头真的全掉线。后续采集进程PID/启动时间未变、API恢复8在线约80FPS。源码发现存储snapshot失败会使API返回空状态，目录遍历临时文件stat竞态是待验证可能原因；没有当时完整storage响应/异常栈，不能断言根因已经证实或修复。本轮未修改状态接口、存储健康逻辑或保护条件。

c的94条有效监督全部running/8在线/alerts为空，FPS79.6230—80.4371；a的31条有效监督、b中止前8条有效监督也正常，但不能忽略b失败读数而宣称全程状态零波动。网页、采集、检测未重启，Mage复核有明确的切换及冷启动暂停。

GPU子进程由zty运行并持有同一gpu0锁；root监督复用既有独立签名健康及严格网页保护函数，不导入生产包。600秒控制上限，外部systemd720秒、MemoryMax20GiB、CPUQuota800%、KillMode=mixed、TimeoutStop45秒，finally及ExecStopPost双重恢复。c之后不再加载其他模型。

18:09:40最终签名验收：Mage ready=true、production_admitted=3、已完成推理累计50.731秒、最近9.287秒、active_kind为空、production_errors=0、memory_pressure=false。网页8在线、79.8985FPS、alerts为空。主/拍屏提示、证据修订和v31 release均未变，生产核心文件SHA仍 `83ec549bb9209018bfde16ccbf03f424a1c49000f83881a19122e99b6e5bfede`。地址仍192.168.104.53:8767。

## 留存和下一步

这是brainstorming技能下的离线可行性spike，使用TDD做输入/结果契约及注意力适配的小测试，verification-before-completion用新结果、7项测试和签名恢复验收约束结论；临时代码不作为生产功能。源代码、配置、人工标签及网页数据不回滚、不改写，Git仅保存本报告。

私有完整输入清单、模型校验、三个尝试日志及controller journal、trial-a失败证据、trial-c完整响应/时序/内存数据、恢复签名结果和重放脚本，保存于 `/Users/a1/Documents/监控/output/videochat3-probe-20260917`；原输入及Mage缓存基线位于既有 `output/gemma4-aligned-20260917/evidence` 和 `output/gemma4-objectexamples-pair-20260917/evidence`。远端实验位于 `/home/zty/YL/JianKong/00_staging/videochat3_probe_20260917`。

不直接替换。后续若继续优化，应先独立解决协议输出稳定性，再用事实诊断检查非手机样例为何被接受，并增加独立正/负样本；本轮没有追加提示词搜索、30帧或拍屏空间关系实验。
