# Gemma E4B：简洁提示词英文翻译对照（2026-09-17）

## 结果

完成同一冻结18例、18次原生视频请求。12个真实非通话手机使用事件全部KEEP，p07第一轮从中文版FILTER变为KEEP，按用户真值不再误删；6个非手机工程对照仍全部保留，5个KEEP、n09为UNCERTAIN。

| 同18例 | 简洁中文提示词 | 本轮英文翻译 |
| --- | --- | --- |
| 12个手机正样本 | 保留11、误删p07 | 保留12、误删0 |
| 6个非手机对照 | 排除0 | 排除0 |

相比[前一轮完整中文试验](GEMMA4_E4B_USER_PROMPT_PROBE_20260917.md)，最终业务LABEL只变化两例：p07 FILTER→KEEP，n04 UNCERTAIN→KEEP。n04仍是非手机误报保留，不能把这项变化算作改善。其余16例LABEL不变。

因此本批英文表达救回了p07，但没有提高非手机排除能力；保留／过滤层面与Mage既有缓存基线相同，不代表逐例LABEL完全相同或模型总体优劣。未加入经确认的真实通话对照，不能推断通话过滤能力。保留原人工标签、保留p07用户确认为普通使用的真值，不写人工复核库。

本轮只能说明实际这两段表达导致本批结果不同，不证明Gemma普遍只适合英文。自然翻译的措辞及token序列也发生变化，没有多个翻译版本或独立新样本盲测，不能断言纯语言就是全部原因。只输出业务LABEL，不能杜撰模型对p07的内部解释。

## 实际英文提示词

```text
Using the continuous video frames of the target person, verify the object and the person's usage behavior.
KEEP: The person is clearly seen using a real mobile phone, including viewing, tapping, gaming, sending messages, holding/aiming the phone, or taking photos or videos.
FILTER: The object is not a mobile phone; no active phone use is visible; or a real mobile phone is held against the ear in a calling posture.
UNCERTAIN: The image or temporal evidence is insufficient. Do not guess.
```

同样只追加输出格式，英文协议为：

```text
Return exactly one line without explanation: for KEEP output LABEL=KEEP_NON_CALL_PHONE_USE; for FILTER output LABEL=FILTER_FALSE_POSITIVE; for UNCERTAIN output LABEL=UNCERTAIN.
```

没有加入“必须离耳才保留”、通话中多帧证据、手机指向屏幕或3D关系等额外业务条件。三模式均配置上述同一有效英文提示，集合SHA256为adc2a2d4fc35e25912e773d931288278befdbf6bbd2911ce200d9e1b55f59886；user_prompt.txt含末尾换行SHA256为88005b225a96522dc5dccfac1204f1b37c2074c3b55c4afb391e3f920499edf5。源生产提示修订600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993不变。

## 对齐、审计与暂停

同18例、同54个冻结FFV1视频、同采样索引与裁剪像素、同p07优先及其余开发／回归编号顺序；每次发送校验视频SHA。native20先判，只有FILTER才进入focus20，只有再次FILTER才看early8；KEEP或UNCERTAIN保留、三轮全FILTER才过滤。实际本轮所有18例native即结束，未触发focus及early，不能说做了18次三轮强制评测。

原生input_video，不拆成独立图片请求、不绘制检测框，448×448、4FPS；候选坐标只用于人物裁剪和手部放大。沿用同Google官方Gemma E4B QAT Q4_0主权重／mmproj、固定llama.cpp binary和模型模板，context16384、parallel1、fit off、ngl99、image min/max560、video-fps0、timestamp interval200ms、reasoning off、temperature0、max_tokens24、cache_prompt=false。没有升级或重新下载依赖，不声称重复完整权重SHA核验。

18/18个响应正常stop、严格LABEL合法，360/360个预期视频帧全部解码，服务器懒回调总数也360；记录的位置警告0、无完整响应截断。各请求总prompt11904tokens，小于16384上下文。native18次平均6.4325秒，p07为7.364秒，模型加载到HTTP健康35.08秒。不含Mage恢复，不与Mage缓存HTTP返回作推理速度倍数比较。

14:58:32停止Mage，15:01:04请求重启，监督控制153.0秒；PROBE_COMPLETE18且systemd单元Succeeded。等待签名调度空闲才停，finally及ExecStopPost双重恢复、脚本600秒／外部720秒、MemoryMax20GiB、CPUQuota800%，前台中止保护不改。70条运行监督均8在线、FPS79.5766–80.4061、alerts=[]；GPU总显存采样最高5139MiB。Mage复核暂停还含冷加载，不能称没有影响复核；网页和采集不重启，地址仍192.168.104.53:8767。

预检曾观察/api/status为unknown且storage.state=unavailable，之后恢复running；试验启动及运行监督均通过，不用这次恢复替代存储瞬时异常根因诊断。未为完成试验绕过保护或更改前台代码。

## 恢复与归档

线上Mage提示词、规则、源码、标定及人工标签不改。15:03:16最终签名验收ready=true、内存压力false，第一／第二阶段提示及拍屏证据修订保持原值；已接纳4条新production任务，有真实完成推理累计47.097秒、最近9.405秒，另有任务正在处理，production_errors=0。网页running／8在线、80.1626FPS、alerts=[]，临时Gemma18880关闭，生产8879恢复。健康及生产请求完成验收保存在私有restoration-verification.json，不只依赖systemd active；没有连续测量首次ready时间，不将验收时间当作精确停机时长。

输入、有效提示manifest、完整响应、props、日志、监督和临时代码保存于 /Users/a1/Documents/监控/output/gemma4-englishprompt-20260917 与.54的00_staging/gemma4_englishprompt_20260917。Git仅保存本文，不上传媒体、权重或凭证。复用的请求／解析／聚合3项契约测试在.54通过，不能替代模型语义回归或称为新的生产功能实现。
