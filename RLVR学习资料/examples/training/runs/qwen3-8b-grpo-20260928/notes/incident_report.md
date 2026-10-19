# 事故报告：run-01 第 129–134 步，根盘写满，判题被记成 0.0

## 现象

- `logs/run-01/metrics.jsonl`：第 128 步 `critic/rewards/mean` 0.480，第 129 步 0.397，130 步 0.270，131 步 0.091，132–134 步 0.000；`critic/rewards/min` 从 −1.1 变成 0.0；`timing_s/reward` 从 25 秒左右掉到 2–7 秒。
- `logs/run-01/incident_step129-135.log`：1,139 行 `[Error] Task failed: [Errno 28] No space left on device, completion: <think>…`。这是 v0.4.1 原版 prime 管理器的写法：判题进程抛异常，`single_compute_score` 捕获后只打印这一行并返回 None，上层把这条记成 0.0。
- `rollouts/run-01/131.jsonl.gz`：256 条里 205 条 score 是 0.0，23 条 1.1，24 条 0.1，4 条 -1.1。
- 第 131–160 步 `actor/kl_loss` 从 0.050 跳到 0.061 后在 0.06 附近缓慢回落（第 160 步 0.057）；内部量 `pass_rate_given_format` 从 0.50 回落到 0.47，第 160 步 0.52 恢复。

## 原因

```
$ df -h /
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1    300G  300G     0 100% /
$ du -sh /root/ckpt/run-01/global_step_* /root/models /root/.cache /tmp/ray /var/log
 98G  /root/ckpt/run-01/global_step_64
 98G  /root/ckpt/run-01/global_step_128
 16G  /root/models
 47G  /root/.cache
 29G  /tmp/ray
 4G   /var/log
```

`trainer.default_local_dir=/root/ckpt/run-01` 在 300 GB 的根盘上；每份 checkpoint 98 GB（fp32 模型分片 33 GB 加 Adam 状态 66 GB）。第 128 步存完第二份后根盘 99%：两份 checkpoint 196 GB，模型 16 GB，pip 和 HF 缓存 47 GB，Ray 会话目录 29 GB（4 个 worker 的 stdout 里每步 256 段判分日志，30 个小时累计），系统和日志约 10 GB。最后几 GB 被 Ray 日志和判题沙箱的临时目录（每条样本一个）吃掉。

## 为什么是部分失败而不是全部

空间在零附近抖动：Ray 轮转日志、沙箱退出时删临时目录，都会腾出几 MB，个别写入能成功。第 129 步 20% 失败（51 条），130 步 45%（115 条），131 步 80%（205 条），132 步起 100%。部分失败比全部失败更坏：同一组里 0.0 和真实分数混在一起，0.0 进入组内均值和标准差，错误回答拿到正的 advantage（三条 0.1 一条 0.0 时各 +0.5）。全部失败时整组同分，advantage 为 0，反而只是浪费了这一步。

## 处理

- 15:21（第 134 步 15:20 结束、第 135 步判题开始之前）`mv /root/ckpt/run-01/global_step_64 /data/ckpt-archive/run-01/`，清 pip 缓存，`ray logs` 目录改到 /data；根盘回到 55%。第 135 步 15:34 结束，reward 0.453，恢复。
- 之后每份 checkpoint 存完 10 分钟内由 watcher 搬到 `/data/ckpt-archive/run-01/`，根盘保持在 70% 以下（`mv` 跨盘保留 mtime，所以清单里的时间戳仍是写入时间）。
- 没有回滚到第 128 步重跑：6 步里有 3 步整组同分没有 outcome 梯度，另 3 步被污染；KL 和熵项照常。评估影响：第 144 步验证 0.495，与第 128 步的 0.495 持平（都是 91 题通过），事故后的 16 步没涨；第 160 步 0.510 起回到趋势线。代价是 6 步约 1.5 小时加一段偏离。

## 带进 run-02 的改动

- `trainer.default_local_dir=/data/ckpt/run-02`，`trainer.max_actor_ckpt_to_keep=3`。
- `reward_model.reward_manager=prime_plus`：判题故障先重试，仍失败整组作废。run-02 第 77、212、288 步各遇到 1–2 次沙箱启动失败，第 212、288 步各作废 1 组（16 条），曲线上看不出来。
- 监控加 `df` 告警（85%），Ray 会话目录放到 /data。
