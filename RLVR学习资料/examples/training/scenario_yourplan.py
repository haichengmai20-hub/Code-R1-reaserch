"""「你的方案」推演：按带学时用户在 05 七环里做的决定生成两轮训练的产物（your-01 诊断 → your-02 修 → held-out 验证）。

复用 scenario_qwen3_8b.py 的合成器、真实判分和评测模型，只换假设。2026-10-19 起去掉 TACO（本机没有 TACO 数据，
判分块和 rollout 做不出来；LeetCode 去污染版 1889 题在本机是真的），保留 your 方案里比参考更合理的几处：
回答上限 6144、prime_plus 从第一轮就用、checkpoint 一开始就放数据盘、mini-batch 4。

- your-01（第 1–200 步）：4×A800 80GB；起点 Qwen3-8B-Base（格式靠系统提示学）；LeetCode 去污染版 1889 题；上限 6144，n=8，
  16 题/步，200 步（1.7 个 epoch）；mini-batch 4 题（每步 4 次更新）；prime_plus；lr 5e-7，low_var_kl 0.001，熵系数 0；
  验证每 16 步贪心（v2 test 200 题，也是原定的最终评测集）；checkpoint 放数据盘全留。
  事件：第 65–66、129–130、193–194 步存 checkpoint 后数据盘写回饱和，沙箱启动超时被 prime_plus 判为判题故障，重试后仍有 2–4 组作废。
- 第 200 步后（第六环）补 LCB 142 题 held-out；（第七环）用第 200 步 checkpoint 对 1889 题采样 8 条/题筛题：去掉全对、全错，
  再切 200 题当新验证集，其余续训。
- your-02（第 201–400 步，从 your-01 第 200 步续训）：判题临时目录改 tmpfs（事故消失）；筛题后的数据；验证集换成切出的 200 题；
  dev 与 LCB 只在第 400 步离线评一次。clip_ratio_high 0.28 曾在计划里，2026-10-19 去掉：它对着的证据（pass@8 不动）在 Base 起点下不存在，熵留作观察项。
产物：两轮的 metrics / validation / 完整控制台 / wandb / checkpoint 清单 / rollout 落盘（候选代码是参考解或经真实判分代码验证会失败的变异体）、
三个模型的评测明细、归因、曲线。控制台里没有逐样本判分块：prime_plus 不打印它们（与参考 run-02 相同）。
用法：$PY -B RLVR学习资料/examples/training/scenario_yourplan.py（venv 里有 matplotlib 才画 PNG）
"""
import datetime as dt
import gzip
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenario_qwen3_8b as ref  # noqa: E402

OUT = HERE / "runs" / "qwen3-8b-grpo-yourplan-20261018"
RUNS = ("your-01", "your-02")
N_PROMPT, N_SAMPLE = 16, 8
BATCH = N_PROMPT * N_SAMPLE
ref.N_SAMPLE, ref.BATCH = N_SAMPLE, BATCH          # 复用参考的抽样与落盘函数时按 n=8 分组
MAX_LEN = 6144
STEPS = {"your-01": (1, 200), "your-02": (201, 400)}
T0 = {"your-01": dt.datetime(2026, 10, 18, 8, 0, 0), "your-02": dt.datetime(2026, 10, 20, 14, 0, 0)}
WANDB_ID = {"your-01": "k2m9p4xa", "your-02": "p7d3v2mq"}
N_SOURCE, N_VAL = 1889, 200                        # 去污染版训练集；your-02 切出的验证集
PROMPT_TOK = ref.PROMPT_TOK_MEAN                   # 312
# 每份 checkpoint（98 GB）落到数据盘后，页缓存写回把盘吃满 3–4 分钟；判题临时目录与它同盘，
# 沙箱启动超过 10 s 被 prime_plus 记为判题故障：重试一次，仍失败的整组作废。随存盘步周期性出现（your-02 改 tmpfs 后消失）。
IO_EVENT = {65: 0.12, 66: 0.08, 129: 0.11, 130: 0.07, 193: 0.12, 194: 0.06}
FR, AR = ref.FORMAT_R, ref.ANSWER_R
CKPT = {r: f"/data/ckpt/{r}" for r in RUNS}
DATA = {"your-01": "/data/code-r1-plus-2k-leetcode2k-dedup/train.parquet", "your-02": "/data/code-r1-plus-2k-leetcode2k-filtered-your01/train.parquet"}
VAL = {"your-01": "/data/code-r1-plus-2k-leetcode2k/test.parquet", "your-02": "/data/code-r1-plus-2k-leetcode2k-filtered-your01/val200.parquet"}


# ----------------------------------------------------------------------------- 动力学
def dyn(run, t):
    """无噪声的闭式状态。your-02 的起点由 your-01 第 200 步的状态和筛题结果推出。"""
    if run == "your-01":
        tag = 0.32 * math.exp(-t / 12) + 0.01                   # Base 起点三分之一没按标签写，30 步内学会
        trunc = 0.12 * math.exp(-t / 20) + 0.03                 # 6144 上限：循环先多后少
        p = 0.30 + 0.27 * (1 - math.exp(-t / 120))             # 起点与参考 run-01/02 的 0.28–0.30 一致；n=8 比 n=16 慢一点
        length = 850 + 1100 * math.exp(-t / 18) + 900 * (1 - math.exp(-max(t - 40, 0) / 150))   # 先降（循环被罚）后升（思考变长）
        kl = 0.090 * (1 - math.exp(-t / 100))                  # Base 起点走得远，每步 4 次更新更快
        ent = 0.30 + 0.85 * math.exp(-t / 32)                  # 熵系数 0：前 60 步格式锁定，之后缓降
        return dict(tag=tag, trunc=trunc, p=p, length=length, kl=kl, ent=ent, conc=3.0)
    tau = t - 200
    s200 = dyn("your-01", 200)
    tag = 0.01
    trunc = 0.03 - 0.01 * (1 - math.exp(-tau / 100))
    p = FILTER["p_filtered_at_start"] + 0.12 * (1 - math.exp(-tau / 110))   # 筛题后的题集：去掉全错题后均值更高，再慢慢涨
    length = s200["length"] + 350 * (1 - math.exp(-tau / 120))
    kl = s200["kl"] + 0.035 * (1 - math.exp(-tau / 120))
    ent = 0.30 - 0.05 * (1 - math.exp(-tau / 150))             # 熵系数仍为 0，没加 clip-higher：继续缓降
    return dict(tag=tag, trunc=trunc, p=p, length=length, kl=kl, ent=ent, conc=5.0)   # 筛题后难度更集中


def make_filter():
    """第七环的筛题：用 your-01 第 200 步对 1889 题采样 8 条/题，去掉全对、全错；再切 200 题当验证集。"""
    s = dyn("your-01", 200)
    all_pass = ref._beta_moment(s["p"], 3.0, N_SAMPLE)
    all_fail = ref._beta_moment(s["p"], 3.0, N_SAMPLE, fail=True)
    n_all_pass, n_all_fail = int(round(N_SOURCE * all_pass)), int(round(N_SOURCE * all_fail))
    kept = N_SOURCE - n_all_pass - n_all_fail
    train_rows = kept - N_VAL
    p_f0 = (s["p"] * N_SOURCE - n_all_pass) / kept               # 全对题按 p=1、全错题按 p=0 扣掉
    return {"sampled_with": "your-01/global_step_200", "samples_per_task": N_SAMPLE, "n_source": N_SOURCE, "all_pass_removed": n_all_pass,
            "all_fail_removed": n_all_fail, "kept": kept, "val_split": N_VAL, "train_rows": train_rows, "batches_per_epoch": train_rows // 16,
            "p_filtered_at_start": round(p_f0, 4), "all_pass_rate_at_200": round(all_pass, 4), "all_fail_rate_at_200": round(all_fail, 4)}


FILTER = make_filter()
N_ROWS = {"your-01": N_SOURCE, "your-02": FILTER["train_rows"]}
BATCHES_PER_EPOCH = {r: N_ROWS[r] // 16 for r in RUNS}
TOTAL_EPOCHS = {"your-01": 2, "your-02": 3}        # 200 / 118 = 1.7；续训 200 / 75 ≈ 2.7
for r in RUNS:
    ref.T0[r], ref.WANDB_ID[r], ref.N_ROWS[r], ref.BATCHES_PER_EPOCH[r] = T0[r], WANDB_ID[r], N_ROWS[r], BATCHES_PER_EPOCH[r]


def overrides(run):
    rep = {"data.train_files=/data/code-r1-plus-2k-leetcode2k-dedup/train.parquet": f"data.train_files={DATA[run]}",
           "data.val_files=/data/code-r1-plus-2k-leetcode2k/test.parquet": f"data.val_files={VAL[run]}",
           "data.max_response_length=4096": f"data.max_response_length={MAX_LEN}", "actor_rollout_ref.rollout.n=16": f"actor_rollout_ref.rollout.n={N_SAMPLE}",
           "actor_rollout_ref.actor.ppo_mini_batch_size=16": "actor_rollout_ref.actor.ppo_mini_batch_size=4",
           "actor_rollout_ref.actor.entropy_coeff=0.001": "actor_rollout_ref.actor.entropy_coeff=0",
           "reward_model.reward_manager=prime": "reward_model.reward_manager=prime_plus",
           "trainer.experiment_name=run-01-qwen3-8b-lc2k": f"trainer.experiment_name={run}-qwen3-8b-base-lc2k" + ("-filtered" if run == "your-02" else ""),
           "trainer.total_epochs=3": f"trainer.total_epochs={TOTAL_EPOCHS[run]}", "trainer.total_training_steps=320": f"trainer.total_training_steps={STEPS[run][1]}",
           "trainer.rollout_data_dir=/data/rollouts/run-01": f"trainer.rollout_data_dir=/data/rollouts/{run}",
           "trainer.default_local_dir=/root/ckpt/run-01": f"trainer.default_local_dir={CKPT[run]}"}
    out = [rep.get(o, o) for o in ref.OVERRIDES["run-01"]]
    assert len(rep) == sum(1 for o in out if o in rep.values())
    if run == "your-02":
        out.append("trainer.resume_mode=auto")
    return out


for r in RUNS:
    ref.OVERRIDES[r] = overrides(r)


def config(run):
    cfg = ref.resolved_config("run-01")
    cfg["data"].update({"train_files": DATA[run], "val_files": VAL[run], "max_response_length": MAX_LEN})
    cfg["actor_rollout_ref"]["rollout"].update({"response_length": MAX_LEN, "n": N_SAMPLE})
    cfg["actor_rollout_ref"]["actor"].update({"ppo_mini_batch_size": 4, "entropy_coeff": 0})
    cfg["actor_rollout_ref"]["actor"]["optim"]["total_training_steps"] = STEPS[run][1]
    cfg["critic"]["optim"]["total_training_steps"] = STEPS[run][1]
    cfg["critic"]["ppo_mini_batch_size"] = 4
    cfg["critic"]["rollout_n"] = N_SAMPLE
    cfg["reward_model"]["reward_manager"] = "prime_plus"
    cfg["trainer"].update({"total_epochs": TOTAL_EPOCHS[run], "total_training_steps": STEPS[run][1], "experiment_name": overrides(run)[[o.startswith("trainer.experiment_name=") for o in overrides(run)].index(True)].split("=", 1)[1],
                           "default_local_dir": CKPT[run], "max_actor_ckpt_to_keep": None, "rollout_data_dir": f"/data/rollouts/{run}"})
    if run == "your-02":
        cfg["trainer"]["resume_mode"] = "auto"
    return cfg


def launch_script(run):
    body = " \\\n    ".join(overrides(run))
    if run == "your-01":
        head = ("# your-01：按 05 带学时的决定配置（4×A800，Qwen3-8B-Base，LeetCode 去污染版 1889 题，6144 / n=8 / 200 步，mini-batch 4，prime_plus，熵 0）。\n"
                "set -euo pipefail\ncd /root/verl\nexport CODER1_EXEC=firejail\nexport CODER1_PLUS_SANDBOX=firejail\nexport CODER1_PLUS_TMP=/data/tmp/judge\n"
                "export WANDB_MODE=offline\nexport HF_HUB_OFFLINE=1\nexport RUN=your-01\nmkdir -p /data/ckpt/$RUN /data/rollouts/$RUN /data/tmp/judge logs/$RUN\n")
    else:
        head = (f"# your-02：第七环的决定——筛题（your-01 第 200 步采样 8 条/题，去掉全对 {FILTER['all_pass_removed']}、全错 {FILTER['all_fail_removed']}，切 {N_VAL} 题做验证，训练 {FILTER['train_rows']} 行）、\n"
                "# 从 your-01 第 200 步续训 200 步、验证集换成切出的 200 题、判题临时目录改 tmpfs。其余与 your-01 相同（计划里的 clip_ratio_high 0.28 已去掉）。\n"
                "set -euo pipefail\ncd /root/verl\nexport CODER1_EXEC=firejail\nexport CODER1_PLUS_SANDBOX=firejail\nexport CODER1_PLUS_TMP=/dev/shm/judge          # 第七环：不再和 checkpoint 同盘\n"
                "export WANDB_MODE=offline\nexport HF_HUB_OFFLINE=1\nexport RUN=your-02\nmkdir -p /data/ckpt/$RUN /data/rollouts/$RUN /dev/shm/judge logs/$RUN\n"
                "# 续训起点：复制 your-01 的第 200 步，去掉 data.pt（否则 StatefulDataLoader 会按旧数据集的状态恢复；实跑前核对 v0.4.1 的 _load_checkpoint）\n"
                "if [ ! -d /data/ckpt/$RUN/global_step_200 ]; then\n    cp -r /data/ckpt/your-01/global_step_200 /data/ckpt/$RUN/\n"
                "    rm -f /data/ckpt/$RUN/global_step_200/data.pt\n    echo 200 > /data/ckpt/$RUN/latest_checkpointed_iteration.txt\nfi\n")
    return "#!/usr/bin/env bash\n" + head + "set -x\npython3 -m verl.trainer.main_ppo \\\n    " + body + " \\\n    \"$@\" 2>&1 | tee logs/$RUN/grpo.log\n"


# ----------------------------------------------------------------------------- 指标
def metrics(run):
    rng = random.Random(61 if run == "your-01" else 62)
    first, last = STEPS[run]
    rows, wall = [], 0.0
    for t in range(first, last + 1):
        s = dyn(run, t)
        p_cor = min(0.97, max(0.0, s["p"] + rng.gauss(0, 0.016 if run == "your-01" else 0.014)))
        fmt = min(0.9, s["tag"] + s["trunc"])
        infra = IO_EVENT.get(t, 0.0) if run == "your-01" else 0.0
        # 判题故障：prime_plus 先重试（成功率一半），仍失败的按 uid 整组作废（分数设 0.0，advantage 为 0）
        n_infra_raw = int(round(BATCH * infra))
        n_infra = int(round(n_infra_raw * 0.5))
        n_groups_neutral = min(N_PROMPT, int(math.ceil(n_infra / 2.5))) if n_infra else 0
        n_zero = n_groups_neutral * N_SAMPLE
        scored = BATCH - n_zero
        n_fmt = int(round(scored * fmt))
        n_pass = int(round((scored - n_fmt) * p_cor))
        n_fail = scored - n_fmt - n_pass
        reward = ((FR + AR) * n_pass + FR * n_fail - (FR + AR) * n_fmt) / BATCH
        all_pass = ref._beta_moment(p_cor, s["conc"], N_SAMPLE)
        all_fail = ref._beta_moment(p_cor, s["conc"], N_SAMPLE, fail=True)
        length = s["length"] + rng.gauss(0, 40)
        gen = BATCH * length / 1500 + 15 + rng.gauss(0, 8)            # 128 条一波，两个 TP=2 的 A800 副本，每引擎 64 条并发
        reward_s = 25 + rng.gauss(0, 3) + (40 if infra else 0)
        tok_scale = (BATCH * (PROMPT_TOK + length)) / (256 * (ref.PROMPT_TOK_MEAN + 2300))   # 相对参考 run-01 的 token 量
        old_lp = (18 + 2300 / 900) * ref.COMPUTE_SLOWDOWN * tok_scale + rng.gauss(0, 4)
        ref_s = (14 + 2300 / 1100) * ref.COMPUTE_SLOWDOWN * tok_scale + rng.gauss(0, 4)
        upd = (70 + 2300 / 400) * ref.COMPUTE_SLOWDOWN * tok_scale * 1.1 + rng.gauss(0, 8)   # 4 个 mini-batch 多 10%
        dump = 1.2 + length / 4000
        testing = (230 + rng.gauss(0, 10)) if (t % ref.TEST_FREQ == 0 or t == last) else 0.0     # v0.4.1 最后一步也验证、也存
        save = (150 + rng.gauss(0, 8)) if (t % ref.SAVE_FREQ == 0 or t == last) else 0.0
        step_s = gen + reward_s + old_lp + ref_s + 0.3 + upd + dump + testing + save + 4
        wall += step_s
        total_tok = BATCH * (PROMPT_TOK + length)
        adv_mean = round(rng.gauss(0, 0.005), 4)
        adv_max = rng.choice([2.47, 2.47, 1.87, 1.53])   # n=8 的组内标准化上限 (1-1/8)/std
        adv_min = -rng.choice([2.47, 1.87, 1.53, 1.0])
        kl = max(1e-5, s["kl"] * (1 + rng.gauss(0, 0.05)))
        clipfrac = max(0.0, rng.gauss(0.034, 0.008))
        grad = rng.gauss((0.45 + 0.50 * math.exp(-t / 40)) if run == "your-01" else 0.42, 0.08)
        row = {
            "step": t, "training/global_step": t, "training/epoch": (t - first) // BATCHES_PER_EPOCH[run],
            "critic/score/mean": round(reward, 6), "critic/score/max": 1.1, "critic/score/min": -1.1 if n_fmt else (0.0 if n_zero else 0.1),
            "critic/rewards/mean": round(reward, 6), "critic/rewards/max": 1.1, "critic/rewards/min": -1.1 if n_fmt else (0.0 if n_zero else 0.1),
            "critic/advantages/mean": adv_mean, "critic/advantages/max": adv_max, "critic/advantages/min": adv_min,
            "critic/returns/mean": adv_mean, "critic/returns/max": adv_max, "critic/returns/min": adv_min,
            "response_length/mean": round(length, 1), "response_length/max": MAX_LEN, "response_length/min": int(max(150, rng.gauss(500, 110))),
            "response_length/clip_ratio": round(max(0.0, s["trunc"] + rng.gauss(0, 0.005)), 4),
            "prompt_length/mean": round(PROMPT_TOK + rng.gauss(0, 18), 1), "prompt_length/max": int(rng.gauss(690, 60)), "prompt_length/min": int(rng.gauss(118, 12)),
            "prompt_length/clip_ratio": 0.0,
            "actor/entropy": round(s["ent"] + rng.gauss(0, 0.012), 4),
            "actor/pg_loss": round(rng.gauss(-0.0015, 0.003), 5),
            "actor/pg_clipfrac": round(clipfrac, 4), "actor/ppo_kl": round(max(0.0, rng.gauss(5.5e-4, 1.5e-4)), 6),
            "actor/pg_clipfrac_lower": 0.0,
            "actor/kl_loss": round(kl, 6), "actor/kl_coef": 0.001,
            "actor/grad_norm": round(grad, 4), "actor/lr": 5e-7,
            "perf/max_memory_allocated_gb": round(59.8 + length / 1500 + rng.gauss(0, 0.4), 2),
            "perf/max_memory_reserved_gb": round(67.5 + length / 1500 + rng.gauss(0, 0.4), 2),
            "perf/cpu_memory_used_gb": round(121 + rng.gauss(0, 2), 2),
            "perf/total_num_tokens": int(total_tok), "perf/time_per_step": round(step_s, 2), "perf/throughput": round(total_tok / (step_s * ref.N_GPUS), 2),
            "timing_s/gen": round(gen, 2), "timing_s/reward": round(reward_s, 2), "timing_s/old_log_prob": round(old_lp, 2), "timing_s/ref": round(ref_s, 2),
            "timing_s/adv": 0.3, "timing_s/update_actor": round(upd, 2), "timing_s/dump_rollout_generations": round(dump, 2), "timing_s/step": round(step_s, 2),
            "timing_per_token_ms/gen": round(gen * 1000 / (BATCH * length), 4), "timing_per_token_ms/ref": round(ref_s * 1000 / total_tok, 4),
            "timing_per_token_ms/adv": round(0.3 * 1000 / total_tok, 5), "timing_per_token_ms/update_actor": round(upd * 1000 / total_tok, 4),
        }
        if testing:
            row["timing_s/testing"] = round(testing, 2)
        if save:
            row["timing_s/save_checkpoint"] = round(save, 2)
        row.update({"scenario/n_zero": n_zero, "scenario/n_format_fail": n_fmt, "scenario/n_pass": n_pass, "scenario/n_fail": n_fail,
                    "scenario/format_fail_rate": round(fmt, 4), "scenario/format_fail_truncation": round(s["trunc"], 4),
                    "scenario/pass_rate_given_format": round(p_cor, 4),
                    "scenario/infra_errors_raw": n_infra_raw, "scenario/infra_errors_after_retry": n_infra, "scenario/neutralized_groups": n_groups_neutral,
                    "scenario/group_all_pass": round(all_pass, 4), "scenario/group_all_fail": round(all_fail, 4), "scenario/wall_seconds": round(wall, 1)})
        rows.append(row)
    return rows


def validation(run):
    """your-01 用 v2 test 200 题（6144 上限贪心）；your-02 用筛题后切出的 200 题（难度更集中、起点更高）。"""
    rng = random.Random(79 if run == "your-01" else 80)
    first, last = STEPS[run]
    steps = [first - 1] + [t for t in range(first, last + 1) if t % ref.TEST_FREQ == 0 or t == last]
    out = []
    for t in steps:
        if run == "your-01":
            fmt = 0.40 * math.exp(-t / 18) + 0.03             # 贪心、6144 上限：标签缺失 + 循环
            p = 0.25 + 0.30 * (1 - math.exp(-t / 120))
        else:
            tau = t - 200
            fmt = 0.03 - 0.01 * (1 - math.exp(-tau / 100))
            p = min(0.9, FILTER["p_filtered_at_start"] + 0.06 + 0.10 * (1 - math.exp(-tau / 110)))   # 贪心比采样高一截；筛题后的题集起点更高
        if t != first - 1:
            p += rng.gauss(0, 0.008)
        n_fmt = int(round(200 * fmt)); n_pass = min(int(round(200 * p)), 200 - n_fmt)
        score = ((FR + AR) * n_pass + FR * (200 - n_fmt - n_pass) - (FR + AR) * n_fmt) / 200
        out.append({"step": t, "val-core/code_plus/reward/mean@1": round(score, 6), "scenario/val_n_pass": n_pass, "scenario/val_n_format_fail": n_fmt,
                    "scenario/val_pass_rate": round(n_pass / 200, 4), "scenario/val_format_fail": round(n_fmt / 200, 4),
                    "scenario/val_set": "v2test-200" if run == "your-01" else "filtered-val200"})
    return out


def attribution(rows, windows):
    """与参考的 attribution 相同的分解：reward 增量 = 格式部分 + 正确率部分 + 残差。"""
    def pick(t):
        return next(r for r in rows if r["step"] == t)
    out = []
    for a, b in windows:
        ra, rb = pick(a), pick(b)
        fa, fb = ra["scenario/format_fail_rate"], rb["scenario/format_fail_rate"]
        pa, pb = ra["scenario/pass_rate_given_format"], rb["scenario/pass_rate_given_format"]
        d = rb["critic/rewards/mean"] - ra["critic/rewards/mean"]

        def rew(f, p):
            ok = 1 - f
            return (FR + AR) * ok * p + FR * ok * (1 - p) - (FR + AR) * f
        f_part = rew(fb, pa) - rew(fa, pa)
        c_part = rew(fb, pb) - rew(fb, pa)
        out.append({"from": a, "to": b, "d_reward": round(d, 3), "format_part": round(f_part, 3), "correctness_part": round(c_part, 3),
                    "residual": round(d - f_part - c_part, 3), "format_share": round(f_part / d, 2) if d else None})
    return out


# ----------------------------------------------------------------------------- 控制台 / wandb / checkpoint / rollout
def console_lines(run, cfg, rows, val):
    rng = random.Random(202 if run == "your-01" else 203)
    first, last = STEPS[run]
    lines = ref.header_lines(run, cfg, rng)
    lines = [l.replace("Total training steps: 320", f"Total training steps: {last}").replace("Total steps: 320", f"Total steps: {last}") for l in lines]
    if run == "your-02":
        i = next(i for i, l in enumerate(lines) if l.startswith("Checkpoint tracker file does not exist"))
        lines[i:i + 2] = [f"Load from checkpoint folder: {CKPT[run]}/global_step_200", "Setting global step to 200", f"Resuming from {CKPT[run]}/global_step_200"]
    assert f"Size of train dataloader: {BATCHES_PER_EPOCH[run]}" in "\n".join(lines)
    v0 = val[0]["val-core/code_plus/reward/mean@1"]
    lines += ref.validation_lines(first - 1, val[0]) + [f"'Initial validation metrics: {{'val-core/code_plus/reward/mean@1': {v0}}}'", f"step:{first - 1} - val-core/code_plus/reward/mean@1:{v0:.3f}"]
    wall = 0.0
    for row in rows:
        row = dict(row)
        t = row["step"]
        if row["scenario/infra_errors_raw"]:
            for _ in range(row["scenario/infra_errors_raw"]):
                lines.append(f"[prime_plus] infra error: sandbox start timeout (10.0s) in /data/tmp/judge/{rng.randrange(16**8):08x}; retrying once")
            rep = {"policy": "neutralize_group", "samples": BATCH, "infra_errors_after_retry": row["scenario/infra_errors_after_retry"],
                   "neutralized_groups": row["scenario/neutralized_groups"], "neutralized_samples": row["scenario/n_zero"], "retried": row["scenario/infra_errors_raw"]}
            lines.append(f"[prime_plus] judge infrastructure errors: {rep}")
        lines.append(ref.BATCH_KEYS); lines.append(f"Dumped generations to /data/rollouts/{run}/{t}.jsonl")
        if t % ref.TEST_FREQ == 0 or t == last:
            vv = next(v for v in val if v["step"] == t); lines += ref.validation_lines(t, vv); row["val-core/code_plus/reward/mean@1"] = vv["val-core/code_plus/reward/mean@1"]
        if t % ref.SAVE_FREQ == 0 or t == last:
            lines.append(f"local_global_step_folder: {CKPT[run]}/global_step_{t}")
        lines.append(ref.console_metric_line(row))
        wall += row["timing_s/step"]
        done = t - first + 1
        if t % 10 == 0 or t == last:
            remain = (last - t) * (wall / done)
            lines.append(f"Training Progress: {100 * t // last:3d}%|{'█' * (t * 10 // last)}{' ' * (10 - t * 10 // last)}| {t}/{last} "
                         f"[{int(wall // 3600)}:{int(wall % 3600 // 60):02d}:{int(wall % 60):02d}<{int(remain // 3600)}:{int(remain % 3600 // 60):02d}:{int(remain % 60):02d}, {wall / done:.2f}s/it]")
    vlast = val[-1]["val-core/code_plus/reward/mean@1"]
    lines.append(f"'Final validation metrics: {{'val-core/code_plus/reward/mean@1': {vlast}}}'")
    return lines


def write_console_log(run, cfg, rows, val, out_dir):
    lines = console_lines(run, cfg, rows, val)
    text = "\n".join(lines) + "\n"
    (out_dir / "grpo.log.gz").write_bytes(gzip.compress(text.encode("utf-8")))
    first = STEPS[run][0]
    head_end = next(i for i, l in enumerate(lines) if l.startswith(f"step:{first + 1} ")) + 1
    (out_dir / "grpo.head.log").write_text("\n".join(lines[:head_end]) + "\n", encoding="utf-8")
    if run == "your-01":
        start = next(i for i, l in enumerate(lines) if l.startswith(f"local_global_step_folder: {CKPT[run]}/global_step_64"))
        end = next(i for i, l in enumerate(lines) if l.startswith("step:67 "))
        (out_dir / "incident_step65-66.log").write_text("\n".join(lines[start:end + 1]) + "\n", encoding="utf-8")
    else:
        start = next(i for i, l in enumerate(lines) if l.startswith(f"local_global_step_folder: {CKPT[run]}/global_step_256"))
        end = next(i for i, l in enumerate(lines) if l.startswith("step:259 "))
        (out_dir / "after_save_step256-258.log").write_text("\n".join(lines[start:end + 1]) + "\n", encoding="utf-8")   # 修好后存盘后一步的样子
    return text.count("\n")


def write_wandb(run, cfg, rows, val, out_root):
    base = T0[run]
    d = out_root / "wandb" / f"offline-run-{base.strftime('%Y%m%d_%H%M%S')}-{WANDB_ID[run]}" / "files"
    d.mkdir(parents=True, exist_ok=True)
    y = ("wandb_version: 1\n\n_wandb:\n  desc: null\n  value:\n    cli_version: 0.19.8\n    python_version: 3.10.12\n"
         "    t:\n      1: [1, 11, 41, 49, 55, 63, 71, 98]\n      2: [1, 11, 41, 49, 55, 63, 71, 98]\n      3: [13, 16, 23, 61]\n"
         "      4: 3.10.12\n      5: 0.19.8\n      8: [5]\n      13: linux-x86_64\n")
    for k, v in cfg.items():
        y += f"{k}:\n  desc: null\n  value:\n" + ref.to_yaml(v, 2)
    (d / "config.yaml").write_text(y, encoding="utf-8")
    meta = {"os": "Linux-5.15.0-113-generic-x86_64-with-glibc2.35", "python": "3.10.12", "startedAt": base.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "program": "/root/verl/verl/trainer/main_ppo.py", "codePath": "verl/trainer/main_ppo.py",
            "git": {"remote": "https://github.com/volcengine/verl.git", "commit": ref.VERL_COMMIT},
            "root": "/root/verl", "host": "a800-4x-node07", "username": "root", "executable": "/usr/bin/python3", "cpu_count": 64, "cpu_count_logical": 128,
            "gpu": "NVIDIA A800-SXM4-80GB", "gpu_count": 4, "cuda": "12.4", "memory": {"total": "512 GiB"}, "args": overrides(run)}
    (d / "wandb-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    vmap = {v["step"]: v["val-core/code_plus/reward/mean@1"] for v in val}
    last_step = STEPS[run][1]
    last = {k: v for k, v in rows[-1].items() if not k.startswith("scenario/") and k != "step"}
    last.update({"val-core/code_plus/reward/mean@1": vmap[last_step], "_step": last_step, "_runtime": rows[-1]["scenario/wall_seconds"],
                 "_timestamp": (base + dt.timedelta(seconds=rows[-1]["scenario/wall_seconds"])).timestamp()})
    (d / "wandb-summary.json").write_text(json.dumps(last, indent=2), encoding="utf-8")
    with open(d / "history.export.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"_step": STEPS[run][0] - 1, "_runtime": 0, "val-core/code_plus/reward/mean@1": vmap[STEPS[run][0] - 1]}) + "\n")
        for r in rows:
            rec = {k: v for k, v in r.items() if not k.startswith("scenario/")}
            rec["_step"] = r["step"]; rec["_runtime"] = r["scenario/wall_seconds"]
            rec["_timestamp"] = (base + dt.timedelta(seconds=r["scenario/wall_seconds"])).timestamp()
            if r["step"] in vmap:
                rec["val-core/code_plus/reward/mean@1"] = vmap[r["step"]]
            f.write(json.dumps(rec) + "\n")


def write_checkpoint_manifest(run, cfg, rows, out_root, prev_rows=None):
    d = out_root / "checkpoints" / run
    d.mkdir(parents=True, exist_ok=True)
    root = cfg["trainer"]["default_local_dir"]
    first, last = STEPS[run]
    keep = [t for t in range(first, last + 1) if t % ref.SAVE_FREQ == 0 or t == last]
    note = "数据盘，全部保留；v0.4.1 最后一步也存" if run == "your-01" else "global_step_200 是从 your-01 复制来的续训起点（去掉了 data.pt，mtime 是复制时间）；其余为本轮所存"
    lines = [f"# find {root} -type f -printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' | sort -k3   （{note}）"]
    per_rank_model = ref.MODEL_PARAMS * 4 // ref.N_GPUS
    per_rank_optim = ref.MODEL_PARAMS * 8 // ref.N_GPUS
    total = 0

    def one(step, stamp, with_data):
        nonlocal total
        for rank in range(ref.N_GPUS):
            for name, size in ((f"model_world_size_{ref.N_GPUS}_rank_{rank}.pt", per_rank_model + 1_234_567 + rank * 17),
                               (f"optim_world_size_{ref.N_GPUS}_rank_{rank}.pt", per_rank_optim + 2_345_678 + rank * 13),
                               (f"extra_state_world_size_{ref.N_GPUS}_rank_{rank}.pt", 15_874 + rank)):
                lines.append(f"{size}\t{stamp}\t{root}/global_step_{step}/actor/{name}")
                total += size
        lines.append(f"45\t{stamp}\t{root}/global_step_{step}/actor/fsdp_config.json")
        for name, size in (("config.json", 726), ("generation_config.json", 239), ("tokenizer.json", 11_422_654), ("tokenizer_config.json", 9_732),
                           ("vocab.json", 2_776_833), ("merges.txt", 1_671_853), ("special_tokens_map.json", 613), ("added_tokens.json", 707)):
            lines.append(f"{size}\t{stamp}\t{root}/global_step_{step}/actor/huggingface/{name}")
            total += size
        if with_data:
            lines.append(f"2174\t{stamp}\t{root}/global_step_{step}/data.pt")

    n_copies = len(keep)
    if run == "your-02":
        one(200, (T0[run] - dt.timedelta(minutes=35)).strftime("%Y-%m-%d %H:%M"), False)
        n_copies += 1
    for step in keep:
        one(step, (T0[run] + dt.timedelta(seconds=rows[step - first]["scenario/wall_seconds"])).strftime("%Y-%m-%d %H:%M"), True)
    final_stamp = (T0[run] + dt.timedelta(seconds=rows[-1]["scenario/wall_seconds"])).strftime("%Y-%m-%d %H:%M")
    lines.append(f"3\t{final_stamp}\t{root}/latest_checkpointed_iteration.txt")
    lines.append(f"# 合计 {total / 1e9:.1f} GB，每份 checkpoint 约 {(total / n_copies) / 1e9:.1f} GB（模型 fp32 分片 {per_rank_model * ref.N_GPUS / 1e9:.1f} GB + 优化器 {per_rank_optim * ref.N_GPUS / 1e9:.1f} GB）")
    (d / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "latest_checkpointed_iteration.txt").write_text(str(last), encoding="utf-8")


def write_rollout_dump(run, cfg, rows, judge, recs, out_path, step):
    """一步的 rollout 落盘：16 题 × 8 条。作废的组整组记 0.0；候选代码是参考解（过）或经真实判分代码验证会失败的变异体（不过）。"""
    rng = random.Random(1000 + step)
    row = rows[step - STEPS[run][0]]
    prompts = ref.pick_prompts(judge, recs, rng)
    n_zero_groups = row["scenario/neutralized_groups"]
    kinds_scored = ref.sample_kinds(rng, {**row, "scenario/n_zero": 0})          # 有分数的样本按整数计数分类
    kinds_scored = [k for k in kinds_scored if k != "zero"][: BATCH - n_zero_groups * N_SAMPLE]
    lines = []
    for g, rec in enumerate(prompts):
        for j in range(N_SAMPLE):
            i = g * N_SAMPLE + j
            if g < n_zero_groups:
                kind, score = rng.choice(["pass", "fail", "trunc"]), 0.0
            else:
                kind = kinds_scored[i - n_zero_groups * N_SAMPLE]
                score = {"trunc": -1.1, "tag": -1.1, "pass": 1.1, "fail": 0.1}[kind]
            code = judge.candidate_code(rec) if kind != "fail" else judge.failing_code(rng, rec)
            resp, _ = ref.make_response(rng, rec, code, kind, cfg["data"]["max_response_length"], row["response_length/mean"])
            lines.append(json.dumps({"input": ref.decoded_prompt(rec), "output": resp, "score": score, "step": step}, ensure_ascii=False))
    out_path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))
    return len(lines)


# ----------------------------------------------------------------------------- 评测（基线 / your-01 第 200 步 / your-02 第 400 步）
def evaluate_all():
    dev, lcb = ref.eval_problems()
    ref.SKILL["dev"].update({"run-01": -0.20, "run-02": 0.30})
    ref.SKILL["lcb"].update({"run-01": -1.65, "run-02": -1.25})
    results = ref.evaluate(dev, lcb)
    name = {"base": "base", "run-01": "your-01", "run-02": "your-02"}
    return {k: {name[m]: v for m, v in d.items()} for k, d in results.items()}


def summarize(results):
    models = ["base", "your-01", "your-02"]

    def agg(r):
        return round(sum(x["passed"] for x in r) / len(r), 4)

    def sampled(r):
        per = {sd: sum(x["pass_counts_by_seed"][sd] for x in r) / (8 * len(r)) for sd in ("1", "2", "3")}
        m = sum(per.values()) / 3; std = (sum((v - m) ** 2 for v in per.values()) / 2) ** 0.5
        p8 = {sd: sum(1 for x in r if x["pass_counts_by_seed"][sd] > 0) / len(r) for sd in ("1", "2", "3")}
        return {"pass@1_by_seed": {k: round(v, 4) for k, v in per.items()}, "pass@1_mean": round(m, 4), "pass@1_std": round(std, 4), "pass@8_mean": round(sum(p8.values()) / 3, 4)}

    def paired(a, b):
        win = sum(1 for x, y in zip(a, b) if y["passed"] and not x["passed"]); loss = sum(1 for x, y in zip(a, b) if x["passed"] and not y["passed"])
        return {"win": win, "loss": loss, "same": len(b) - win - loss, "net": win - loss, "net_points": round(100 * (win - loss) / len(b), 1)}

    def by_diff(rows):
        out = {}
        for r in rows:
            d = r["difficulty"].lower(); out.setdefault(d, [0, 0]); out[d][0] += r["passed"]; out[d][1] += 1
        return {d: {"pass1": round(v[0] / v[1], 4), "n": v[1]} for d, v in out.items()}

    summary = {"models": models, "checkpoints": {"your-01": "global_step_200", "your-02": "global_step_400"},
               "note": "dev = v2 test 200 题（your-01 训练中也用它验证；your-02 改用切出的 200 题验证，dev 只在最后评）；lcb = LiveCodeBench 2025-05-01 以后 142 题 held-out，只评一次、不用于选 checkpoint"}
    for setname in ("dev", "lcb"):
        g, s = results[f"{setname}_greedy"], results[f"{setname}_sampled"]
        summary[setname] = {"n": len(g["base"]), "greedy_pass1": {m: agg(g[m]) for m in models}, "greedy_pass_count": {m: sum(x["passed"] for x in g[m]) for m in models},
                            "greedy_format_fail": {m: round(sum(not x["format_ok"] for x in g[m]) / len(g[m]), 4) for m in models},
                            "greedy_by_difficulty": {m: by_diff(g[m]) for m in models}, "sampled": {m: sampled(s[m]) for m in models},
                            "paired_greedy": {"your-01_vs_base": paired(g["base"], g["your-01"]), "your-02_vs_base": paired(g["base"], g["your-02"]),
                                              "your-02_vs_your-01": paired(g["your-01"], g["your-02"])}}
    return summary


def eval_tables_md(summary):
    out = ["# 评测结果表（由 scenario_yourplan.py 生成，report.md 的结论引用这里的数字）", ""]
    for setname, title in (("dev", "dev：LeetCodeDataset v2 test，200 题"), ("lcb", "held-out：LiveCodeBench 2025-05-01 以后，142 题")):
        e = summary[setname]
        out += [f"## {title}", "", "| | 基线 Qwen3-8B-Base | your-01（第 200 步） | your-02（第 400 步） |", "|---|---:|---:|---:|"]
        ms = ("base", "your-01", "your-02")
        out.append("| 贪心 pass@1 | " + " | ".join(f"{e['greedy_pass1'][m]:.3f}（{e['greedy_pass_count'][m]} 题）" for m in ms) + " |")
        out.append("| 贪心格式错 | " + " | ".join(f"{100 * e['greedy_format_fail'][m]:.1f}%" for m in ms) + " |")
        out.append("| 采样 pass@1（3 seed） | " + " | ".join(f"{e['sampled'][m]['pass@1_mean']:.3f} ± {e['sampled'][m]['pass@1_std']:.3f}" for m in ms) + " |")
        out.append("| 采样 pass@8 | " + " | ".join(f"{e['sampled'][m]['pass@8_mean']:.3f}" for m in ms) + " |")
        out.append("")
        out += ["| 配对（贪心，赢 / 输 / 同） | 净胜 | 百分点 |", "|---|---:|---:|"]
        for k, label in (("your-01_vs_base", "your-01 对基线"), ("your-02_vs_base", "your-02 对基线"), ("your-02_vs_your-01", "your-02 对 your-01")):
            p = e["paired_greedy"][k]
            out.append(f"| {label}：{p['win']} / {p['loss']} / {p['same']} | {p['net']} | {p['net_points']:+.1f} |")
        out.append("")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- 主流程
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config").mkdir(exist_ok=True)
    recs, _ = ref.train_records()
    judge = ref.Judge()
    report = {"generator": "scenario_yourplan.py", "nature": "推演：按用户在 05 带学时的决定生成；控制台无逐样本判分块（prime_plus 不打印），rollout 落盘里的代码经真实判分代码验证",
              "runs": list(RUNS), "n": N_SAMPLE, "max_response_length": MAX_LEN, "filter": FILTER}
    all_rows, all_val = {}, {}
    for run in RUNS:
        cfg, rows, val = config(run), metrics(run), validation(run)
        all_rows[run], all_val[run] = rows, val
        first, last = STEPS[run]
        (OUT / "config" / f"{run}.sh").write_text(launch_script(run), encoding="utf-8")
        (OUT / "config" / f"{run}.resolved.yaml").write_text(ref.to_yaml(cfg), encoding="utf-8")
        logs = OUT / "logs" / run; logs.mkdir(parents=True, exist_ok=True)
        with open(logs / "metrics.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("scenario/")}) + "\n")
        with open(logs / "scenario_internal.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"step": r["step"], **{k: v for k, v in r.items() if k.startswith("scenario/")}}) + "\n")
        with open(logs / "validation.jsonl", "w", encoding="utf-8") as f:
            for v in val:
                f.write(json.dumps(v) + "\n")
        n_lines = write_console_log(run, cfg, rows, val, logs)
        write_wandb(run, cfg, rows, val, OUT)
        write_checkpoint_manifest(run, cfg, rows, OUT)
        (OUT / "rollouts" / run).mkdir(parents=True, exist_ok=True)
        dumps = {}
        for step in ((65, 200) if run == "your-01" else (201, 400)):
            dumps[step] = write_rollout_dump(run, cfg, rows, judge, recs, OUT / "rollouts" / run / f"{step}.jsonl.gz", step)
        hours = rows[-1]["scenario/wall_seconds"] / 3600
        base = T0[run]
        marks = (64, 65, 66, 67, 128, 129, 130, 131, 192, 193, 194, 200) if run == "your-01" else (201, 256, 257, 320, 321, 384, 385, 400)
        windows = ((1, 64), (64, 128), (128, 200)) if run == "your-01" else ((201, 256), (256, 320), (320, 400))
        report[run] = {"steps": [first, last], "console_physical_lines": n_lines, "rollout_dumps": dumps,
                       "wall_hours": round(hours, 1), "gpu_cost_cny": round(hours * ref.N_GPUS * ref.GPU_HOUR_CNY), "started": base.isoformat(),
                       "finished": (base + dt.timedelta(hours=hours)).isoformat(),
                       "step_end_times": {t: (base + dt.timedelta(seconds=rows[t - first]["scenario/wall_seconds"])).strftime("%m-%d %H:%M:%S") for t in marks},
                       "mean_step_s": round(sum(r["timing_s/step"] for r in rows) / len(rows), 1), "mean_gen_s": round(sum(r["timing_s/gen"] for r in rows) / len(rows), 1),
                       "mean_update_s": round(sum(r["timing_s/update_actor"] for r in rows) / len(rows), 1),
                       "first": {k: rows[0][k] for k in ("critic/rewards/mean", "response_length/mean", "response_length/clip_ratio", "actor/entropy", "actor/kl_loss",
                                                        "scenario/format_fail_rate", "scenario/pass_rate_given_format", "scenario/group_all_fail")},
                       "final": {k: rows[-1][k] for k in ("critic/rewards/mean", "response_length/mean", "response_length/clip_ratio", "actor/kl_loss", "actor/entropy", "actor/pg_clipfrac",
                                                         "scenario/format_fail_rate", "scenario/pass_rate_given_format", "scenario/group_all_pass", "scenario/group_all_fail")},
                       "val_first_last": [val[0]["val-core/code_plus/reward/mean@1"], val[-1]["val-core/code_plus/reward/mean@1"]],
                       "val_pass_first_last": [val[0]["scenario/val_pass_rate"], val[-1]["scenario/val_pass_rate"]],
                       "val_max": max((v["val-core/code_plus/reward/mean@1"], v["step"]) for v in val),
                       "neutralized_groups_total": sum(r["scenario/neutralized_groups"] for r in rows),
                       "io_event": {t: {"infra_raw": rows[t - first]["scenario/infra_errors_raw"], "after_retry": rows[t - first]["scenario/infra_errors_after_retry"],
                                        "neutralized_groups": rows[t - first]["scenario/neutralized_groups"], "reward": rows[t - first]["critic/rewards/mean"]} for t in marks},
                       "attribution": attribution(rows, windows)}
        att = OUT / "attribution"; att.mkdir(exist_ok=True)
        (att / f"reward_decomposition_{run}.json").write_text(json.dumps(report[run]["attribution"], indent=2), encoding="utf-8")
    results = evaluate_all()
    ev = OUT / "eval"
    for setname in ("dev", "lcb"):
        for mode in ("greedy", "sampled"):
            d = ev / f"{setname}_{mode}"; d.mkdir(parents=True, exist_ok=True)
            for model, rows_ in results[f"{setname}_{mode}"].items():
                with open(d / f"{model}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows_:
                        f.write(json.dumps(r) + "\n")
    summary = summarize(results)
    (ev / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ev / "tables.md").write_text(eval_tables_md(summary), encoding="utf-8")
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "data" / "filter_manifest.json").write_text(json.dumps(FILTER, ensure_ascii=False, indent=2), encoding="utf-8")
    report["eval"] = summary
    report["plots"] = ref.plot(all_rows, all_val, OUT / "curves")
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "eval"}, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("dev", "lcb")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
