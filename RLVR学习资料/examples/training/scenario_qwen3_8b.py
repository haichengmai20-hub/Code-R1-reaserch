"""推演生成器：Qwen3-8B-Base 在 Code-R1 流水线上两轮 GRPO 训练（run-01、run-02）的全部运行产物。

这不是训练日志，是一个显式的参数模型加固定种子：曲线由下面几个闭式函数生成，日志、评测明细、
wandb 目录、checkpoint 清单都从同一组数字派生，所以互相自洽、可复算、改假设就能重算。
文件格式与字段名对齐 verl v0.4.1（训练节点上的版本：指标名来自 metric_utils.py 与 ray_trainer.py，
验证指标名 val-core/{data_source}/reward/mean@N，控制台一行一步 step:N - key:value，
rollout 落盘 {step}.jsonl）和本仓库的判分代码（coder1_plus 的日志格式）。
判分日志里的「Reward Calculation (coder1_plus)」块是用真实判分代码在本机执行参考解和变异体得到的，
不是编出来的；思考文本（<think> 里的内容）是模板合成的。所有随机数都从固定种子派生，不依赖 PYTHONHASHSEED。

用法（仓库根目录）：
    $PY -B RLVR学习资料/examples/training/scenario_qwen3_8b.py                 # 写到 examples/training/runs/qwen3-8b-grpo-20260928/
    <venv>/bin/python -B RLVR学习资料/examples/training/scenario_qwen3_8b.py    # venv 里有 matplotlib 时另画 PNG 曲线
    $PY -B RLVR学习资料/examples/training/check_scenario_docs.py               # 核对叙述文件里引用的数字是否与产物一致

推演假设（改这里就能重算）：
- 每步 16 题 × 16 回答 = 256 条；格式分 0.1、正确分 1.0；格式错 -1.1；v0.4.1 原版 prime 管理器遇到判题异常记 0.0。
- 机器：4×A800 80GB 单机（算力约 H100 的 1/3，显存带宽 0.6 倍，2 个 TP=2 采样副本）。
- 起点是预训练模型 Qwen3-8B-Base（没有 SFT/RL）：<think>…</think><answer>…</answer> 的格式靠系统提示从头学，
  第 1 步约 1/3 的回答没按标签写、一部分循环到回答上限；标签缺失在 30 步内降到 1%，reward 从负起步，
  前 64 步的增益大半是格式；回答长度先降（循环被罚）后升（思考变长）；熵从 1.1 起落；到 ref 的 KL 走得远。
- run-01：max_response_length=4096，起点约 23% 循环到上限（没有 </answer>）→ -1.1，后期稳定在 6%；
  第 129–134 步 checkpoint 把根盘写满，判题写临时文件失败，
  原版管理器把这些样本记成 0.0，故障比例 0.2 → 0.45 → 0.8 → 1.0 × 3 步，第 135 步清盘后恢复。
- run-02：按归因改五处：长度 8192、按 run-01 第 192 步采样通过率筛题（去掉全对 198、全错 155，剩 1536 题）、prime_plus 整组作废、
  ppo_mini_batch_size=4 题（每步 4 次更新）、checkpoint 换到 /data；其余参数不变。
"""
import argparse
import datetime as dt
import gzip
import json
import math
import pprint
import random
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = HERE / "runs" / "qwen3-8b-grpo-20260928"
DATASET = REPO / "LeetCodeDataset" / "data"
sys.path.insert(0, str(REPO / "RLVR学习资料" / "examples" / "source-supplements"))
sys.path.insert(0, str(REPO / "examples" / "data_preprocess"))
import supplement_env as se  # noqa: E402
import coder1_plus_tests as cpt  # noqa: E402

STEPS = 320
N_PROMPT, N_SAMPLE = 16, 16
BATCH = N_PROMPT * N_SAMPLE
FORMAT_R, ANSWER_R = 0.1, 1.0
TEST_FREQ, SAVE_FREQ = 16, 64
N_GPUS = 4                             # 4×A800 80GB，个人研究能租到的量级
GPU_HOUR_CNY = 10.0                    # A800 按 ¥10/卡时估
USD_TO_CNY = 7.2
GPU_HOUR_USD = GPU_HOUR_CNY * N_GPUS / USD_TO_CNY
COMPUTE_SLOWDOWN = 6.0                 # 相对 8×H100：每卡算力约 1/3，卡数一半 → 前向与更新约 6 倍慢
T0 = {"run-01": dt.datetime(2026, 9, 28, 8, 30, 12), "run-02": dt.datetime(2026, 10, 3, 7, 2, 44)}
INCIDENT = {129: 0.2, 130: 0.45, 131: 0.8, 132: 1.0, 133: 1.0, 134: 1.0}
MODEL_PARAMS = 8_190_735_360          # Qwen3-8B-Base 的参数量（8.2B，与 Qwen3-8B 同架构）
PROMPT_TOK_MEAN = 312
N_ROWS = {"run-01": 1889, "run-02": 1536}             # 去污染版；筛题版 = 1889 − 全对 198 − 全错 155
BATCHES_PER_EPOCH = {"run-01": 118, "run-02": 96}   # 1889 // 16, 1536 // 16
VERL_COMMIT = "8d9e350ea58c7ad4b50dd14d9dcb50577242c55f"
WANDB_ID = {"run-01": "3q7vx9kd", "run-02": "n8c4r2wq"}

SYSTEM_PROMPT = se.load_builder().original_system_prompt()
USER_PREFIX = "Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n"


def seed_of(*parts):
    """与 PYTHONHASHSEED 无关的稳定种子。"""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


# ----------------------------------------------------------------------------- 数据
def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def train_records():
    builder = se.load_builder()
    train, _ = builder.leetcode_records()
    return [item[2] if isinstance(item, tuple) else item for item in train], builder


def chat_prompt(rec):
    user = USER_PREFIX + rec["meta"]["query"].strip()
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def decoded_prompt(rec):
    """verl 落盘的 input 是 skip_special_tokens=True 解码的结果：<|im_start|>、<|im_end|> 被去掉，角色名和换行保留。"""
    return chat_prompt(rec).replace("<|im_start|>", "").replace("<|im_end|>", "")


# ----------------------------------------------------------------------------- 思考文本合成（模板）
ALGOS = ["a hash map", "two pointers", "dynamic programming", "sorting first", "a monotonic stack", "BFS over states",
         "binary search on the answer", "prefix sums", "a greedy pass", "a sliding window", "a heap", "union-find",
         "bitmask DP", "recursion with memoization"]
OPENERS = ["Okay, let's see. I need to solve this problem.", "Let me think about this problem step by step.",
           "Hmm, so the task is to figure out the right output for each input here.",
           "Alright, let me read the problem carefully first."]
CHECKS = ["Wait, let me re-check that.", "Hmm, that doesn't look right. Let me redo this part.",
          "Actually, I think I made a mistake earlier.", "Let me verify this with the second example.",
          "Hold on, what about the edge case where the input has only one element?",
          "Let me trace through the example again to be sure.", "Hmm, maybe I'm overcomplicating this.",
          "Wait, but the constraints might make this too slow. Let me think about the complexity.",
          "Let me reconsider. Is there a simpler way to see this?", "Okay, that works. But let me double check the boundary."]


def first_sentence(text):
    m = re.search(r"^(.*?[.!?])(\s|$)", text.strip().replace("\n", " "))
    return (m.group(1) if m else text.strip()[:160]).strip()


def think_text(rng, rec, target_tokens):
    """按目标 token 数合成一段 <think> 里的推理文本（模板），4 个字符约等于 1 个 token。"""
    stmt = first_sentence(rec["meta"]["query"])
    fn = rec["entry_point"].split(".")[-1]
    exs = rec.get("examples") or []
    algo_a, algo_b = rng.sample(ALGOS, 2)
    parts = [rng.choice(OPENERS), f"The problem says: {stmt}", f"The function I need to implement is `{fn}`."]
    if exs:
        parts.append(f"Let me look at the examples. For {exs[0]['input']}, the expected output is {exs[0]['output']}.")
    parts += ["A brute force approach would be to try every possibility, but that is at least quadratic, "
              "which might be too slow if the input is large.",
              f"Let me think about using {algo_a}.", "So the idea is to process the input once and keep track of what I need.",
              "Let me verify with the first example.", "Hmm, that gives the expected answer, so the approach seems right."]
    body = " ".join(parts)
    i = 0
    while len(body) < target_tokens * 4:
        c = rng.choice(CHECKS)
        if i % 5 == 3 and len(exs) > 1:
            c += f" For {exs[1]['input']} the answer should be {exs[1]['output']}."
        if i % 7 == 5:
            c += f" Alternatively, {algo_b} could also work here, but it is more complicated."
        body += " " + c + " " + rng.choice([
            "Let me write down the steps: iterate, update the state, and return the result at the end.",
            "So far: the state is updated at every position, and the answer is the best value seen.",
            "I think the logic holds. Let me also make sure the return type is an int as required.",
            "Right, that handles the case I was worried about.",
        ])
        i += 1
    return body[: int(target_tokens * 4)]


def answer_block(code):
    return "<answer>\nHere is a self-contained solution.\n\n```python\n" + code.rstrip() + "\n```\n</answer>"


def make_response(rng, rec, code, kind, max_tokens, mean_tokens):
    """kind: pass / fail（带 answer）、trunc（思考被截断）、tag（有代码没 answer 标签）。"""
    if kind == "trunc":
        return "<think>\n" + think_text(rng, rec, max_tokens), max_tokens   # 没有 </think>、没有 </answer>
    think_tok = max(200, int(rng.lognormvariate(math.log(max(mean_tokens - 260, 300)), 0.35)))
    think_tok = min(think_tok, max_tokens - 260)
    think = think_text(rng, rec, think_tok)
    tail = ("\n\n```python\n" + code.rstrip() + "\n```\n") if kind == "tag" else ("\n\n" + answer_block(code))
    return "<think>\n" + think + "\n</think>" + tail, think_tok + len(tail) // 4


# ----------------------------------------------------------------------------- 真实判分（本机执行参考解与变异体）
class Judge:
    def __init__(self):
        self.router, self.coder1, self.plus = se.load_reward_score()
        self.builder = se.load_builder()
        self.fail_cache, self.pass_cache = {}, {}

    def gt(self, rec):
        return self.builder.build_ground_truth(rec)

    def candidate_code(self, rec):
        return se.self_contained_candidate(rec["prompt"], rec["completion"])

    def reference_passes(self, rec):
        key = rec["task_id"]
        if key not in self.pass_cache:
            _, info = self.plus.compute_score_detailed(se.wrap_response(self.candidate_code(rec)), self.gt(rec), {"prompt": "x"})
            self.pass_cache[key] = info["status"] == "passed"
        return self.pass_cache[key]

    def failing_code(self, rng, rec):
        """返回一个真的过不了测试的变异体；都过得了就返回一个改坏返回值的版本。"""
        key = rec["task_id"]
        if key in self.fail_cache:
            return self.fail_cache[key]
        base = self.candidate_code(rec)
        gt = self.gt(rec)
        for _, code in cpt.generate_mutants(base, max_mutants=8, seed=rng.randrange(10 ** 6)):
            _, info = self.plus.compute_score_detailed(se.wrap_response(code), gt, {"prompt": "x"})
            if info["status"] == "failed":
                self.fail_cache[key] = code
                return code
        code = base.replace("return ", "return 0 if True else ", 1) if "return " in base else base
        self.fail_cache[key] = code
        return code

    def block(self, response, rec):
        score, info = self.plus.compute_score_detailed(response, self.gt(rec), {"prompt": "x"})
        marker = "✅" if score == FORMAT_R + ANSWER_R else "❌"
        return score, (marker * 16 + "Reward Calculation (coder1_plus)" + marker * 16 + "\n" + info["log"] + "\n" + marker * 16
                       + f"Final Reward = {score}" + marker * 16 + "\n\n")


# ----------------------------------------------------------------------------- 曲线
def _beta_moment(mean, conc, n, fail=False):
    a, b = mean * conc, (1 - mean) * conc
    if fail:
        a, b = b, a
    return math.exp(math.lgamma(a + n) + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(a + b + n))


def step_state(run, t, rng):
    # Base 模型共同的起点：三分之一的回答没按 <think>…</think><answer>…</answer> 写（tag），一部分循环到上限（trunc）
    tag = 0.32 * math.exp(-t / 12) + 0.01
    if run == "run-01":
        trunc = 0.18 * math.exp(-t / 20) + 0.06                      # 4096 上限：循环先多后少，后期思考变长又顶到上限
        p_cor = 0.30 + 0.30 * (1 - math.exp(-t / 120))
        length = 800 + 1100 * math.exp(-t / 18) + 900 * (1 - math.exp(-max(t - 40, 0) / 150))   # 先降（循环被罚）后升（思考变长）
        infra = INCIDENT.get(t, 0.0)
        kl = 0.070 * (1 - math.exp(-t / 110)) + (0.012 * math.exp(-max(t - 140, 0) / 25) if t >= 131 else 0)
        ent = 0.32 + 0.80 * math.exp(-t / 35)
        if 131 <= t <= 160:
            p_cor -= 0.025 * math.exp(-(t - 140) ** 2 / 200)
        clipfrac, ppo_kl, grad_norm, max_len = 0.0, 0.0, rng.gauss(0.42 + 0.50 * math.exp(-t / 40), 0.05), 4096
    else:
        trunc = 0.06 * math.exp(-t / 25) + 0.012                     # 8192 上限
        p_cor = 0.28 + 0.40 * (1 - math.exp(-t / 130))               # 筛掉全对全错后起点更低、上限更高
        length = 1200 + 1500 * math.exp(-t / 18) + 1500 * (1 - math.exp(-max(t - 40, 0) / 150))
        infra = 0.0
        kl = 0.085 * (1 - math.exp(-t / 110))
        ent = 0.30 + 0.85 * math.exp(-t / 35)
        clipfrac = max(0.0, rng.gauss(0.040, 0.010))
        ppo_kl = max(0.0, rng.gauss(6.0e-4, 1.5e-4))
        grad_norm, max_len = rng.gauss(0.40 + 0.50 * math.exp(-t / 40), 0.08), 8192
    fmt_fail = min(0.95, trunc + tag)
    p_cor = min(0.97, max(0.0, p_cor + rng.gauss(0, 0.012)))
    return dict(trunc=trunc, tag=tag, fmt_fail=fmt_fail, p_cor=p_cor, length=length, infra=infra, kl=kl, ent=ent,
                clipfrac=clipfrac, ppo_kl=ppo_kl, grad_norm=grad_norm, max_len=max_len)


def counts_for(fmt_fail, p_cor, infra, n=BATCH):
    """把比例变成整数计数：零分、格式失败、通过、失败。批均值只能落在 0.1/n 的格点上。"""
    n_zero = int(round(n * infra))
    scored = n - n_zero
    n_fmt = int(round(scored * fmt_fail))
    n_pass = int(round((scored - n_fmt) * p_cor))
    n_fail = scored - n_fmt - n_pass
    return n_zero, n_fmt, n_pass, n_fail


def mean_of(n_zero, n_fmt, n_pass, n_fail, n=BATCH):
    return ((FORMAT_R + ANSWER_R) * n_pass + FORMAT_R * n_fail - (FORMAT_R + ANSWER_R) * n_fmt) / n


def run_metrics(run):
    rng = random.Random(53 if run == "run-01" else 54)
    rows, wall = [], 0.0
    for t in range(1, STEPS + 1):
        s = step_state(run, t, rng)
        n_zero, n_fmt, n_pass, n_fail = counts_for(s["fmt_fail"], s["p_cor"], s["infra"])
        reward = mean_of(n_zero, n_fmt, n_pass, n_fail)
        all_pass = _beta_moment(s["p_cor"], 3.0, N_SAMPLE)
        all_fail = _beta_moment(s["p_cor"], 3.0, N_SAMPLE, fail=True) * (1 if s["fmt_fail"] < 0.1 else 0.85)
        if s["infra"] >= 1.0:
            all_pass, all_fail = 0.0, 0.0
        length = s["length"] + rng.gauss(0, 35)
        gen_tok_s = 2250 if s["max_len"] == 4096 else 1900      # 两个 TP=2 的 A800 副本，128 条一波
        gen = BATCH * length / gen_tok_s + 15 + rng.gauss(0, 10)
        reward_s = (25 if s["infra"] == 0 else 25 * (1 - s["infra"]) + 3) + rng.gauss(0, 1.0 if s["infra"] else 3)
        old_lp = (18 + length / 900) * COMPUTE_SLOWDOWN + rng.gauss(0, 6)
        ref = (14 + length / 1100) * COMPUTE_SLOWDOWN + rng.gauss(0, 6)
        adv = 0.4 + rng.gauss(0, 0.05)
        upd = ((70 if run == "run-01" else 92) + length / 400) * COMPUTE_SLOWDOWN + rng.gauss(0, 12)
        dump = 1.6 + length / 4000 + rng.gauss(0, 0.2)
        testing = (240 + rng.gauss(0, 12)) if (t % TEST_FREQ == 0) else 0.0
        save = (142 + rng.gauss(0, 8)) if (t % SAVE_FREQ == 0) else 0.0
        step_s = gen + reward_s + old_lp + ref + adv + upd + dump + testing + save + 4
        wall += step_s
        total_tok = BATCH * (PROMPT_TOK_MEAN + length)
        adv_max = rng.choice([3.75, 3.75, 2.56, 2.0]) if all_pass < 0.98 else 0.0
        adv_min = -rng.choice([3.75, 2.56, 2.0, 1.63]) if all_fail < 0.98 else 0.0
        if s["infra"] >= 1.0:
            adv_max = adv_min = 0.0
        r_min = 0.0 if n_zero else (-1.1 if n_fmt else 0.1)
        r_max = 0.0 if s["infra"] >= 1.0 else 1.1
        adv_mean = round(rng.gauss(0, 0.004), 4)
        kl = max(1e-5, s["kl"] * (1 + rng.gauss(0, 0.04)))
        row = {
            "step": t, "training/global_step": t, "training/epoch": (t - 1) // BATCHES_PER_EPOCH[run],
            "critic/score/mean": round(reward, 6), "critic/score/max": r_max, "critic/score/min": r_min,
            "critic/rewards/mean": round(reward, 6), "critic/rewards/max": r_max, "critic/rewards/min": r_min,
            "critic/advantages/mean": adv_mean, "critic/advantages/max": adv_max, "critic/advantages/min": adv_min,
            "critic/returns/mean": adv_mean, "critic/returns/max": adv_max, "critic/returns/min": adv_min,
            "response_length/mean": round(length, 1), "response_length/max": s["max_len"],
            "response_length/min": int(max(120, rng.gauss(420, 90))), "response_length/clip_ratio": round(max(0.0, s["trunc"] + rng.gauss(0, 0.004)), 4),
            "prompt_length/mean": round(PROMPT_TOK_MEAN + rng.gauss(0, 18), 1), "prompt_length/max": int(rng.gauss(690, 60)),
            "prompt_length/min": int(rng.gauss(118, 12)), "prompt_length/clip_ratio": 0.0,
            "actor/entropy": round(s["ent"] + rng.gauss(0, 0.01), 4),
            "actor/pg_loss": round(rng.gauss(-0.0012, 0.0025), 5),
            "actor/pg_clipfrac": round(s["clipfrac"], 4), "actor/ppo_kl": round(s["ppo_kl"], 6),
            "actor/pg_clipfrac_lower": 0.0,
            "actor/kl_loss": round(kl, 6), "actor/kl_coef": 0.001,
            "actor/grad_norm": round(s["grad_norm"], 4), "actor/lr": 5e-7,
            "perf/max_memory_allocated_gb": round(55.4 + length / 1500 + rng.gauss(0, 0.4), 2),
            "perf/max_memory_reserved_gb": round(63.0 + length / 1500 + rng.gauss(0, 0.4), 2),
            "perf/cpu_memory_used_gb": round(118 + rng.gauss(0, 2), 2),
            "perf/total_num_tokens": int(total_tok), "perf/time_per_step": round(step_s, 2),
            "perf/throughput": round(total_tok / (step_s * N_GPUS), 2),
            "timing_s/gen": round(gen, 2), "timing_s/reward": round(reward_s, 2), "timing_s/old_log_prob": round(old_lp, 2),
            "timing_s/ref": round(ref, 2), "timing_s/adv": round(adv, 3), "timing_s/update_actor": round(upd, 2),
            "timing_s/dump_rollout_generations": round(dump, 2), "timing_s/step": round(step_s, 2),
            "timing_per_token_ms/gen": round(gen * 1000 / (BATCH * length), 4),
            "timing_per_token_ms/ref": round(ref * 1000 / total_tok, 4),
            "timing_per_token_ms/adv": round(adv * 1000 / total_tok, 5),
            "timing_per_token_ms/update_actor": round(upd * 1000 / total_tok, 4),
        }
        if testing:
            row["timing_s/testing"] = round(testing, 2)
        if save:
            row["timing_s/save_checkpoint"] = round(save, 2)
        row.update({
            "scenario/n_zero": n_zero, "scenario/n_format_fail": n_fmt, "scenario/n_pass": n_pass, "scenario/n_fail": n_fail,
            "scenario/format_fail_rate": round(s["fmt_fail"], 4), "scenario/format_fail_truncation": round(s["trunc"], 4),
            "scenario/pass_rate_given_format": round(s["p_cor"], 4), "scenario/pass_rate_total": round((1 - s["fmt_fail"]) * s["p_cor"], 4),
            "scenario/infra_zero_fraction": s["infra"], "scenario/group_all_pass": round(all_pass, 4),
            "scenario/group_all_fail": round(all_fail, 4), "scenario/wall_seconds": round(wall, 1),
        })
        rows.append(row)
    return rows


def validation_curve(run):
    rng = random.Random(77 if run == "run-01" else 78)
    out = []
    for t in range(0, STEPS + 1, TEST_FREQ):
        if run == "run-01":
            fmt = 0.42 * math.exp(-t / 18) + 0.05                    # 贪心：标签缺失 + 循环到 4096
            p = 0.24 + 0.33 * (1 - math.exp(-t / 120)) - (0.02 if 131 <= t <= 160 else 0)
        else:
            fmt = 0.36 * math.exp(-t / 18) + 0.02                    # 8192 上限下循环少一些
            p = 0.26 + 0.35 * (1 - math.exp(-t / 130))
        if t:
            p += rng.gauss(0, 0.008)
        n_fmt = int(round(200 * fmt))
        n_pass = min(int(round(200 * p)), 200 - n_fmt)
        score = ((FORMAT_R + ANSWER_R) * n_pass + FORMAT_R * (200 - n_fmt - n_pass) - (FORMAT_R + ANSWER_R) * n_fmt) / 200
        out.append({"step": t, "val-core/code_plus/reward/mean@1": round(score, 6),
                    "scenario/val_n_pass": n_pass, "scenario/val_n_format_fail": n_fmt,
                    "scenario/val_pass_rate": round(n_pass / 200, 4), "scenario/val_format_fail": round(n_fmt / 200, 4)})
    return out


# ----------------------------------------------------------------------------- 配置
OVERRIDES = {
    "run-01": [
        "algorithm.adv_estimator=grpo", "data.train_files=/data/code-r1-plus-2k-leetcode2k-dedup/train.parquet",
        "data.val_files=/data/code-r1-plus-2k-leetcode2k/test.parquet", "data.train_batch_size=16", "data.max_prompt_length=2048",
        "data.max_response_length=4096", "data.filter_overlong_prompts=True", "data.filter_overlong_prompts_workers=8",
        "actor_rollout_ref.model.path=/root/models/Qwen3-8B-Base", "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True", "actor_rollout_ref.actor.optim.lr=5e-7",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=True", "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl", "actor_rollout_ref.actor.entropy_coeff=0.001",
        "actor_rollout_ref.actor.clip_ratio=0.2", "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.n=16", "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0", "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5", "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True", "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8", "reward_model.reward_manager=prime",
        "trainer.logger=[console,wandb]", "trainer.project_name=code-r1-qwen3", "trainer.experiment_name=run-01-qwen3-8b-lc2k",
        "trainer.n_gpus_per_node=4", "trainer.nnodes=1", "trainer.val_before_train=True", "trainer.test_freq=16",
        "trainer.save_freq=64", "trainer.total_epochs=3", "trainer.total_training_steps=320",
        "trainer.rollout_data_dir=/data/rollouts/run-01", "trainer.default_local_dir=/root/ckpt/run-01",
    ],
}
OVERRIDES["run-02"] = [o.replace("code-r1-plus-2k-leetcode2k-dedup", "code-r1-plus-2k-leetcode2k-filtered")
                       .replace("max_response_length=4096", "max_response_length=8192")
                       .replace("ppo_mini_batch_size=16", "ppo_mini_batch_size=4")
                       .replace("max_num_batched_tokens=8192", "max_num_batched_tokens=12288")
                       .replace("reward_manager=prime", "reward_manager=prime_plus")
                       .replace("run-01-qwen3-8b-lc2k", "run-02-qwen3-8b-lc2k-filtered")
                       .replace("total_epochs=3", "total_epochs=4")
                       .replace("/data/rollouts/run-01", "/data/rollouts/run-02")
                       .replace("/root/ckpt/run-01", "/data/ckpt/run-02") for o in OVERRIDES["run-01"]] + ["trainer.max_actor_ckpt_to_keep=3"]


def resolved_config(run):
    r1 = run == "run-01"
    return {
        "data": {"tokenizer": None, "use_shm": False,
                 "train_files": "/data/code-r1-plus-2k-leetcode2k-dedup/train.parquet" if r1 else "/data/code-r1-plus-2k-leetcode2k-filtered/train.parquet",
                 "val_files": "/data/code-r1-plus-2k-leetcode2k/test.parquet",
                 "prompt_key": "prompt", "reward_fn_key": "data_source", "max_prompt_length": 2048,
                 "max_response_length": 4096 if r1 else 8192, "train_batch_size": 16, "val_batch_size": None,
                 "shuffle": True, "validation_shuffle": False, "return_raw_input_ids": False, "return_raw_chat": False,
                 "return_full_prompt": False, "filter_overlong_prompts": True, "filter_overlong_prompts_workers": 8,
                 "truncation": "error", "trust_remote_code": False, "custom_cls": {"path": None, "name": None}},
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {"path": "/root/models/Qwen3-8B-Base", "custom_chat_template": None, "use_shm": False, "external_lib": None,
                      "override_config": {}, "enable_gradient_checkpointing": True, "enable_activation_offload": False,
                      "use_remove_padding": True, "lora_rank": 0, "lora_alpha": 16, "target_modules": "all-linear",
                      "use_liger": False, "use_fused_kernels": False, "trust_remote_code": False},
            "actor": {"strategy": "fsdp", "ppo_mini_batch_size": 16 if r1 else 4, "ppo_micro_batch_size": None,
                      "ppo_micro_batch_size_per_gpu": 4, "use_dynamic_bsz": False, "ppo_max_token_len_per_gpu": 16384,
                      "grad_clip": 1.0, "clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0,
                      "policy_loss": {"loss_mode": "vanilla"}, "loss_agg_mode": "token-mean", "entropy_coeff": 0.001,
                      "use_kl_loss": True, "kl_loss_coef": 0.001, "kl_loss_type": "low_var_kl", "use_torch_compile": True,
                      "ppo_epochs": 1, "shuffle": False, "ulysses_sequence_parallel_size": 1,
                      "checkpoint": {"save_contents": ["model", "optimizer", "extra"], "load_contents": ["model", "optimizer", "extra"]},
                      "optim": {"lr": 5e-7, "lr_warmup_steps": -1, "lr_warmup_steps_ratio": 0.0, "min_lr_ratio": 0.0,
                                "num_cycles": 0.5, "warmup_style": "constant", "total_training_steps": 320, "weight_decay": 0.01},
                      "fsdp_config": {"wrap_policy": {"min_num_params": 0}, "param_offload": False, "optimizer_offload": False,
                                      "offload_policy": False, "reshard_after_forward": True, "fsdp_size": -1, "forward_prefetch": False}},
            "ref": {"strategy": "fsdp", "fsdp_config": {"param_offload": False, "reshard_after_forward": True, "forward_prefetch": False,
                                                        "wrap_policy": {"min_num_params": 0}},
                    "log_prob_micro_batch_size": None, "log_prob_micro_batch_size_per_gpu": 8, "log_prob_use_dynamic_bsz": False,
                    "log_prob_max_token_len_per_gpu": 16384, "ulysses_sequence_parallel_size": 1},
            "rollout": {"name": "vllm", "mode": "sync", "temperature": 1.0, "top_k": -1, "top_p": 1.0, "use_fire_sampling": False,
                        "prompt_length": 2048, "response_length": 4096 if r1 else 8192, "dtype": "bfloat16",
                        "gpu_memory_utilization": 0.5, "ignore_eos": False, "enforce_eager": True, "free_cache_engine": True,
                        "load_format": "dummy_dtensor", "layered_summon": False, "tensor_model_parallel_size": 2,
                        "max_num_batched_tokens": 8192 if r1 else 12288, "max_model_len": None, "max_num_seqs": 1024,
                        "log_prob_micro_batch_size": None, "log_prob_micro_batch_size_per_gpu": 8, "log_prob_use_dynamic_bsz": False,
                        "log_prob_max_token_len_per_gpu": 16384, "disable_log_stats": True, "enable_chunked_prefill": True,
                        "do_sample": True, "n": 16, "multi_stage_wake_up": False,
                        "val_kwargs": {"top_k": -1, "top_p": 1.0, "temperature": 0, "n": 1, "do_sample": False},
                        "calculate_log_probs": False}},
        "critic": {"rollout_n": 16, "strategy": "fsdp", "optim": {"lr": 1e-5, "lr_warmup_steps_ratio": 0.0, "min_lr_ratio": None,
                                                                    "warmup_style": "constant", "total_training_steps": 320, "weight_decay": 0.01},
                   "model": {"path": "~/models/deepseek-llm-7b-chat", "tokenizer_path": "/root/models/Qwen3-8B-Base", "override_config": {},
                             "external_lib": None, "enable_gradient_checkpointing": True, "use_remove_padding": False,
                             "fsdp_config": {"param_offload": False, "optimizer_offload": False, "wrap_policy": {"min_num_params": 0}, "fsdp_size": -1}},
                   "ppo_mini_batch_size": 16 if r1 else 4, "ppo_micro_batch_size": None, "ppo_micro_batch_size_per_gpu": None,
                   "forward_micro_batch_size_per_gpu": None, "use_dynamic_bsz": False, "ppo_max_token_len_per_gpu": 32768,
                   "forward_max_token_len_per_gpu": 32768, "ulysses_sequence_parallel_size": 1, "ppo_epochs": 1, "shuffle": False,
                   "grad_clip": 1.0, "cliprange_value": 0.5, "checkpoint": {"save_contents": ["model", "optimizer", "extra"], "load_contents": ["model", "optimizer", "extra"]}},
        "reward_model": {"enable": False, "strategy": "fsdp", "reward_manager": "prime" if r1 else "prime_plus", "launch_reward_fn_async": False},
        "custom_reward_function": {"path": None, "name": "compute_score"},
        "algorithm": {"gamma": 1.0, "lam": 1.0, "adv_estimator": "grpo", "norm_adv_by_std_in_grpo": True,
                      "use_kl_in_reward": False, "kl_penalty": "kl",
                      "kl_ctrl": {"type": "fixed", "kl_coef": 0.001, "horizon": 10000, "target_kl": 0.1}},
        "trainer": {"balance_batch": True, "total_epochs": 3 if r1 else 4, "total_training_steps": 320, "project_name": "code-r1-qwen3",
                    "experiment_name": "run-01-qwen3-8b-lc2k" if r1 else "run-02-qwen3-8b-lc2k-filtered",
                    "logger": ["console", "wandb"], "log_val_generations": 0,
                    "rollout_data_dir": f"/data/rollouts/{run}", "validation_data_dir": None,
                    "nnodes": 1, "n_gpus_per_node": 4, "save_freq": 64, "resume_mode": "auto", "resume_from_path": None,
                    "val_before_train": True, "val_only": False, "test_freq": 16, "critic_warmup": 0,
                    "default_hdfs_dir": None, "del_local_ckpt_after_load": False,
                    "default_local_dir": "/root/ckpt/run-01" if r1 else "/data/ckpt/run-02",
                    "max_actor_ckpt_to_keep": None if r1 else 3, "max_critic_ckpt_to_keep": None,
                    "ray_wait_register_center_timeout": 300, "device": "cuda"},
        "ray_init": {"num_cpus": None},
    }


def to_yaml(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        out = ""
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out += f"{pad}{k}:\n" + to_yaml(v, indent + 1)
            else:
                out += f"{pad}{k}: {scalar(v)}\n"
        return out
    if isinstance(obj, list):
        return "".join(f"{pad}- {scalar(v)}\n" for v in obj)
    return pad + scalar(obj) + "\n"


def scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    if isinstance(v, list):
        return "[" + ", ".join(scalar(x) for x in v) + "]" if v else "[]"
    if isinstance(v, dict) and not v:
        return "{}"
    s = str(v)
    return f"'{s}'" if (":" in s or s.startswith("/") or s.startswith("~") or s in ("null", "true", "false") or " " in s) else s


# ----------------------------------------------------------------------------- 控制台日志
def ts(base, seconds, fmt="%Y-%m-%d %H:%M:%S"):
    return (base + dt.timedelta(seconds=seconds)).strftime(fmt)


def console_metric_line(row):
    parts = [f"step:{row['step']}"]
    for k, v in row.items():
        if k == "step" or k.startswith("scenario/"):
            continue
        if isinstance(v, (int, float)):
            parts.append(f"{k}:{v:.3f}")
    return " - ".join(parts)


QWEN3_CONFIG = """Qwen3Config {
  "_name_or_path": "/root/models/Qwen3-8B-Base",
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "pad_token_id": 151643,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}"""
BATCH_KEYS = ("_TensorDictKeysView(['input_ids', 'attention_mask', 'position_ids', 'prompts', 'responses', 'response_mask', "
              "'old_log_probs', 'ref_log_prob', 'token_level_scores', 'token_level_rewards', 'advantages', 'returns'])")


def error_line(rec_prompt_head):
    return f"[Error] Task failed: [Errno 28] No space left on device, completion: {rec_prompt_head}"


def header_lines(run, cfg, rng):
    base = T0[run]
    wid = WANDB_ID[run]
    n_rows = N_ROWS[run]
    lines = ["+ python3 -m verl.trainer.main_ppo " + " ".join(OVERRIDES[run]),
             f"{ts(base, 2)},311\tINFO worker.py:1852 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265",
             f"TaskRunner hostname: a800-4x-node07, PID: {rng.randrange(20000, 60000)}",
             pprint.pformat(cfg, width=120),
             "wandb: Tracking run with wandb version 0.19.8",
             "wandb: W&B syncing is set to `offline` in this directory.  ",
             "wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.",
             "Using dataset class: RLHFDataset",
             f"dataset len: {n_rows}",
             f"Filtering prompts longer than 2048 tokens (num_proc=8): 100%|██████████| {n_rows}/{n_rows} [00:02<00:00, {n_rows / 2.3:.2f} examples/s]",
             f"filter dataset len: {n_rows}",
             "Using dataset class: RLHFDataset",
             "dataset len: 200",
             "Filtering prompts longer than 2048 tokens (num_proc=8): 100%|██████████| 200/200 [00:00<00:00, 779.05 examples/s]",
             "filter dataset len: 200",
             f"Size of train dataloader: {BATCHES_PER_EPOCH[run]}, Size of val dataloader: 1",
             "Total training steps: 320",
             "[validate_config] All configuration checks passed successfully!"]
    wp = rng.randrange(20000, 60000)
    lines.append(f"(WorkerDict pid={wp}) Model config after override: " + QWEN3_CONFIG.replace("\n", "\n" + f"(WorkerDict pid={wp}) "))
    lines.append(f"(WorkerDict pid={wp}) wrap_policy: functools.partial(<function transformer_auto_wrap_policy at 0x7f3c1a2b4e60>, transformer_layer_cls={{<class 'transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer'>}})")
    lines.append(f"(WorkerDict pid={wp}) Total steps: 320, num_warmup_steps: 0")
    lines.append(f"Checkpoint tracker file does not exist: {cfg['trainer']['default_local_dir']}/latest_checkpointed_iteration.txt")
    lines.append("Training from scratch")
    return lines


def validation_lines(step, vrow):
    return [f"test_gen_batch meta info: {{'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True, 'global_steps': {step}}}",
            "validation generation end",
            "len reward_extra_infos_dict['reward']: 200"]


def write_console_log(run, cfg, rows, val, judge, recs, out_dir):
    rng = random.Random(101 if run == "run-01" else 102)
    base = T0[run]
    lines = header_lines(run, cfg, rng)
    v0 = val[0]["val-core/code_plus/reward/mean@1"]
    lines += validation_lines(0, val[0])
    lines.append(f"'Initial validation metrics: {{'val-core/code_plus/reward/mean@1': {v0}}}'")
    lines.append(f"step:0 - val-core/code_plus/reward/mean@1:{v0:.3f}")
    prime = cfg["reward_model"]["reward_manager"] == "prime"
    head_blocks, head_prompts = ([], []) if not prime else step_one_blocks(judge, recs, rng, rows[0])
    wall = 0.0
    for row in rows:
        t = row["step"]
        if prime:
            if t == 1:
                lines.extend(head_blocks)
            n_err = row["scenario/n_zero"]
            if n_err:
                for i in range(n_err):
                    rec = recs[(t * 131 + i * 7) % len(recs)]
                    head = ("<think>\nOkay, let's see. I need to solve this problem. The problem says: " + first_sentence(rec["meta"]["query"]))[:80]
                    lines.append(error_line(head.replace("\n", "\\n")))
            lines.append("[Shutdown] 64 subprocess(es) terminated.")
        else:
            if t in (77, 212, 288):
                rep = {"policy": "neutralize_group", "samples": BATCH, "infra_errors": 1 if t == 77 else 2, "neutralized_groups": 0 if t == 77 else 1,
                       "neutralized_samples": 0 if t == 77 else 16, "retries": 1 if t == 77 else 2}
                lines.append(f"[prime_plus] judge infrastructure errors: {rep}")
        lines.append(BATCH_KEYS)
        lines.append(f"Dumped generations to {cfg['trainer']['rollout_data_dir']}/{t}.jsonl")
        if t % TEST_FREQ == 0:
            vv = next(v for v in val if v["step"] == t)
            lines += validation_lines(t, vv)
            row = dict(row)
            row["val-core/code_plus/reward/mean@1"] = vv["val-core/code_plus/reward/mean@1"]
        if t % SAVE_FREQ == 0:
            d = cfg["trainer"]["default_local_dir"]
            lines.append(f"local_global_step_folder: {d}/global_step_{t}")
            keep = cfg["trainer"]["max_actor_ckpt_to_keep"]
            if keep and t // SAVE_FREQ > keep:
                lines.append(f"Checkpoint manager remove previous save local path: {d}/global_step_{t - keep * SAVE_FREQ}/actor")
        lines.append(console_metric_line(row))
        wall += row["timing_s/step"]
        if t % 10 == 0 or t == STEPS:
            remain = (STEPS - t) * (wall / t)
            lines.append(f"Training Progress: {100 * t // STEPS:3d}%|{'█' * (t * 10 // STEPS)}{' ' * (10 - t * 10 // STEPS)}| {t}/{STEPS} [{int(wall // 3600)}:{int(wall % 3600 // 60):02d}:{int(wall % 60):02d}<{int(remain // 3600)}:{int(remain % 3600 // 60):02d}:{int(remain % 60):02d}, {wall / t:.2f}s/it]")
    vlast = val[-1]["val-core/code_plus/reward/mean@1"]
    lines.append(f"'Final validation metrics: {{'val-core/code_plus/reward/mean@1': {vlast}}}'")
    text = "\n".join(lines) + "\n"
    (out_dir / "grpo.log.gz").write_bytes(gzip.compress(text.encode("utf-8")))
    head_end = next(i for i, l in enumerate(lines) if l.startswith("step:2 ")) + 1
    (out_dir / "grpo.head.log").write_text("\n".join(lines[:head_end]) + "\n", encoding="utf-8")
    if prime:
        start = next(i for i, l in enumerate(lines) if l.startswith("step:128 "))
        end = next(i for i, l in enumerate(lines) if l.startswith("step:136 "))
        (out_dir / "incident_step129-135.log").write_text("\n".join(lines[start:end + 1]) + "\n", encoding="utf-8")
    return text.count("\n")


def pick_prompts(judge, recs, rng):
    """抽 16 道参考解本身能通过补强版判分的题。"""
    chosen = []
    pool = list(recs)
    rng.shuffle(pool)
    for rec in pool:
        if judge.reference_passes(rec):
            chosen.append(rec)
        if len(chosen) == N_PROMPT:
            break
    return chosen


def sample_kinds(rng, row):
    """按该步的整数计数分配 256 条样本的类别，与指标行的均值精确一致。"""
    n_zero, n_fmt, n_pass, n_fail = row["scenario/n_zero"], row["scenario/n_format_fail"], row["scenario/n_pass"], row["scenario/n_fail"]
    n_trunc = int(round(n_fmt * row["scenario/format_fail_truncation"] / max(row["scenario/format_fail_rate"], 1e-9)))
    kinds = ["trunc"] * n_trunc + ["tag"] * (n_fmt - n_trunc) + ["pass"] * n_pass + ["fail"] * n_fail + ["zero"] * n_zero
    rng.shuffle(kinds)
    return kinds


def step_one_blocks(judge, recs, rng, row):
    """第 1 步的 256 条判分输出：用真实判分代码在本机执行参考解或变异体。"""
    prompts = pick_prompts(judge, recs, rng)
    kinds = sample_kinds(rng, row)
    blocks, used = [], []
    for i, kind in enumerate(kinds):
        rec = prompts[i // N_SAMPLE]
        code = judge.candidate_code(rec) if kind != "fail" else judge.failing_code(rng, rec)
        resp, _ = make_response(rng, rec, code, kind, 4096, row["response_length/mean"])
        score, block = judge.block(resp, rec)
        expected = {"trunc": -1.1, "tag": -1.1, "pass": 1.1, "fail": 0.1}[kind]
        assert abs(score - expected) < 1e-9, (kind, score, rec["task_id"])
        blocks.append(block.rstrip("\n"))
        used.append(rec)
    return blocks, used


def write_rollout_dump(run, cfg, rows, judge, recs, out_path, step):
    rng = random.Random(1000 + step)
    row = rows[step - 1]
    prompts = pick_prompts(judge, recs, rng)
    kinds = sample_kinds(rng, row)
    lines = []
    for i, kind in enumerate(kinds):
        rec = prompts[i // N_SAMPLE]
        real_kind = kind if kind != "zero" else rng.choice(["pass", "fail", "trunc"])
        code = judge.candidate_code(rec) if real_kind != "fail" else judge.failing_code(rng, rec)
        resp, _ = make_response(rng, rec, code, real_kind, cfg["data"]["max_response_length"], row["response_length/mean"])
        score = 0.0 if kind == "zero" else {"trunc": -1.1, "tag": -1.1, "pass": 1.1, "fail": 0.1}[kind]
        lines.append(json.dumps({"input": decoded_prompt(rec), "output": resp, "score": score, "step": step}, ensure_ascii=False))
    out_path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))


# ----------------------------------------------------------------------------- wandb / checkpoint
def write_wandb(run, cfg, rows, val, out_root):
    base = T0[run]
    d = out_root / "wandb" / f"offline-run-{base.strftime('%Y%m%d_%H%M%S')}-{WANDB_ID[run]}" / "files"
    d.mkdir(parents=True, exist_ok=True)
    y = ("wandb_version: 1\n\n_wandb:\n  desc: null\n  value:\n    cli_version: 0.19.8\n    python_version: 3.10.12\n"
         "    t:\n      1: [1, 11, 41, 49, 55, 63, 71, 98]\n      2: [1, 11, 41, 49, 55, 63, 71, 98]\n      3: [13, 16, 23, 61]\n"
         "      4: 3.10.12\n      5: 0.19.8\n      8: [5]\n      13: linux-x86_64\n")
    for k, v in cfg.items():
        y += f"{k}:\n  desc: null\n  value:\n" + to_yaml(v, 2)
    (d / "config.yaml").write_text(y, encoding="utf-8")
    meta = {"os": "Linux-5.15.0-113-generic-x86_64-with-glibc2.35", "python": "3.10.12", "startedAt": base.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "program": "/root/verl/verl/trainer/main_ppo.py", "codePath": "verl/trainer/main_ppo.py",
            "git": {"remote": "https://github.com/volcengine/verl.git", "commit": VERL_COMMIT},
            "root": "/root/verl", "host": "a800-4x-node07", "username": "root", "executable": "/usr/bin/python3", "cpu_count": 64, "cpu_count_logical": 128,
            "gpu": "NVIDIA A800-SXM4-80GB", "gpu_count": 4, "cuda": "12.4", "memory": {"total": "512 GiB"}, "args": OVERRIDES[run]}
    (d / "wandb-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    last = {k: v for k, v in rows[-1].items() if not k.startswith("scenario/") and k != "step"}
    last.update({"val-core/code_plus/reward/mean@1": val[-1]["val-core/code_plus/reward/mean@1"], "_step": STEPS,
                 "_runtime": rows[-1]["scenario/wall_seconds"],
                 "_timestamp": (base + dt.timedelta(seconds=rows[-1]["scenario/wall_seconds"])).timestamp()})
    (d / "wandb-summary.json").write_text(json.dumps(last, indent=2), encoding="utf-8")
    with open(d / "history.export.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"_step": 0, "_runtime": 0, "val-core/code_plus/reward/mean@1": val[0]["val-core/code_plus/reward/mean@1"]}) + "\n")
        for r in rows:
            rec = {k: v for k, v in r.items() if not k.startswith("scenario/")}
            rec["_step"] = r["step"]; rec["_runtime"] = r["scenario/wall_seconds"]
            rec["_timestamp"] = (base + dt.timedelta(seconds=r["scenario/wall_seconds"])).timestamp()
            if r["step"] % TEST_FREQ == 0:
                rec["val-core/code_plus/reward/mean@1"] = next(v for v in val if v["step"] == r["step"])["val-core/code_plus/reward/mean@1"]
            f.write(json.dumps(rec) + "\n")


def write_checkpoint_manifest(run, cfg, rows, out_root):
    d = out_root / "checkpoints" / run
    d.mkdir(parents=True, exist_ok=True)
    base = T0[run]
    root = "/data/ckpt-archive/run-01" if run == "run-01" else cfg["trainer"]["default_local_dir"]
    keep = [64, 128, 192, 256, 320] if run == "run-01" else [192, 256, 320]
    note = ("run-01 写在 /root/ckpt/run-01（根盘），第 134 步根盘写满后搬到 /data/ckpt-archive/run-01；mv 跨盘保留 mtime，所以时间戳仍是写入时间"
            if run == "run-01" else "max_actor_ckpt_to_keep=3：训练器只删 global_step_N/actor，global_step_64/128 的 data.pt 残留")
    lines = [f"# find {root} -type f -printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' | sort -k3   （{note}）"]
    per_rank_model = MODEL_PARAMS * 4 // N_GPUS
    per_rank_optim = MODEL_PARAMS * 8 // N_GPUS
    total = 0
    for step in keep:
        stamp = (base + dt.timedelta(seconds=rows[step - 1]["scenario/wall_seconds"])).strftime("%Y-%m-%d %H:%M")
        for rank in range(N_GPUS):
            for name, size in ((f"model_world_size_{N_GPUS}_rank_{rank}.pt", per_rank_model + 1_234_567 + rank * 17),
                               (f"optim_world_size_{N_GPUS}_rank_{rank}.pt", per_rank_optim + 2_345_678 + rank * 13),
                               (f"extra_state_world_size_{N_GPUS}_rank_{rank}.pt", 15_874 + rank)):
                lines.append(f"{size}\t{stamp}\t{root}/global_step_{step}/actor/{name}")
                total += size
        lines.append(f"45\t{stamp}\t{root}/global_step_{step}/actor/fsdp_config.json")
        for name, size in (("config.json", 726), ("generation_config.json", 239), ("tokenizer.json", 11_422_654), ("tokenizer_config.json", 9_732),
                           ("vocab.json", 2_776_833), ("merges.txt", 1_671_853), ("special_tokens_map.json", 613), ("added_tokens.json", 707)):
            lines.append(f"{size}\t{stamp}\t{root}/global_step_{step}/actor/huggingface/{name}")
            total += size
        lines.append(f"2174\t{stamp}\t{root}/global_step_{step}/data.pt")
    if run == "run-02":
        for step in (64, 128):
            stamp = (base + dt.timedelta(seconds=rows[step - 1]["scenario/wall_seconds"])).strftime("%Y-%m-%d %H:%M")
            lines.append(f"2174\t{stamp}\t{root}/global_step_{step}/data.pt")
    final_stamp = (base + dt.timedelta(seconds=rows[STEPS - 1]["scenario/wall_seconds"])).strftime("%Y-%m-%d %H:%M")
    lines.append(f"3\t{final_stamp}\t{root}/latest_checkpointed_iteration.txt")
    lines.append(f"# 合计 {total / 1e9:.1f} GB，每份 checkpoint 约 {(total / len(keep)) / 1e9:.1f} GB（模型 fp32 分片 {per_rank_model * N_GPUS / 1e9:.1f} GB + 优化器 {per_rank_optim * N_GPUS / 1e9:.1f} GB）")
    (d / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "latest_checkpointed_iteration.txt").write_text(str(STEPS), encoding="utf-8")


# ----------------------------------------------------------------------------- 评测明细
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def eval_problems():
    test = load_jsonl(DATASET / "LeetCodeDataset-v2-test-problems.jsonl")
    # 数据集里有拼写成 Meidum 的难度标签，按真实数据原样读入，归一化后使用
    dev = [{"task_id": r["task_id"], "question_id": r["meta"]["question_id"],
            "difficulty": {"meidum": "Medium"}.get(r["meta"]["difficulty"].lower(), r["meta"]["difficulty"].capitalize())} for r in test]
    rng = random.Random(2025)
    lcb, seen = [], set()
    counters = {"leetcode": 3541, "atcoder": 405, "codeforces": 2100}
    letters = {"atcoder": "cdefg", "codeforces": "BCDE"}
    for i in range(142):
        platform = rng.choices(["leetcode", "atcoder", "codeforces"], [0.46, 0.30, 0.24])[0]
        while True:
            if platform == "leetcode":
                qid = str(counters["leetcode"]); counters["leetcode"] += rng.choice([1, 1, 2])
            elif platform == "atcoder":
                qid = f"abc{counters['atcoder']}_{rng.choice(letters['atcoder'])}"
                if rng.random() < 0.35:
                    counters["atcoder"] += 1
            else:
                qid = f"{counters['codeforces']}{rng.choice(letters['codeforces'])}"
                if rng.random() < 0.5:
                    counters["codeforces"] += 1
            if qid not in seen:
                seen.add(qid); break
        date = dt.date(2025, 5, 1) + dt.timedelta(days=int(i * 1.05) + rng.randrange(0, 2))
        diff = rng.choices(["easy", "medium", "hard"], [0.3, 0.45, 0.25])[0]
        lcb.append({"question_id": qid, "platform": platform, "contest_date": date.isoformat(), "difficulty": diff})
    assert len({p["question_id"] for p in lcb}) == 142
    return dev, lcb


DIFF_OFFSET = {"Easy": 1.6, "Medium": 0.3, "Hard": -1.1, "easy": 1.2, "medium": -0.1, "hard": -1.4}
SKILL = {"dev": {"base": -1.22, "run-01": 0.07, "run-02": 0.43}, "lcb": {"base": -2.81, "run-01": -1.70, "run-02": -1.20}}
FMT_RATE = {"base": 0.20, "run-01": 0.03, "run-02": 0.02}   # 贪心与采样里格式错（标签缺失、循环到上限）的比例


def evaluate(dev, lcb):
    """每题一个潜在难度，三个模型共用；贪心结果用同一随机数比较阈值，所以模型间高度相关，
    只有少数题会从过变成不过；采样结果按题的通过概率独立抽样，题的概率呈两极分布，pass@8 才不会虚高。
    Base 起点：格式错按 FMT_RATE 同时吃掉贪心和采样，所以 pass@8 也会随训练大幅上升。"""
    rng = random.Random(11)
    results = {"dev_greedy": {}, "dev_sampled": {}, "lcb_greedy": {}, "lcb_sampled": {}}
    latent = {("dev", p["task_id"]): rng.gauss(0, 1.3) for p in dev}
    latent.update({("lcb", p["question_id"]): rng.gauss(0, 1.3) for p in lcb})
    greedy_u = {k: rng.random() for k in latent}
    fmt_u = {k: rng.random() for k in latent}
    fmt_rate = FMT_RATE
    for model in ("base", "run-01", "run-02"):
        for setname, probs, key in (("dev", dev, "task_id"), ("lcb", lcb, "question_id")):
            skill = SKILL[setname][model]
            greedy_rows, sampled_rows = [], []
            for p in probs:
                k = (setname, p[key])
                z = skill + 1.4 * (DIFF_OFFSET[p["difficulty"]] + latent[k])
                ps = sigmoid(z) * (1 - fmt_rate[model])              # 采样也会有格式错
                pg = sigmoid(z + 0.25)
                fmt = fmt_u[k] < fmt_rate[model]
                jitter = random.Random(seed_of("jitter", model, *k)).gauss(0, 0.06)
                passed = (not fmt) and (greedy_u[k] + jitter) < pg
                length = int(random.Random(seed_of("len", model, *k)).lognormvariate(math.log({"base": 1500, "run-01": 1700, "run-02": 2600}[model]), 0.5))
                greedy_rows.append({key: p[key], "difficulty": p["difficulty"], "format_ok": not fmt, "passed": passed,
                                    "reward": -1.1 if fmt else (1.1 if passed else 0.1), "response_tokens": min(length, 8192),
                                    "judge": "format_error" if fmt else ("passed" if passed else "failed")})
                seeds = {}
                for seed in (1, 2, 3):
                    srng = random.Random(seed_of("sample", model, *k, seed))
                    seeds[str(seed)] = sum(1 for _ in range(8) if srng.random() < ps)
                sampled_rows.append({key: p[key], "difficulty": p["difficulty"], "n": 8, "pass_counts_by_seed": seeds})
            results[f"{setname}_greedy"][model] = greedy_rows
            results[f"{setname}_sampled"][model] = sampled_rows
    return results


def summarize_eval(results):
    def agg(rows):
        return round(sum(r["passed"] for r in rows) / len(rows), 4)

    def fmt(rows):
        return round(sum(not r["format_ok"] for r in rows) / len(rows), 4)

    def sampled(rows):
        per_seed = {s: sum(r["pass_counts_by_seed"][s] for r in rows) / (8 * len(rows)) for s in ("1", "2", "3")}
        vals = list(per_seed.values())
        mean = sum(vals) / 3
        std = (sum((v - mean) ** 2 for v in vals) / 2) ** 0.5
        pass8_seed = {s: sum(1 for r in rows if r["pass_counts_by_seed"][s] > 0) / len(rows) for s in ("1", "2", "3")}
        return {"pass@1_by_seed": {k: round(v, 4) for k, v in per_seed.items()}, "pass@1_mean": round(mean, 4), "pass@1_std": round(std, 4),
                "pass@8_by_seed": {k: round(v, 4) for k, v in pass8_seed.items()}, "pass@8_mean": round(sum(pass8_seed.values()) / 3, 4)}

    def paired(a, b):
        win = sum(1 for x, y in zip(a, b) if y["passed"] and not x["passed"])
        loss = sum(1 for x, y in zip(a, b) if x["passed"] and not y["passed"])
        return {"win": win, "loss": loss, "same": len(b) - win - loss, "net": win - loss, "net_points": round(100 * (win - loss) / len(b), 1)}

    def by_diff(rows):
        out = {}
        for r in rows:
            d = r["difficulty"].lower()
            out.setdefault(d, [0, 0])
            out[d][0] += r["passed"]; out[d][1] += 1
        return {d: {"pass1": round(v[0] / v[1], 4), "n": v[1]} for d, v in out.items()}

    summary = {"models": ["base", "run-01", "run-02"]}
    for setname in ("dev", "lcb"):
        g, s = results[f"{setname}_greedy"], results[f"{setname}_sampled"]
        summary[setname] = {
            "n": len(g["base"]),
            "greedy_pass1": {m: agg(g[m]) for m in g},
            "greedy_pass_count": {m: sum(r["passed"] for r in g[m]) for m in g},
            "greedy_format_fail": {m: fmt(g[m]) for m in g},
            "greedy_by_difficulty": {m: by_diff(g[m]) for m in g},
            "sampled": {m: sampled(s[m]) for m in s},
            "paired_greedy": {"run-01_vs_base": paired(g["base"], g["run-01"]), "run-02_vs_base": paired(g["base"], g["run-02"]),
                              "run-02_vs_run-01": paired(g["run-01"], g["run-02"])},
        }
    # EvalPlus 是补全式贪心评测，没有格式门，Base 本来就能做；按 Qwen3-8B-Base 的 EvalPlus 量级（约 0.68，实跑前核对）写，训练后只涨 2–3 个点
    summary["evalplus"] = {"humaneval_plus_pass1": {"base": 0.671, "run-01": 0.689, "run-02": 0.701},
                           "mbpp_plus_pass1": {"base": 0.681, "run-01": 0.694, "run-02": 0.706},
                           "note": "EvalPlus 0.3.1，贪心，max_tokens 8192；数字为推演"}
    return summary


def eval_tables_md(summary):
    m = ["base", "run-01", "run-02"]
    def row(label, vals):
        return f"| {label} | " + " | ".join(vals) + " |\n"
    out = ""
    for setname, title in (("dev", "dev：LeetCodeDataset v2 test，200 题"), ("lcb", "held-out：LiveCodeBench 晚于数据截止的 142 题")):
        d = summary[setname]
        out += f"## {title}\n\n| | 基线 | run-01 | run-02 |\n|---|---:|---:|---:|\n"
        out += row("贪心 pass@1", [f"{d['greedy_pass1'][x]:.3f}（{d['greedy_pass_count'][x]}/{d['n']}）" for x in m])
        out += row("贪心格式失败率", [f"{d['greedy_format_fail'][x]:.3f}" for x in m])
        out += row("采样 pass@1（3 seed 均值 ± 样本标准差）", [f"{d['sampled'][x]['pass@1_mean']:.3f} ± {d['sampled'][x]['pass@1_std']:.3f}" for x in m])
        out += row("采样 pass@8（3 seed 均值）", [f"{d['sampled'][x]['pass@8_mean']:.3f}" for x in m])
        diffs = ["easy", "medium", "hard"]
        out += row("贪心按难度 easy / medium / hard", [" / ".join(f"{d['greedy_by_difficulty'][x][k]['pass1']:.3f}" for k in diffs) for x in m])
        out += f"\n按题配对（贪心，赢 / 输 / 平，净胜）：" + "；".join(
            f"{k.replace('_vs_', ' 对 ')} {v['win']} / {v['loss']} / {v['same']}，净胜 {v['net']} 题（{v['net_points']:+.1f} 个点）" for k, v in d["paired_greedy"].items()) + "。\n\n"
        out += "难度题数：" + "、".join(f"{k} {d['greedy_by_difficulty']['base'][k]['n']}" for k in diffs) + "。\n\n"
    e = summary["evalplus"]
    out += "## extra：EvalPlus（贪心）\n\n| | 基线 | run-01 | run-02 |\n|---|---:|---:|---:|\n"
    out += row("HumanEval+", [f"{e['humaneval_plus_pass1'][x]:.3f}" for x in m]) + row("MBPP+", [f"{e['mbpp_plus_pass1'][x]:.3f}" for x in m])
    return out


# ----------------------------------------------------------------------------- 归因
def attribution(rows):
    def pick(t):
        return next(r for r in rows if r["step"] == t)
    out = []
    for a, b in ((1, 64), (64, 128), (128, 192), (192, 320)):
        ra, rb = pick(a), pick(b)
        fa, fb = ra["scenario/format_fail_rate"], rb["scenario/format_fail_rate"]
        pa, pb = ra["scenario/pass_rate_given_format"], rb["scenario/pass_rate_given_format"]
        d = rb["critic/rewards/mean"] - ra["critic/rewards/mean"]

        def rew(f, p):
            ok = 1 - f
            return (FORMAT_R + ANSWER_R) * ok * p + FORMAT_R * ok * (1 - p) - (FORMAT_R + ANSWER_R) * f
        f_part = rew(fb, pa) - rew(fa, pa)
        c_part = rew(fb, pb) - rew(fb, pa)
        out.append({"from": a, "to": b, "d_reward": round(d, 3), "format_part": round(f_part, 3), "correctness_part": round(c_part, 3),
                    "residual": round(d - f_part - c_part, 3), "format_share": round(f_part / d, 2) if d else None})
    return out


# ----------------------------------------------------------------------------- 曲线
def plot(run_rows, run_val, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = {r: [x["step"] for x in rows] for r, rows in run_rows.items()}

    def fig(name, series, ylabel):
        plt.figure(figsize=(7.2, 3.6), dpi=130)
        for label, (xs, ys) in series.items():
            plt.plot(xs, ys, label=label, linewidth=1.2)
        plt.xlabel("step"); plt.ylabel(ylabel); plt.grid(alpha=0.3); plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out_dir / name); plt.close()

    for key, ylabel, name in (("critic/rewards/mean", "critic/rewards/mean", "reward.png"),
                              ("response_length/mean", "response_length/mean (tokens)", "response_length.png"),
                              ("response_length/clip_ratio", "response_length/clip_ratio (truncated fraction)", "clip_ratio.png"),
                              ("actor/kl_loss", "actor/kl_loss (low_var_kl)", "kl_loss.png"),
                              ("actor/entropy", "actor/entropy", "entropy.png"),
                              ("actor/pg_clipfrac", "actor/pg_clipfrac", "pg_clipfrac.png"),
                              ("actor/grad_norm", "actor/grad_norm", "grad_norm.png"),
                              ("timing_s/step", "timing_s/step (s)", "timing_step.png")):
        fig(name, {r: (steps[r], [x[key] for x in rows]) for r, rows in run_rows.items()}, ylabel)
    fig("val_core.png", {r: ([x["step"] for x in v], [x["val-core/code_plus/reward/mean@1"] for x in v]) for r, v in run_val.items()},
        "val-core/code_plus/reward/mean@1")
    fig("group_all_same.png", {f"{r} all_pass": (steps[r], [x["scenario/group_all_pass"] for x in rows]) for r, rows in run_rows.items()}
        | {f"{r} all_fail": (steps[r], [x["scenario/group_all_fail"] for x in rows]) for r, rows in run_rows.items()}, "fraction of groups (scenario)")
    return True


# ----------------------------------------------------------------------------- 主流程
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(RUN))
    parser.add_argument("--skip-judge", action="store_true", help="不在本机执行判分，不生成控制台日志和 rollout 落盘")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, builder = train_records()
    judge = None if args.skip_judge else Judge()
    report = {"generator": "scenario_qwen3_8b.py", "nature": "推演：参数模型生成，不是训练日志；判分块由真实判分代码在本机执行得到", "steps": STEPS}
    all_rows, all_val = {}, {}
    for run in ("run-01", "run-02"):
        cfg = resolved_config(run)
        rows = run_metrics(run)
        val = validation_curve(run)
        all_rows[run], all_val[run] = rows, val
        (out / "config").mkdir(exist_ok=True)
        (out / "config" / f"{run}.resolved.yaml").write_text(to_yaml(cfg), encoding="utf-8")
        logs = out / "logs" / run
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / "metrics.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("scenario/")}) + "\n")
        with open(logs / "scenario_internal.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"step": r["step"], **{k: v for k, v in r.items() if k.startswith("scenario/")}}) + "\n")
        with open(logs / "validation.jsonl", "w", encoding="utf-8") as f:
            for v in val:
                f.write(json.dumps(v) + "\n")
        if judge is not None:
            report[f"{run}_console_physical_lines"] = write_console_log(run, cfg, rows, val, judge, recs, logs)
            (out / "rollouts" / run).mkdir(parents=True, exist_ok=True)
            step = 131 if run == "run-01" else 320
            write_rollout_dump(run, cfg, rows, judge, recs, out / "rollouts" / run / f"{step}.jsonl.gz", step)
        write_wandb(run, cfg, rows, val, out)
        write_checkpoint_manifest(run, cfg, rows, out)
        hours = rows[-1]["scenario/wall_seconds"] / 3600
        base = T0[run]
        report[run] = {"wall_hours": round(hours, 1), "gpu_cost_usd": round(hours * GPU_HOUR_USD), "gpu_cost_cny": round(hours * GPU_HOUR_USD * USD_TO_CNY),
                       "started": base.isoformat(), "finished": (base + dt.timedelta(hours=hours)).isoformat(),
                       "step_end_times": {t: (base + dt.timedelta(seconds=rows[t - 1]["scenario/wall_seconds"])).strftime("%m-%d %H:%M:%S")
                                          for t in (64, 128, 129, 131, 134, 135, 192, 256, 320)},
                       "mean_step_s": round(sum(r["timing_s/step"] for r in rows) / STEPS, 1),
                       "mean_gen_s": round(sum(r["timing_s/gen"] for r in rows) / STEPS, 1),
                       "mean_update_s": round(sum(r["timing_s/update_actor"] for r in rows) / STEPS, 1),
                       "final": {k: rows[-1][k] for k in ("critic/rewards/mean", "response_length/mean", "response_length/clip_ratio", "actor/kl_loss",
                                                            "actor/entropy", "scenario/pass_rate_given_format", "scenario/group_all_pass", "scenario/group_all_fail")},
                       "val_first_last": [val[0]["val-core/code_plus/reward/mean@1"], val[-1]["val-core/code_plus/reward/mean@1"]],
                       "val_pass_first_last": [val[0]["scenario/val_pass_rate"], val[-1]["scenario/val_pass_rate"]],
                       "attribution": attribution(rows)}
    dev, lcb = eval_problems()
    results = evaluate(dev, lcb)
    ev = out / "eval"
    for setname in ("dev", "lcb"):
        for mode in ("greedy", "sampled"):
            d = ev / f"{setname}_{mode}"
            d.mkdir(parents=True, exist_ok=True)
            for model, rows_ in results[f"{setname}_{mode}"].items():
                with open(d / f"{model}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows_:
                        f.write(json.dumps(r) + "\n")
    (ev / "lcb_postcutoff_142.problems.jsonl").write_text("".join(json.dumps(p) + "\n" for p in lcb), encoding="utf-8")
    summary = summarize_eval(results)
    (ev / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ev / "tables.md").write_text("# 评测结果表（由 scenario_qwen3_8b.py 生成，report.md 的结论引用这里的数字）\n\n" + eval_tables_md(summary), encoding="utf-8")
    with open(ev / "paired_comparisons.csv", "w", encoding="utf-8") as f:
        f.write("set,comparison,win,loss,same,net,net_points\n")
        for setname in ("dev", "lcb"):
            for comp, v in summary[setname]["paired_greedy"].items():
                f.write(f"{setname},{comp},{v['win']},{v['loss']},{v['same']},{v['net']},{v['net_points']}\n")
    att = out / "attribution"
    att.mkdir(exist_ok=True)
    with open(att / "group_stats.csv", "w", encoding="utf-8") as f:
        f.write("run,step,group_all_pass,group_all_fail,effective_groups_of_16\n")
        for run, rows in all_rows.items():
            for r in rows:
                if r["step"] % 16 == 0 or r["step"] == 1:
                    f.write(f"{run},{r['step']},{r['scenario/group_all_pass']},{r['scenario/group_all_fail']},{round(16 * (1 - r['scenario/group_all_pass'] - r['scenario/group_all_fail']), 2)}\n")
    (att / "reward_decomposition.json").write_text(json.dumps({run: report[run]["attribution"] for run in all_rows}, indent=2), encoding="utf-8")
    report["eval"] = summary
    report["plots"] = plot(all_rows, all_val, out / "curves")
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "eval"}, ensure_ascii=False, indent=2))
    print(eval_tables_md(summary))


if __name__ == "__main__":
    main()
