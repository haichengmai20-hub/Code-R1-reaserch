"""核对《05 实跑推演》、runs/qwen3-8b-grpo-20260928 和 runs/qwen3-8b-grpo-yourplan-20261018 下叙述文件里引用的数字，是否与生成器产物一致。

用法（仓库根目录）：$PY -B RLVR学习资料/examples/training/check_scenario_docs.py
退出码 0 表示全部对上；1 表示有数字没在对应文件里出现（改了生成器假设后要同步叙述）。
只检查「关键数字以规定格式出现在规定文件里」，不检查句子是否正确。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "qwen3-8b-grpo-20260928"
DOC = HERE.parents[1] / "05_实跑推演_Qwen3-8B.md"
s = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
ev = s["eval"]
r1, r2 = s["run-01"], s["run-02"]
internal = {json.loads(l)["step"]: json.loads(l) for l in (RUN / "logs/run-01/scenario_internal.jsonl").read_text(encoding="utf-8").splitlines() if l}
n_errors = sum(internal[t]["scenario/n_zero"] for t in range(129, 135))
val1 = [json.loads(l) for l in (RUN / "logs/run-01/validation.jsonl").read_text(encoding="utf-8").splitlines() if l]
val2 = [json.loads(l) for l in (RUN / "logs/run-02/validation.jsonl").read_text(encoding="utf-8").splitlines() if l]

def f3(x):
    return f"{x:.3f}"

expect = {
    "README.md": [f"{r1['wall_hours']:.1f} h", f"{r2['wall_hours']:.1f} h", f"${r1['gpu_cost_usd']}", f"${r2['gpu_cost_usd']}",
                  f"¥{r1['gpu_cost_cny']:,}", f"¥{r2['gpu_cost_cny']:,}", f3(val1[0]["val-core/code_plus/reward/mean@1"]), f3(val1[-1]["val-core/code_plus/reward/mean@1"]),
                  f3(val2[0]["val-core/code_plus/reward/mean@1"]), f3(val2[-1]["val-core/code_plus/reward/mean@1"]),
                  f3(ev["dev"]["greedy_pass1"]["base"]), f3(ev["dev"]["greedy_pass1"]["run-01"]), f3(ev["dev"]["greedy_pass1"]["run-02"]),
                  f3(ev["lcb"]["greedy_pass1"]["base"]), f3(ev["lcb"]["greedy_pass1"]["run-01"]), f3(ev["lcb"]["greedy_pass1"]["run-02"]),
                  f"净胜 {ev['lcb']['paired_greedy']['run-01_vs_base']['net']} 题", f"净胜 {ev['lcb']['paired_greedy']['run-02_vs_base']['net']} 题",
                  f3(r1["final"]["critic/rewards/mean"]), f3(r2["final"]["critic/rewards/mean"]),
                  r1["step_end_times"]["128"][-8:-3], r1["step_end_times"]["134"][-8:-3], r1["step_end_times"]["135"][-8:-3]],
    "notes/incident_report.md": [f"{n_errors:,} 行", str(json.load(open(RUN / "summary.json")) and internal[131]["scenario/n_zero"]),
                                 f3(val1[8]["val-core/code_plus/reward/mean@1"]), f3(val1[9]["val-core/code_plus/reward/mean@1"])],
    "attribution/reward_decomposition.md": [f"+{r1['attribution'][0]['d_reward']:.3f}", f"+{r1['attribution'][0]['format_part']:.3f}",
                                            f"{int(round(100 * r1['attribution'][0]['format_share']))}%", f"+{r2['attribution'][0]['d_reward']:.3f}"],
    "eval/report.md": [f3(ev["dev"]["greedy_pass1"]["run-01"]), f3(ev["lcb"]["greedy_pass1"]["run-02"]),
                       f"{ev['lcb']['sampled']['base']['pass@1_mean']:.3f} ± {ev['lcb']['sampled']['base']['pass@1_std']:.3f}",
                       f"{ev['lcb']['paired_greedy']['run-02_vs_base']['win']} / {ev['lcb']['paired_greedy']['run-02_vs_base']['loss']} / {ev['lcb']['paired_greedy']['run-02_vs_base']['same']}",
                       f"{ev['lcb']['paired_greedy']['run-01_vs_base']['net_points']:+.1f} 个点", f"{ev['lcb']['paired_greedy']['run-02_vs_base']['net_points']:+.1f} 个点"],
    "notes/decision_log.md": [f"{r1['wall_hours']:.1f} h", f"${r1['gpu_cost_usd']}", f"{r1['attribution'][0]['format_part']:.3f} / {r1['attribution'][0]['d_reward']:.3f}"],
}
doc_expect = [f"{r1['wall_hours']:.1f} 小时", f"{r2['wall_hours']:.1f} 小时", f"${r1['gpu_cost_usd']}", f"${r2['gpu_cost_usd']}",
              f3(ev["dev"]["greedy_pass1"]["base"]), f3(ev["dev"]["greedy_pass1"]["run-01"]), f3(ev["dev"]["greedy_pass1"]["run-02"]),
              f3(ev["lcb"]["greedy_pass1"]["base"]), f3(ev["lcb"]["greedy_pass1"]["run-01"]), f3(ev["lcb"]["greedy_pass1"]["run-02"]),
              f3(val1[0]["val-core/code_plus/reward/mean@1"]), f3(val1[-1]["val-core/code_plus/reward/mean@1"]), f3(val2[0]["val-core/code_plus/reward/mean@1"]), f3(val2[-1]["val-core/code_plus/reward/mean@1"]),
              f"{r1['attribution'][0]['format_part']:.3f}", f"{int(round(100 * r1['attribution'][0]['format_share']))}%", f"{n_errors:,} 行",
              f"净胜 {ev['lcb']['paired_greedy']['run-01_vs_base']['net']} 题", f"净胜 {ev['lcb']['paired_greedy']['run-02_vs_base']['net']} 题",
              f"{ev['lcb']['paired_greedy']['run-02_vs_base']['net_points']:+.1f}", str(internal[131]["scenario/n_zero"])]

# 「你的 run」（scenario_yourplan.py 的产物，两轮）与它的叙述文件、你的决策日志
YOUR = HERE / "runs" / "qwen3-8b-grpo-yourplan-20261018"
y = json.loads((YOUR / "summary.json").read_text(encoding="utf-8"))
y1, y2, yev = y["your-01"], y["your-02"], y["eval"]
ymet1 = {json.loads(l)["step"]: json.loads(l) for l in (YOUR / "logs/your-01/metrics.jsonl").read_text(encoding="utf-8").splitlines() if l}
ymet2 = {json.loads(l)["step"]: json.loads(l) for l in (YOUR / "logs/your-02/metrics.jsonl").read_text(encoding="utf-8").splitlines() if l}
pd, pl = yev["dev"]["paired_greedy"], yev["lcb"]["paired_greedy"]
y_eval = [f3(yev["dev"]["greedy_pass1"][m]) for m in ("base", "your-01", "your-02")] + [f3(yev["lcb"]["greedy_pass1"][m]) for m in ("base", "your-01", "your-02")] + \
         [f"净胜 {pd['your-01_vs_base']['net']} 题", f"净胜 {pd['your-02_vs_your-01']['net']} 题", f"净胜 {pl['your-01_vs_base']['net']} 题", f"净胜 {pl['your-02_vs_your-01']['net']} 题",
          f"{pl['your-02_vs_your-01']['net_points']:+.1f} 个点", f"{pl['your-01_vs_base']['net_points']:+.1f} 个点"]
your_expect = {
    "README.md": [f"{y1['wall_hours']:.1f} h", f"¥{y1['gpu_cost_cny']:,}", f"{y2['wall_hours']:.1f} h", f"¥{y2['gpu_cost_cny']:,}",
                  f3(y1["val_first_last"][0]), f3(y1["val_first_last"][1]), f3(y2["val_first_last"][0]), f3(y2["val_first_last"][1]),
                  f3(ymet1[1]["critic/rewards/mean"]), f3(ymet1[200]["critic/rewards/mean"]), f3(ymet2[201]["critic/rewards/mean"]), f3(ymet2[400]["critic/rewards/mean"]),
                  f"{y1['mean_step_s']:.0f} s", f"{y1['mean_gen_s']:.0f} s", f"{y1['mean_update_s']:.0f} s", f"{y2['mean_step_s']:.0f} s",
                  f"全对 {y['filter']['all_pass_removed']} 题、全错 {y['filter']['all_fail_removed']} 题", f"训练 {y['filter']['train_rows']} 行",
                  f3(ymet1[65]["critic/rewards/mean"]), f"{y1['neutralized_groups_total']} 组"] + y_eval,
    "eval/report.md": y_eval + [f"赢 {pd['your-01_vs_base']['win']} / 输 {pd['your-01_vs_base']['loss']}",
                                f"{yev['dev']['sampled']['base']['pass@1_mean']:.3f} ± {yev['dev']['sampled']['base']['pass@1_std']:.3f}",
                                f"{yev['dev']['sampled']['your-02']['pass@8_mean']:.3f}", f"{yev['lcb']['sampled']['your-02']['pass@8_mean']:.3f}"],
    "notes/incident_report.md": [f3(ymet1[t]["critic/rewards/mean"]) for t in (65, 129, 193)] + [f"{y1['neutralized_groups_total']} 组", f"{ymet2[257]['timing_s/reward']:.1f}"],
    "notes/your-02_plan.md": [f"{y2['wall_hours']:.1f} 小时", f"¥{y2['gpu_cost_cny']:,}", f"{ymet2[257]['timing_s/reward']:.1f}", f"{yev['dev']['sampled']['your-02']['pass@8_mean']:.3f}",
                              f"全对 {y['filter']['all_pass_removed']}、全错 {y['filter']['all_fail_removed']}"],
    "attribution/reward_decomposition.md": [f"+{y1['attribution'][0]['d_reward']:.3f}", f"+{y1['attribution'][0]['format_part']:.3f}",
                                            f"{int(round(100 * y1['attribution'][0]['format_share']))}%", f"+{y2['attribution'][0]['d_reward']:.3f}"],
}
decisions_expect = [f"{y1['wall_hours']:.1f} 小时", f"¥{y1['gpu_cost_cny']:,}", f"{y2['wall_hours']:.1f} 小时", f"¥{y2['gpu_cost_cny']:,}", f3(ymet1[65]["critic/rewards/mean"]),
                    f"+{y1['attribution'][0]['d_reward']:.3f}", f"{int(round(100 * y1['attribution'][0]['format_share']))}%",
                    f3(yev["lcb"]["greedy_pass1"]["your-01"]), f3(yev["lcb"]["greedy_pass1"]["your-02"]), f"净胜 {pl['your-01_vs_base']['net']} 题", f"净胜 {pl['your-02_vs_your-01']['net']} 题",
                    f"{yev['dev']['sampled']['your-01']['pass@8_mean']:.3f}", f"{yev['dev']['sampled']['your-02']['pass@8_mean']:.3f}"]
doc_expect.append(f"{pl['your-02_vs_your-01']['net_points']:+.1f} 个点")

def read(path):
    return path.read_text(encoding="utf-8").replace("\u2212", "-")   # 叙述里的负号写成 U+2212 也算对上

missing = []
for rel, items in expect.items():
    text = read(RUN / rel)
    missing += [(rel, it) for it in items if it not in text]
for rel, items in your_expect.items():
    text = read(YOUR / rel)
    missing += [(f"yourplan/{rel}", it) for it in items if it not in text]
dec = read(RUN / "notes/your_decisions.md")
missing += [("notes/your_decisions.md", it) for it in decisions_expect if it not in dec]
doc = read(DOC)
missing += [("05_实跑推演_Qwen3-8B.md", it) for it in doc_expect if it not in doc]
if missing:
    print("数字未在叙述文件里找到（生成器产物与叙述不一致）：")
    for rel, it in missing:
        print(f"  {rel}: {it!r}")
    sys.exit(1)
print(f"check_scenario_docs: {sum(len(v) for v in expect.values()) + len(doc_expect) + sum(len(v) for v in your_expect.values()) + len(decisions_expect)} 个数字全部对上")
