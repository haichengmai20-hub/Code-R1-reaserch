"""补强项 6：判题基础设施故障的处理策略。纯 Python，不依赖 torch，可在 CPU 上单独测试。

原版 PrimeRewardManager 遇到评分超时或异常，把这一条直接记成 0.0，而 0.0 会进入同组的均值和标准差。
例如同组四条回答都格式合格但测试失败，本应都是 0.1；其中一条因判题故障被记成 0.0 后，
组内 advantage 变成 [+0.5, +0.5, +0.5, -1.5]，三条错误回答反而被鼓励。见 RLVR学习资料 03 作业 3.2。

这里改成：
1. 每条样本的评分带状态：scored 表示正常得分（格式失败、测试失败也算正常得分），infra_error 表示判题故障。
2. infra_error 先有限次重试。
3. 重试后仍失败的，按 uid 整组处理（neutralize_group）：整组分数设成同一个值，组内 advantage 全为 0，
   这组不提供 outcome 信号，KL 与熵项照常。legacy_zero 复现原版行为，便于对照。
4. 返回统计：故障条数、整组作废的组数与样本数。故障率超过阈值时应停下排查，而不是继续训练。
"""

import math


def score_with_retries(score_fn, max_retries=2):
    """调用 score_fn() -> (分数, 信息)。返回 (分数或 None, 状态, 尝试次数)。"""
    attempts = 0
    for attempts in range(1, max_retries + 2):
        try:
            score, info = score_fn()
        except Exception:  # 逃出判分器的异常一律视为判题故障，不归咎于候选回答
            continue
        if (info or {}).get("status") == "infra_error" or score is None:
            continue
        return float(score), "scored", attempts
    return None, "infra_error", attempts


def resolve_group_failures(scores, statuses, uids, policy="neutralize_group", neutral_value=0.0):
    """把判题故障转成最终分数。返回 (分数列表, 统计)。"""
    if not (len(scores) == len(statuses) == len(uids)):
        raise ValueError("scores, statuses and uids must have the same length")
    failed = [status == "infra_error" for status in statuses]
    report = {"policy": policy, "samples": len(scores), "infra_errors": sum(failed)}
    if policy == "legacy_zero":
        report.update(neutralized_groups=0, neutralized_samples=0)
        return [0.0 if bad else float(score) for score, bad in zip(scores, failed)], report
    if policy != "neutralize_group":
        raise ValueError(f"unknown policy: {policy}")
    bad_groups = {uid for uid, bad in zip(uids, failed) if bad}
    final = [neutral_value if uid in bad_groups else float(score) for score, uid in zip(scores, uids)]
    report.update(neutralized_groups=len(bad_groups), neutralized_samples=sum(1 for uid in uids if uid in bad_groups))
    return final, report


def grpo_advantages(scores, uids, epsilon=1e-6):
    """与 core_algos.compute_grpo_outcome_advantage 相同的组内标准化（样本标准差，单例组均值 0、标准差 1）。"""
    groups = {}
    for score, uid in zip(scores, uids):
        groups.setdefault(uid, []).append(score)
    stats = {}
    for uid, values in groups.items():
        if len(values) == 1:
            stats[uid] = (0.0, 1.0)
        else:
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            stats[uid] = (mean, std)
    return [(score - stats[uid][0]) / (stats[uid][1] + epsilon) for score, uid in zip(scores, uids)]
