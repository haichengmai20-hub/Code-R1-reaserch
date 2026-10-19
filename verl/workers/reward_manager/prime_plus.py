"""补强项 6 的奖励管理器：PrimeRewardManager 的平行版本，原版 prime.py 不动。

与原版相同：并发评分、把标量奖励放在最后一个有效回答 token 上。
与原版不同：判题故障不再直接记 0.0，而是先重试，仍失败时按 uid 整组作废，策略见 infra_policy.py。
data_source 为 code_plus 时使用 coder1_plus 的结构化结果，能区分候选失败与判题故障；其他数据源沿用
_default_compute_score，只有抛出的异常会被当作判题故障。
在训练配置里写 reward_model.reward_manager=prime_plus 启用。

本模块依赖 torch，本仓库的 macOS 学习环境只测了 infra_policy.py 的纯逻辑；整体需要在 Linux 训练环境验证。
"""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from functools import partial
from multiprocessing import cpu_count

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

from . import infra_policy


def _score_once(compute_score, data_source, solution_str, ground_truth, extra_info):
    if data_source == "code_plus":
        from verl.utils.reward_score import coder1_plus
        if not isinstance(extra_info, dict) and hasattr(extra_info, "item"):
            extra_info = extra_info.item()
        return coder1_plus.compute_score_detailed(solution_str, ground_truth, extra_info)
    return compute_score(data_source, solution_str, ground_truth, extra_info), {"status": "scored"}


class PrimePlusRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, max_retries=2, policy="neutralize_group",
                 judge_timeout=600) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.max_retries = max_retries
        self.policy = policy
        self.judge_timeout = judge_timeout
        self.last_report = None

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        prompt_length = data.batch['prompts'].shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_strs = self.tokenizer.batch_decode(data.batch['responses'], skip_special_tokens=True)
        ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = [item.non_tensor_batch.get('extra_info', None) for item in data]
        # 训练时 fit() 已经按题分配 uid；验证时没有 uid，就让每条样本自成一组。
        uids = data.non_tensor_batch.get('uid')
        uids = list(uids) if uids is not None else [f"sample-{i}" for i in range(len(data))]

        with ThreadPoolExecutor(max_workers=max(cpu_count(), 1)) as pool:
            futures = [
                pool.submit(infra_policy.score_with_retries,
                            partial(_score_once, self.compute_score, source, text, truth, extra), self.max_retries)
                for source, text, truth, extra in zip(data_sources, sequences_strs, ground_truths, extra_infos)
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=self.judge_timeout))
                except FuturesTimeout:  # 判分器本身卡住，算判题故障
                    results.append((None, "infra_error", 0))

        scores, report = infra_policy.resolve_group_failures([r[0] for r in results], [r[1] for r in results], uids,
                                                             policy=self.policy)
        report["retries"] = sum(max(0, r[2] - 1) for r in results)
        self.last_report = report
        if report["infra_errors"]:
            print(f"[prime_plus] judge infrastructure errors: {report}")

        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
        return reward_tensor
