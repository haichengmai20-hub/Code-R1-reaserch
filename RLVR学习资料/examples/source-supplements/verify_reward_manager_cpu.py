"""奖励管理器的 CPU 端到端检查：原版 PrimeRewardManager 与补强版 PrimePlusRewardManager。

和本目录其他脚本不同，这里导入真实的 verl 包（DataProto、torch 张量），所以要在装了依赖的虚拟环境里跑。
uv 管理的 Python 不允许直接 pip 安装，先建 venv（2026-10-15 在 macOS arm64、Python 3.12 实测可用的版本）：

    python3.12 -m venv ~/.venvs/code-r1-cpu
    ~/.venvs/code-r1-cpu/bin/pip install "numpy<2" pandas requests "torch==2.4.0" "tensordict==0.5.0" "ray==2.40.0"
    ~/.venvs/code-r1-cpu/bin/python -B RLVR学习资料/examples/source-supplements/verify_reward_manager_cpu.py

不需要 GPU、模型或 tokenizer：用一个假的 tokenizer 把 4 条固定回答放进同一个 uid 组，
回答分别是 LeetCode 53 的参考解、返回总和的错解、提前退出、缺格式标签。只执行这 4 个固定程序。

三种组合：
1. 原版管理器 + 原版 GT：没有 firejail 时，判题出错被记成 0.0（02 讲的缺口 3）。
2. 补强版管理器 + 补强版 GT：受保护协议加本机沙箱，得到 1.1、0.1、0.1、-1.1。
3. 补强版管理器 + 原版 GT：没有 firejail 时，判题故障先重试，仍失败就整组作废。
"""
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]

import supplement_env  # noqa: E402  只用 load_builder，不调用会打桩的 load_reward_score

try:
    import torch
    from verl import DataProto
    from verl.workers.reward_manager.prime import PrimeRewardManager
    from verl.workers.reward_manager.prime_plus import PrimePlusRewardManager
except ImportError as exc:  # 缺依赖时给出明确提示，不算检查失败
    print(f"缺依赖，跳过：{exc}。按本文件开头的说明建 venv 后再运行。")
    sys.exit(2)

REFERENCE = ("from typing import List\nclass Solution:\n    def maxSubArray(self, nums: List[int]) -> int:\n"
             "        ans = f = nums[0]\n        for x in nums[1:]:\n            f = max(f, 0) + x\n"
             "            ans = max(ans, f)\n        return ans")
WRONG_SUM = "class Solution:\n    def maxSubArray(self, nums):\n        return sum(nums)"
EARLY_EXIT = "import sys\nsys.exit(0)"
NAMES = ["reference", "wrong_sum", "early_exit", "bad_format"]
RESPONSES = [supplement_env.wrap_response(REFERENCE), supplement_env.wrap_response(WRONG_SUM),
             supplement_env.wrap_response(EARLY_EXIT), "```python\n" + REFERENCE + "\n```"]


class FixedTokenizer:
    """responses 张量的第一个 token 存回答编号，解码时取回对应的固定回答。"""

    def batch_decode(self, ids, skip_special_tokens=True):
        return [RESPONSES[int(row[0])] for row in ids]


def make_batch(data_source, ground_truth, extra_info):
    n, prompt_len, response_len = len(RESPONSES), 3, 5
    return DataProto.from_dict(
        tensors={"prompts": torch.zeros(n, prompt_len, dtype=torch.long),
                 "responses": torch.tensor([[i] * response_len for i in range(n)]),
                 "attention_mask": torch.ones(n, prompt_len + response_len, dtype=torch.long)},
        non_tensors={"data_source": [data_source] * n,
                     "reward_model": [{"style": "rule", "ground_truth": ground_truth}] * n,
                     "extra_info": [extra_info] * n,
                     "uid": ["maximum-subarray"] * n})


def run(manager, batch):
    """返回最后一个 token 上的分数，以及判分过程中被吞掉的执行错误条数。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):  # 判分器会为每条回答打印长日志
        tensor = manager(batch)
    swallowed = (out.getvalue() + err.getvalue()).count("FileNotFoundError")
    return [round(float(x), 3) for x in tensor[:, -1]], swallowed


def main():
    record = json.loads((REPO / "LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl")
                        .read_text(encoding="utf-8").splitlines()[142])
    assert record["task_id"] == "maximum-subarray"
    original_gt = json.dumps({"functional": f"{record['test']}\n\ncheck({record['entry_point'].strip()})"})
    plus_gt = supplement_env.load_builder().build_ground_truth(record)
    extra = {"prompt": "LeetCode 53 fixed candidates", "index": 0}
    has_firejail = shutil.which("firejail") is not None

    result = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "firejail": has_firejail, "plus_sandbox": os.environ.get("CODER1_PLUS_SANDBOX", "auto")}
    checks = {}

    scores, swallowed = run(PrimeRewardManager(FixedTokenizer(), 0), make_batch("code", original_gt, extra))
    result["original_manager_original_gt"] = dict(zip(NAMES, scores))
    result["original_manager_execution_errors_turned_into_0.0"] = swallowed
    expected = [1.1, 0.1, 1.1, -1.1] if has_firejail else [0.0, 0.0, 0.0, -1.1]
    checks["original_manager"] = scores == expected

    manager = PrimePlusRewardManager(FixedTokenizer(), 0)
    scores, _ = run(manager, make_batch("code_plus", plus_gt, extra))
    result["plus_manager_plus_gt"] = dict(zip(NAMES, scores))
    result["plus_manager_plus_gt_report"] = manager.last_report
    checks["plus_manager"] = scores == [1.1, 0.1, 0.1, -1.1] and manager.last_report["infra_errors"] == 0

    manager = PrimePlusRewardManager(FixedTokenizer(), 0, max_retries=1)
    scores, _ = run(manager, make_batch("code", original_gt, extra))
    result["plus_manager_original_gt"] = dict(zip(NAMES, scores))
    result["plus_manager_original_gt_report"] = manager.last_report
    if has_firejail:
        checks["plus_manager_on_original_gt"] = scores == [1.1, 0.1, 1.1, -1.1]
    else:
        checks["plus_manager_on_original_gt"] = (scores == [0.0, 0.0, 0.0, 0.0]
                                                 and manager.last_report["neutralized_groups"] == 1)

    result["checks"] = checks
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
