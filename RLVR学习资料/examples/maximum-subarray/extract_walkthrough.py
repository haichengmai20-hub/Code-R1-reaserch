"""提取流水线的直观演示：五种典型回答真实走过 ⓪①②③④，打印每步中间态。

用法（仓库根目录）：
    PYTHONUTF8=1 python -B "RLVR学习资料/examples/maximum-subarray/extract_walkthrough.py"

只演示「门③ 格式/提取」——走到三态和代码预览为止；执行与判分在门④之后
（沙箱执行见 run_demo.py 的 score 段）。每步调用的都是 verl 真实函数。
"""
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "RLVR学习资料/examples/source-supplements"))
import supplement_env as env  # noqa: E402

_, CODER1, _ = env.load_reward_score()

CORRECT = "class Solution:\n    def maxSubArray(self, nums):\n        ans = f = nums[0]\n        for x in nums[1:]:\n            f = max(f, 0) + x\n            ans = max(ans, f)\n        return ans\n"

CASES = {
    "A 严格合格": f"<think>用 Kadane 一次扫描。</think><answer>```python\n{CORRECT}```</answer>",
    "B 无标签有围栏": f"我的解法如下：\n```python\n{CORRECT}```\n以上。",
    "C answer 内草稿+正文": (
        "<think>先写草稿再写正式的。</think><answer>```python\n"
        "class Solution:\n    def maxSubArray(self, nums): return 0  # 草稿\n"
        "```\n```python\n" + CORRECT + "```\n</answer>"
    ),
    "D 标签齐全但代码裸贴": "<think>直接给代码。</think><answer>class Solution:\n    def maxSubArray(self, nums):\n        return 0\n</answer>",
    "E 尾随废话": f"<think>想清楚了。</think><answer>```python\n{CORRECT}```</answer>\n希望采纳，谢谢！",
}


def walk(name: str, solution_str: str):
    print("=" * 72)
    print(f"◆ {name}")
    print("-" * 72)
    stripped = solution_str.strip()                                     # ⓪
    strict = CODER1.validate_response_structure(solution_str)           # ①
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))  # ②
    if matches:
        scope = matches[-1].group(1)
        scope_desc = f"最后一个 <answer> 的内部（共 {len(matches)} 个 answer，取第 {len(matches)} 个）"
    else:
        scope = solution_str
        scope_desc = "整段原文（无任何 <answer> 标签 → fallback 的退回）"
    blocks = re.findall(CODER1.CODE_PATTERN, scope) if hasattr(CODER1, "CODE_PATTERN") \
        else re.findall(r"```(?:\w+)?\n(.*?)\n```", scope, re.DOTALL)   # ③
    code = "\n".join(blocks).strip()
    fmt = "+0.1" if strict else "-0.1"
    if code:
        state = "strict" if strict else "fallback"
        verdict = f"送去执行（格式分 {fmt}）"
    else:
        state = "none"
        verdict = "不执行，直接 -0.1（与是否写对无关——没有围栏就没有代码）"
    print(f"⓪ strip 后 {len(stripped)} 字符")
    print(f"① 格式检查：{'合格' if strict else '不合格'}（格式分 {fmt}）")
    print(f"② 提取范围：{scope_desc}")
    print(f"   范围预览：{scope[:60]!r}{'...' if len(scope) > 60 else ''}")
    print(f"③ 围栏块：{len(blocks)} 个 -> join 拼接")
    for i, b in enumerate(blocks, 1):
        first = b.splitlines()[0][:56] if b.splitlines() else ""
        print(f"   块{i}: {first!r}（共 {len(b.splitlines())} 行）")
    print(f"④ 三态：{state} → {verdict}")


def main():
    print("提取流水线走例：verl 真实函数 + 五种回答（LeetCode 53 贯穿题）")
    for name, case in CASES.items():
        walk(name, case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
