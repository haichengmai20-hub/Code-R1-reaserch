"""补强项 10：性能轮——每题时间复杂度约束的标定与判定。

原版判分只测输出正确性：O(n^2) 与 O(n) 的正确回答同分，RL 看不见快慢（学习资料 02 的
「执行边界」一节；2026-11-07 讨论定案）。本模块给判分器补一个「性能轮」：

- 数据侧（``examples/data_preprocess/coder1_plus_perf.py``）离线标定：从 ``question_title`` 的
  Constraints 段解析规模上界（上标丢失按 10 的幂识别：字面 ``105`` 视作 ``10^5``），把现有
  测试的输入按上界放大成大输入，参考解实测耗时 × k 得 ``time_limit_ms``，连同确定性生成参数
  写进数据行 ``extra_info.perf``。一次性离线产物，训练时零标定开销；同一题所有 step 用同
  一个值，判分标准不随负载漂移。
- 判分侧：正确性轮（现有协议）通过后，若 ``extra_info`` 带 ``perf`` 字段则加跑性能轮——用
  记录的种子确定性重建大输入，timeout 用每题标定值；超时或大输入上输出不符都算「对但有
  规模相关瑕疵」。判分用的是 ``sandbox.run_python``，所以隔离、计时、状态分类与正确性轮
  完全一致。
- 分值（瑕疵并罚制，接第 9 项字典序）：正确性 1.0/0 保持主键；性能瑕疵 −0.2，与「对但
  无标签」并列为 0.9 档——对+格式+快 1.1、对+一项瑕疵 0.9、对+两项瑕疵 0.7，任何对答
  仍严格高于任何错答。``CODER1_PLUS_REWARD=legacy`` 时性能轮整体跳过（推演口径不变）。
- 设计依据：CodeContests 每题带 ``time_limit`` 数据字段（``contest_problem.proto``）；
  Codeforces Polygon 出题用预期解实测 × 系数标定 TL，并用故意慢的解验证 TL 确实拒绝它。

局限（第一版，超出范围一律跳过、判分行为退回第 9 项）：只支持「恰有一个 List[int] 参数」
的题；性能输出必须可 repr 且小于 4 KB；解析不出规模上界的题不标定。树/链表、stdio、
多参数题待后续版本。
"""
import ast
import re
import subprocess
import sys
import time

MAX_PERF_SIZE = 100_000   # 生成输入的规模上限（防内存爆），实际取 min(约束上界, 此值)
MIN_PERF_SIZE = 1_000     # 规模下限（太小没有复杂度区分度）
DEFAULT_LO, DEFAULT_HI = -10_000, 10_000
TIME_FACTOR = 3.0         # Polygon 惯例：参考解实测 × k；含进程启动开销
MIN_TIME_LIMIT_MS = 500   # 下限：解释器启动本身的抖动要吃掉一两百毫秒
MAX_EXPECTED_CHARS = 4096

_POW10 = re.compile(r"10[1-9]|10{2,}")                 # '105'=10^5（上标丢失）、'1000'=10^3（写全）
_CONSTRAINT_HEAD = re.compile(r"constraints", re.IGNORECASE)
_SIZE_HINT = re.compile(
    r"(?:[\w.\[\]]*len(?:gth)?\s*<=\s*|(?<![\w.])n\s*<=\s*|\|[\w.\[\]]+\|\s*<=\s*)"  # nums.length / n / |x|
    r"(10{2,}|\d+)"
    r"(?:\s*\*\*\s*(\d+)|\s*\*\s*(10{2,}))?")        # 可选尾巴：'10**5' 或 '5 * 104'


def _as_pow10(token: str):
    """'105' -> 10^5（上标丢失）；'1000' -> 10^3（写全）；非 10 的幂返回 None。"""
    m = re.fullmatch(r"10([1-9])", token)
    if m:
        return 10 ** int(m.group(1))
    m = re.fullmatch(r"1(0{2,})", token)
    if m:
        return 10 ** len(m.group(1))
    return None


def parse_size_bound(question_title: str):
    """从完整题面的 Constraints 段解析主输入的规模上界；解析不出返回 None。

    覆盖三类真实写法：``nums.length <= 105``（上标丢失，按 10^5 解释）、``n <= 10**5``（pow 形式）、
    ``s.length <= 5 * 104``（系数乘幂，按 5×10^4 解释）。值域约束（``nums[i] <= 104``）不会被
    误当作规模。解析不出或小于下限的返回 None（该题跳过性能轮）。
    """
    if not question_title:
        return None
    hit = _CONSTRAINT_HEAD.search(question_title)
    tail = question_title[hit.end():] if hit else question_title
    found = _SIZE_HINT.search(tail)
    if not found:
        return None
    token, pow_exp, mul_pow = found.group(1), found.group(2), found.group(3)
    if pow_exp:                                     # '10**5'
        value = int(token) ** int(pow_exp)
    elif mul_pow:                                   # '5 * 104' -> 5 × 10^4
        value = int(token) * _as_pow10(mul_pow)
    elif _as_pow10(token):                          # '105' -> 10^5（上标丢失）
        value = _as_pow10(token)
    else:
        value = int(token)
    return value if value >= MIN_PERF_SIZE else None


def parse_example_input(example_input: str):
    """把 examples[0].input（如 ``nums = [-2, 1, -3]``）解析成 {参数名: 字面值}。"""
    args = {}
    for chunk in example_input.split("\n"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, _, literal = chunk.partition("=")
        try:
            args[name.strip()] = ast.literal_eval(literal.strip())
        except (ValueError, SyntaxError):
            return None
    return args or None


def build_perf_spec(arg_name, size, seed, lo=DEFAULT_LO, hi=DEFAULT_HI):
    """性能轮的确定性输入规格：判分端用它重建同样的大输入。"""
    return {"arg": arg_name, "kind": "list_int", "size": int(size), "seed": int(seed),
            "lo": int(lo), "hi": int(hi)}


def render_perf_program(prelude: str, solution_code: str, entry: str, spec: dict) -> str:
    """渲染性能轮程序：前缀 + 候选代码 + 确定性生成大输入 + 调用并 print(repr(结果))。

    判分方在进程外比对输出（受保护协议的精神：期望答案不进候选进程）。
    """
    gen = (f"_coder1_rng.randint({spec['lo']}, {spec['hi']}) "
           f"for _ in range({spec['size']})")
    return f"""{prelude.rstrip()}

{solution_code.rstrip()}

import random as _coder1_perf_random
_coder1_rng = _coder1_perf_random.Random({spec['seed']})
_coder1_perf_arg = [{gen}]
print(repr({entry.strip()}({spec['arg']}=_coder1_perf_arg)))
"""


def _classify_perf(result):
    if result.status == "infra_error":
        return "infra"
    if result.status == "timeout":
        return "slow"
    if result.ok:
        return "pass" if result.stdout.strip() else "mismatch"   # 空输出视为不符
    return "error"


def run_perf_round(gt: dict, solution_code: str, perf: dict, sandbox):
    """判分侧入口：跑一次性能轮。返回 {"status": pass|slow|mismatch|error|infra, ...}。

    ``perf`` 是数据行 ``extra_info.perf``：{"spec": {...}, "time_limit_ms": N,
    "expected": "<repr>"}。调用方负责把 slow/mismatch/error 记为性能瑕疵、infra 抛
    InfraError 走整组作废（补强项 6）。
    """
    from . import sandbox as _default_sandbox
    box = sandbox or _default_sandbox
    spec = perf["spec"]
    entry = perf.get("entry") or ""
    if not entry:
        return {"status": "error", "reason": "perf field lacks entry point"}
    program = render_perf_program(gt.get("prelude", ""), solution_code, entry, spec)
    started = time.monotonic()
    result = box.run_python(program, timeout=perf["time_limit_ms"] / 1000.0)
    status = _classify_perf(result)
    out = {"status": status, "exec_status": result.status,
           "duration_s": round(time.monotonic() - started, 3),
           "time_limit_ms": perf["time_limit_ms"]}
    if status in ("pass", "mismatch"):
        out["output_matches"] = result.stdout.strip() == perf.get("expected", "")
        if status == "pass":
            status = "pass" if out["output_matches"] else "mismatch"
            out["status"] = status
    return out


def calibrate_one(task: dict, seed: int = 20261107, factor: float = TIME_FACTOR,
                  run=None):
    """数据侧入口：对一道原始 LeetCodeDataset 记录做离线标定。

    返回 perf 字段（{"spec": ..., "time_limit_ms": ..., "expected": ...}）或 None（跳过）。
    ``run`` 是执行器（(program, timeout) -> (ok, stdout, duration)），默认本地 subprocess——
    标定跑的是数据集自带的参考解，是可信代码，不需要沙箱。
    """
    title = (task.get("meta") or {}).get("question_title", "")
    size = parse_size_bound(title)
    if size is None:
        return None, "no_size_bound"
    size = max(MIN_PERF_SIZE, min(size, MAX_PERF_SIZE))
    examples = task.get("examples") or []
    if not examples:
        return None, "no_examples"
    args = parse_example_input(examples[0].get("input", ""))
    if not args or len(args) != 1:
        return None, "unsupported_args"
    name, value = next(iter(args.items()))
    if not (isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool)
                                                      for v in value)):
        return None, "unsupported_arg_type"
    lo, hi = DEFAULT_LO, DEFAULT_HI
    reference = task.get("completion") or ""
    entry = (task.get("entry_point") or "").strip()
    prelude = task.get("prompt") or ""
    if not reference or not entry:
        return None, "no_reference"

    def _run(program, timeout):
        if run is not None:
            return run(program, timeout)
        started = time.monotonic()
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", program],
                              capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0, proc.stdout, time.monotonic() - started

    spec = build_perf_spec(name, size, seed)
    probe = render_perf_program(prelude, reference, entry, spec)
    try:
        ok, stdout, duration = _run(probe, 30)
    except subprocess.TimeoutExpired:
        return None, "reference_too_slow"
    if not ok or not stdout.strip():
        return None, "reference_failed"
    expected = stdout.strip()
    if len(expected) > MAX_EXPECTED_CHARS:
        return None, "expected_too_long"
    # 第二次实测取较慢者，吸收一点负载抖动（Polygon 式宽余）。
    try:
        ok2, _, duration2 = _run(probe, 30)
    except subprocess.TimeoutExpired:
        duration2 = 30.0
    wall = max(duration, duration2 if ok2 else duration)
    time_limit_ms = max(MIN_TIME_LIMIT_MS, int(round(factor * 1000 * wall)))
    return {"spec": spec, "entry": entry, "time_limit_ms": time_limit_ms, "expected": expected,
            "calibrated": {"reference_wall_s": round(wall, 3), "factor": factor,
                           "size": size}}, "ok"
