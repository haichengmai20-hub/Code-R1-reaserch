"""Code-R1 数据预处理的补强版（coder1_plus）。

原版 examples/data_preprocess/coder1.py 一个字都不改。本脚本只补学习资料（RLVR学习资料/）指出、
原版缺少的环节，每补一项都在这里登记：

1. GT 执行契约（2026-10-14）
   LeetCode 题的 GT 额外保存原始 ``prompt`` 前缀，字段名 ``prelude``。判分时按「前缀、候选代码、测试」
   的顺序执行，与 LeetCodeDataset 原生评测器一致，修好测试依赖前缀辅助函数的 121 道树和链表题。
   判分侧见 verl/utils/reward_score/coder1_plus/。代价：前缀里有 ``from typing import *`` 等导入，
   模型漏写 imports 不再被判错，这与原生评测器和 LeetCode 官方判题的行为一致。
2. 可追溯的 metadata
   extra_info 在原版 5 个字段之外，再存 task_id、question_id、来源文件与行号、GT 契约版本和测试版本。
7. 性能轮标定（补强项 10 的数据侧）
   ``--perf_index`` 指向 coder1_plus_perf.py 产出的 perf-index.jsonl，把每题标定的 perf 字段
   （确定性大输入规格、取自参考解实测的 time_limit_ms、期望输出）并进 ``extra_info.perf``。
   判分器据此加跑性能轮；没有标定的题不写该字段，性能轮自动跳过。
3. 任意顺序的比较规则（补强项 4 的数据侧）
   题面写明「in any order」的题，GT 加上 compare=unordered_outer，判分时列表答案忽略外层顺序。
5. 去污染与难度筛选（补强项 7）
   --drop_task_ids 去掉与评测集重叠的题，--keep_task_ids 只保留按通过率筛出的题；清单由
   coder1_plus_selection.py 生成。只作用于训练集，评测集不动。
4. 补充测试（补强项 3）
   --augment_cases 指向 coder1_plus_tests.py 审核后接受的用例，按题追加进 GT 的断言，测试版本随之变化。
   只有参考解与独立 oracle 一致的用例才会被接受，见 coder1_plus_tests.py 与 coder1_plus_oracles.py。

prompt 与原版逐字相同；题目筛选、去重和 extra_info.index 的含义也与原版 leetcode2k() 相同，
所以与原版相比只改了执行契约这一个变量。data_source 写成 ``code_plus``，训练时路由到补强版判分器。

只用标准库即可生成 JSONL；加 --parquet 时另写训练用的 parquet，需要 pandas 与 pyarrow。

用法（在仓库根目录）：
    python examples/data_preprocess/coder1_plus.py --root_dir ./data
    python examples/data_preprocess/coder1_plus.py --root_dir ./data --parquet
    # 再并入原版已处理好的 TACO 行（取原版 parquet 中 dataset 不是 LeetCodeDataset 的行）：
    python examples/data_preprocess/coder1_plus.py --root_dir ./data --parquet --merge_taco_from ./data/code-r1-12k

最后一种用法依赖 pandas，本仓库的 macOS 学习环境没有执行过，需在训练环境验证。
"""

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coder1_plus_tests import extend_test  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "LeetCodeDataset" / "data"
ORIGINAL_PREPROCESS = REPO / "examples" / "data_preprocess" / "coder1.py"

DATA_SOURCE = "code_plus"
GT_CONTRACT = "prelude+solution+functional/v1"  # 与 verl/utils/reward_score/coder1_plus 保持一致
USER_PROMPT_PREFIX = ("Please solve the programming task below using a self-contained code snippet "
                      "in a markdown code block.\n\n")
SCHEMA_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
SHUFFLE_SEED = 666  # 与原版相同的种子；打乱算法不同，只影响文件行序，训练时数据加载器还会再打乱

_CONSTRAINT_HEAD = re.compile(r"constraints", re.IGNORECASE)
_POW10_TOKEN = re.compile(r"(?<![\d.])(10[1-9]|1(0{2,}))(?![\d.])")  # '105'=10^5（上标丢失）、'1000'=10^3


def _as_pow10(token: str):
    m = re.fullmatch(r"10([1-9])", token)
    if m:
        return 10 ** int(m.group(1))
    m = re.fullmatch(r"1(0{2,})", token)
    if m:
        return 10 ** len(m.group(1))
    return None


def constraint_lines(question_title: str):
    """D1 定案「约束该给」的数据侧落地：从完整题面解析 Constraints 并规范化。

    上标丢失按 10 的幂识别（'105'→100000，与 perf.py 同规则）；只收比较行；
    解析不出或题面没有 Constraints 段时返回空列表——该题 prompt 保持原样，
    与「判分器认的才是硬约束」的口径一致（没把握的约束不硬造）。
    """
    if not question_title:
        return []
    hit = _CONSTRAINT_HEAD.search(question_title)
    if not hit:
        return []
    tail = question_title[hit.end():]
    lines = []
    for raw in tail.splitlines():
        line = raw.strip().strip(":").strip()  # 'Constraints:' 的冒号会留在段首
        if not line:
            if lines:
                break  # Constraints 段结束（遇到空行且已收集到内容）
            continue
        if not any(op in line for op in ("<=", ">=", "<", ">")):
            if lines:
                break  # 已收集完，遇到非比较行（例如 Follow up:）即结束
            continue  # 段首的杂行（冒号残留、\xa0 等）跳过，等第一条比较行
        fixed = _POW10_TOKEN.sub(lambda m: str(_as_pow10(m.group(1)) or m.group(1)), line)
        lines.append(fixed.rstrip(".") + ".")
    return lines


def original_system_prompt() -> str:
    """从原版 coder1.py 的语法树里读出 SYSTEM_PROMPT，保证与原版逐字相同。"""
    tree = ast.parse(ORIGINAL_PREPROCESS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"SYSTEM_PROMPT not found in {ORIGINAL_PREPROCESS}")


def check_prompt_template_unchanged() -> None:
    """原版的用户消息模板若被改动，就停下来，避免两边 prompt 悄悄不一致。"""
    source = ORIGINAL_PREPROCESS.read_text(encoding="utf-8")
    expected = USER_PROMPT_PREFIX.replace("\n", "\\n") + "{example['meta']['query'].strip()}"
    if expected not in source:
        raise RuntimeError("The user-prompt template in the original coder1.py changed; update USER_PROMPT_PREFIX.")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def leetcode_records():
    """与原版 leetcode2k() 相同：合并 v2 RL 与 v2 SFT，排除 v2 test 的题号，同一题号只保留首次出现。

    返回 (train, test)，每个元素是 (源文件名, 源文件行号, 原始记录)。
    """
    test_name = "LeetCodeDataset-v2-test-problems.jsonl"
    test = read_jsonl(DATA_DIR / test_name)
    test_ids = {record["meta"]["question_id"] for record in test}
    train, seen = [], set()
    for name in ("LeetCodeDataset-v2-rl-problems.jsonl", "LeetCodeDataset-v2-sft-problems.jsonl"):
        for line_no, record in enumerate(read_jsonl(DATA_DIR / name), 1):
            question_id = record["meta"]["question_id"]
            if question_id in test_ids or question_id in seen:
                continue
            seen.add(question_id)
            train.append((name, line_no, record))
    return train, [(test_name, line_no, record) for line_no, record in enumerate(test, 1)]


ANY_ORDER = re.compile(r"in any order", re.IGNORECASE)
# 题面没写「in any order」、但官方判题接受任意顺序的题，按发现逐个登记，并写明依据。
UNORDERED_OVERRIDES = {
    # 2026-10-14：参考解的输出顺序随哈希种子变化，6 个种子下按列表比较只过 3 次；题目要求返回全部最短路径。
    "word-ladder-ii",
}


def build_ground_truth(record, extra_cases=()) -> str:
    test = extend_test(record["test"], extra_cases) if extra_cases else record["test"]
    functional = f"{test}\n\ncheck({record['entry_point'].strip()})"  # 没有补充用例时与原版相同
    gt = {"prelude": record["prompt"], "functional": functional}
    if ANY_ORDER.search(record["meta"].get("question_title", "")) or record["task_id"] in UNORDERED_OVERRIDES:
        gt["compare"] = "unordered_outer"
    return json.dumps(gt)


def build_row(source_file: str, line_no: int, record, index: int, split: str, system_prompt: str, extra_cases=(),
              perf=None, constraints=None):
    user_prompt = USER_PROMPT_PREFIX + record["meta"]["query"].strip()
    if constraints:  # D1：约束该给（数值消歧 + 复杂度提示）；样式对齐课程 demo 的 ACTOR_PROMPT_APPEND
        user_prompt += "\n\nExplicit task constraints:\n" + "\n".join(f"- {line}" for line in constraints)
    extra_info = {
        # 原版的 5 个字段，含义不变
        "split": split,
        "index": index,
        "reference": record["completion"],
        "prompt": user_prompt,
        "dataset": "LeetCodeDataset",
        # 补强版新增的追溯字段
        "task_id": record["task_id"],
        "question_id": record["meta"]["question_id"],
        "source_file": f"LeetCodeDataset/data/{source_file}",
        "source_line_1_based": line_no,
        "gt_contract": GT_CONTRACT,
        "test_version": f"leetcode-examples+aug{len(extra_cases)}" if extra_cases else "leetcode-examples",
        "added_case_ids": [case["case_id"] for case in extra_cases],
    }
    if perf is not None:  # 补强项 10 的性能轮标定（time_limit_ms + 确定性大输入规格）
        extra_info["perf"] = perf
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "ability": "coding",
        "reward_model": {"style": "rule", "ground_truth": build_ground_truth(record, extra_cases)},
        "extra_info": extra_info,
    }


def build_leetcode_rows(augment=None, perf_index=None, append_constraints=False):
    """augment：{task_id: [接受的用例]}，只会追加 status 为 accepted 的用例。
    perf_index：{task_id: perf}，由 coder1_plus_perf.py 产出的 perf-index.jsonl 转换而来；
    没有标定的题不写 perf 字段，判分时性能轮自动跳过。
    append_constraints：D1 定案「约束该给」——解析 Constraints 段并规范化后追加进 user prompt；
    解析不出的题保持原样。默认关闭以保持与原版逐字相同的 prompt（单变量对照）。"""
    check_prompt_template_unchanged()
    system_prompt = original_system_prompt()
    augment = {task: [case for case in cases if case.get("status") == "accepted"]
               for task, cases in (augment or {}).items()}
    perf_index = perf_index or {}

    def constraints_of(record):
        return constraint_lines(record["meta"].get("question_title", "")) if append_constraints else None

    train, test = leetcode_records()
    train_rows = [build_row(name, line, record, index, "train", system_prompt, augment.get(record["task_id"], ()),
                            perf_index.get(record["task_id"]), constraints_of(record))
                  for index, (name, line, record) in enumerate(train)]
    test_rows = [build_row(name, line, record, index, "test", system_prompt, augment.get(record["task_id"], ()),
                           perf_index.get(record["task_id"]), constraints_of(record))
                 for index, (name, line, record) in enumerate(test)]
    return train_rows, test_rows


def _to_python(value):
    """把 pandas 读回来的 numpy 数组等对象还原成普通的 list 和 dict，便于与新行合并。"""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def taco_rows_from_original(parquet_dir: Path, split: str):
    """从原版输出的 parquet 里取出非 LeetCode 的行，也就是 TACO 的行，原样复用。需要 pandas。"""
    import pandas as pd  # noqa: deferred import, only needed for this option

    frame = pd.read_parquet(parquet_dir / f"{split}.parquet")
    rows = []
    for record in frame[SCHEMA_COLUMNS].to_dict("records"):
        record = _to_python(record)
        if (record.get("extra_info") or {}).get("dataset") != "LeetCodeDataset":
            rows.append(record)
    return rows


def write_jsonl(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(rows, path: Path) -> None:
    import pandas as pd  # noqa: deferred import, needs pandas and pyarrow

    pd.DataFrame(rows, columns=SCHEMA_COLUMNS).to_parquet(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--parquet", action="store_true", help="also write train/test parquet (needs pandas and pyarrow)")
    parser.add_argument("--augment_cases", default=None,
                        help="JSON file {task_id: [case records]} produced by coder1_plus_tests; accepted cases are appended")
    parser.add_argument("--drop_task_ids", default=None, help="JSON list of task_ids to remove from the train split")
    parser.add_argument("--keep_task_ids", default=None, help="JSON list; if given, keep only these task_ids in train")
    parser.add_argument("--merge_taco_from", default=None,
                        help="directory holding the original coder1.py train/test parquet; its TACO rows are merged as-is")
    parser.add_argument("--perf_index", default=None,
                        help="perf-index.jsonl produced by coder1_plus_perf.py; merges each task's perf into extra_info")
    parser.add_argument("--append_constraints", action="store_true",
                        help="parse the Constraints section of question_title, normalize 10^k tokens, "
                             "and append to the user prompt (D1; default off keeps the prompt verbatim "
                             "with the original)")
    args = parser.parse_args(argv)

    augment = json.loads(Path(args.augment_cases).read_text(encoding="utf-8")) if args.augment_cases else None
    perf_index = None
    if args.perf_index:
        perf_index = {}
        with open(args.perf_index, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("perf"):
                        perf_index[entry["task_id"]] = entry["perf"]
    train_rows, test_rows = build_leetcode_rows(augment, perf_index, args.append_constraints)
    if args.drop_task_ids:
        drop = set(json.loads(Path(args.drop_task_ids).read_text(encoding="utf-8")))
        train_rows = [row for row in train_rows if row["extra_info"]["task_id"] not in drop]
    if args.keep_task_ids:
        keep = set(json.loads(Path(args.keep_task_ids).read_text(encoding="utf-8")))
        train_rows = [row for row in train_rows if row["extra_info"]["task_id"] in keep]
    names = ["leetcode2k"]
    if args.merge_taco_from:
        source_dir = Path(args.merge_taco_from)
        train_rows += taco_rows_from_original(source_dir, "train")
        test_rows += taco_rows_from_original(source_dir, "test")
        names.append("taco")
    random.Random(SHUFFLE_SEED).shuffle(train_rows)

    out_dir = Path(args.root_dir) / f"code-r1-plus-{round(len(train_rows) / 1000)}k-{'-'.join(names)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, out_dir / "train.jsonl")
    write_jsonl(test_rows, out_dir / "test.jsonl")
    if args.parquet:
        write_parquet(train_rows, out_dir / "train.parquet")
        write_parquet(test_rows, out_dir / "test.parquet")

    print(f"train rows: {len(train_rows)}  test rows: {len(test_rows)}  -> {out_dir}")
    if perf_index:
        print(f"perf merged for {len(perf_index)} tasks (extra_info.perf)")
    if args.parquet:
        print(f"Train with: data.train_files={out_dir}/train.parquet data.val_files={out_dir}/test.parquet")
        print("Rows with data_source=code_plus are scored by verl/utils/reward_score/coder1_plus.")
    return out_dir


if __name__ == "__main__":
    main()
