"""补强项 10 的数据侧：批量离线标定每题的性能轮参数（time_limit_ms 与大输入规格）。

用法（仓库根目录，标准库即可）：

    python -B examples/data_preprocess/coder1_plus_perf.py --task maximum-subarray        # 单题
    python -B examples/data_preprocess/coder1_plus_perf.py --limit 50                     # 前 50 题
    python -B examples/data_preprocess/coder1_plus_perf.py --all                          # 全量 1890 题
    python -B examples/data_preprocess/coder1_plus_perf.py --task maximum-subarray --emit-merge-row

输出 ``data/perf/<task_id>.json``（单题）或 ``data/perf/perf-index.jsonl``（批量），每条：

    {"task_id": ..., "perf": {"spec": {...}, "entry": ..., "time_limit_ms": ..., "expected": ...},
     "calibrated": {...}, "skip_reason": null}      # 跳过的题 skip_reason 非空、perf 为 null

``--emit-merge-row`` 额外打印把 perf 并进 coder1_plus 数据行 ``extra_info`` 的合并说明。

设计要点（详见 verl/utils/reward_score/coder1_plus/perf.py 的模块注释）：
- 规模上界解析自 question_title 的 Constraints（上标丢失 '105'→10^5），封顶 100_000、下限 1_000；
- 只支持「恰有一个 List[int] 参数」的题，其余记 skip_reason 跳过（判分行为退回第 9 项，不降级）；
- timeout = max(500ms, 3 × 参考解实测墙钟[取两次较慢者]；含进程启动开销，与沙箱计时同口径)；
- 标定跑的是数据集自带参考解（可信代码），本地 subprocess 直跑，不经沙箱。
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_perf_path = REPO / "verl" / "utils" / "reward_score" / "coder1_plus" / "perf.py"
_spec = importlib.util.spec_from_file_location("coder1_plus_perf_core", _perf_path)
perf_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perf_core)

DATASET = REPO / "LeetCodeDataset" / "data"
OUT_DIR = REPO / "data" / "perf"


def iter_leetcode_records():
    """与 coder1.leetcode2k 同口径：合并 v2 RL + v2 SFT，排除 v2 test 的 question_id，同 ID 留首次。"""
    with open(DATASET / "LeetCodeDataset-v2-test-problems.jsonl", encoding="utf-8") as handle:
        test_ids = {json.loads(line)["meta"]["question_id"] for line in handle if line.strip()}
    seen, records = set(), []
    for name in ("LeetCodeDataset-v2-rl-problems.jsonl", "LeetCodeDataset-v2-sft-problems.jsonl"):
        with open(DATASET / name, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = record["meta"]["question_id"]
                if qid in test_ids or qid in seen:
                    continue
                seen.add(qid)
                records.append(record)
    return records


def calibrate(records, task_filter=None, limit=None):
    results, counters = [], {"ok": 0}
    for record in records:
        task_id = record["task_id"]
        if task_filter and task_id != task_filter:
            continue
        if limit is not None and counters["ok"] >= limit:
            break
        started = time.monotonic()
        perf, reason = perf_core.calibrate_one(record)
        if perf is None:
            counters[reason] = counters.get(reason, 0) + 1
            results.append({"task_id": task_id, "perf": None, "skip_reason": reason})
            continue
        counters["ok"] = counters.get("ok", 0) + 1
        entry = {"task_id": task_id, "perf": perf, "skip_reason": None}
        results.append(entry)
        print(f"[ok] {task_id}: size={perf['spec']['size']} "
              f"time_limit_ms={perf['time_limit_ms']} "
              f"ref_wall={perf['calibrated']['reference_wall_s']}s "
              f"({time.monotonic() - started:.1f}s 校准)")
    return results, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", help="只标定指定的 task_id（如 maximum-subarray）")
    group.add_argument("--limit", type=int, help="标定前 N 道可标定的题")
    group.add_argument("--all", action="store_true", help="全量标定")
    parser.add_argument("--emit-merge-row", action="store_true",
                        help="打印把 perf 并进数据行 extra_info 的说明")
    args = parser.parse_args()

    records = iter_leetcode_records()
    print(f"leetcode2k 口径共 {len(records)} 题")
    if not (args.task or args.limit or args.all):
        parser.print_help()
        return 0

    results, counters = calibrate(records, task_filter=args.task,
                                  limit=None if (args.task or args.all) else args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "perf-index.jsonl", "w", encoding="utf-8") as handle:
        for entry in results:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n标定 {counters.get('ok', 0)} 题，跳过 {len(results) - counters.get('ok', 0)} 题：")
    for reason, count in sorted(counters.items(), key=lambda kv: -kv[1]):
        if reason != "ok":
            print(f"  {reason}: {count}")
    print(f"产物：{OUT_DIR / 'perf-index.jsonl'}")

    if args.emit_merge_row and results and results[0]["perf"]:
        perf = results[0]["perf"]
        print("\n合并到 coder1_plus 数据行（extra_info.perf）：")
        print(json.dumps({"extra_info": {"perf": perf}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
