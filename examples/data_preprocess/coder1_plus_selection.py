"""补强项 7：评测集去重与按难度筛选。

原版 leetcode2k() 只按题号排除 v2 test，不检查与外部评测集的重叠，也不按当前模型的通过率筛题。
学习资料 01 全景第六、七步与主框架 A2、A8 要求：
- 去污染：训练题不能与评测题重复，既查相同题号，也查改了题号或措辞的同一道题；
- 难度筛选：GRPO 只有在同题回答有好有坏时才有组内信号，全对或全错的题不提供 outcome 信号，
  可按某个固定 checkpoint 的实测通过率筛掉；这个通过率必须来自真实采样，换 checkpoint 要重新测。

用法（在仓库根目录）：
    # 评测集文件为 JSONL，每行至少有 id 和 text，可选 question_id、task_id
    python examples/data_preprocess/coder1_plus_selection.py decontaminate --train data/.../train.jsonl \\
        --eval eval.jsonl --out drop_task_ids.json
    # 通过率文件为 JSON：{task_id: {"n": 采样数, "c": 通过数}}，来自固定 checkpoint 的真实采样
    python examples/data_preprocess/coder1_plus_selection.py difficulty --stats pass_stats.json --out keep_task_ids.json
得到的清单交给 coder1_plus.py 的 --drop_task_ids 或 --keep_task_ids。
"""

import argparse
import json
import re
from pathlib import Path

PROMPT_PREFIX = ("Please solve the programming task below using a self-contained code snippet "
                 "in a markdown code block.")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str):
    return _TOKEN.findall(text.lower())


def ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_BOILERPLATE = ("Please complete the following python code precisely:", PROMPT_PREFIX)


def description_only(text: str) -> str:
    """只留自然语言描述：去掉代码块（签名和 TreeNode、ListNode 等模板注释）与所有题共用的固定句子。"""
    text = _CODE_BLOCK.sub(" ", text)
    for sentence in _BOILERPLATE:
        text = text.replace(sentence, " ")
    return text


def statement_of(row) -> str:
    """训练行里模型看到的题面描述。"""
    return description_only(row["extra_info"]["prompt"])


def find_contamination(train_rows, eval_items, n=8, threshold=0.5):
    """返回疑似重叠的 (训练题, 评测题) 对。

    两种依据：题号或 task_id 相同；或训练题面的 n 元词组里，至少 threshold 比例出现在同一道评测题里。
    后者能找出改了题号、但措辞基本相同的题。阈值越低越保守，需要人工复核疑似对。
    """
    index = {}
    eval_grams = {}
    for item in eval_items:
        grams = ngrams(tokens(description_only(item["text"])), n)
        eval_grams[item["id"]] = grams
        for gram in grams:
            index.setdefault(gram, set()).add(item["id"])
    eval_by_qid = {item.get("question_id"): item["id"] for item in eval_items if item.get("question_id") is not None}
    eval_by_task = {item.get("task_id"): item["id"] for item in eval_items if item.get("task_id")}
    flagged = []
    for row in train_rows:
        info = row["extra_info"]
        hits = {}
        if info.get("question_id") in eval_by_qid:
            hits[eval_by_qid[info["question_id"]]] = {"reason": "same_question_id", "containment": None}
        if info.get("task_id") in eval_by_task:
            hits.setdefault(eval_by_task[info["task_id"]], {"reason": "same_task_id", "containment": None})
        grams = ngrams(tokens(statement_of(row)), n)
        if grams:
            counts = {}
            for gram in grams:
                for eval_id in index.get(gram, ()):
                    counts[eval_id] = counts.get(eval_id, 0) + 1
            for eval_id, count in counts.items():
                containment = count / len(grams)
                if containment >= threshold:
                    entry = hits.setdefault(eval_id, {"reason": "ngram_overlap", "containment": None})
                    entry["containment"] = round(containment, 3)
        for eval_id, entry in hits.items():
            flagged.append({"task_id": info.get("task_id"), "eval_id": eval_id, **entry})
    return flagged


def select_by_pass_rate(stats, low=0.0, high=1.0, min_samples=4):
    """按固定 checkpoint 的实测通过率筛题：保留 low < c/n < high 的题。返回 (保留列表, 剔除原因)。"""
    keep, dropped = [], {}
    for task_id, item in sorted(stats.items()):
        n, c = int(item["n"]), int(item["c"])
        if n < min_samples:
            dropped[task_id] = f"only {n} samples"
        elif not low < c / n < high:
            dropped[task_id] = f"pass rate {c}/{n} outside ({low}, {high})"
        else:
            keep.append(task_id)
    return keep, dropped


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dec = sub.add_parser("decontaminate")
    dec.add_argument("--train", required=True)
    dec.add_argument("--eval", required=True)
    dec.add_argument("--n", type=int, default=8)
    dec.add_argument("--threshold", type=float, default=0.5)
    dec.add_argument("--out", required=True)
    dif = sub.add_parser("difficulty")
    dif.add_argument("--stats", required=True)
    dif.add_argument("--low", type=float, default=0.0)
    dif.add_argument("--high", type=float, default=1.0)
    dif.add_argument("--min_samples", type=int, default=4)
    dif.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "decontaminate":
        flagged = find_contamination(_read_jsonl(args.train), _read_jsonl(args.eval), args.n, args.threshold)
        Path(args.out).write_text(json.dumps(sorted({f["task_id"] for f in flagged}), indent=2), encoding="utf-8")
        print(f"flagged {len(flagged)} pairs; {len({f['task_id'] for f in flagged})} training tasks to drop")
    else:
        keep, dropped = select_by_pass_rate(json.loads(Path(args.stats).read_text(encoding="utf-8")), args.low,
                                            args.high, args.min_samples)
        Path(args.out).write_text(json.dumps(keep, indent=2), encoding="utf-8")
        print(f"keep {len(keep)} tasks; drop {len(dropped)}")


if __name__ == "__main__":
    main()
