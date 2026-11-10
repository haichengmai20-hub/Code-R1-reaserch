# -*- coding: utf-8 -*-
"""社区题解获取实测：从 doocs/leetcode 拿参考解，并做行为层验收。

补爬虫 v2 的缺口：LeetCode 官网题解在付费墙后，crawler.py 的行为层验收只能
写 skip。本探针换「社区题解开源库」这条获取途径（01 上游节五条来源之二）：

  https://github.com/doocs/leetcode
  raw: solution/<NN00>-<NN99>/<qid>.<Title>/README.md
  README 的 <!-- tabs:start --> 段里，每个「### 方法…」下有 #### Python3 围栏块。

验收（三层里的行为层，最强）：抓回的每个 Python 解按原生拼法
（前缀 + 参考解 + test + check(entry)）在补强版沙箱（macOS sandbox-exec，
禁网、限写、禁读仓库）里跑；全过 = 题面 ↔ 样例 ↔ 社区参考解 三角自洽。
仓库冻结的 completion 作为对照组用同一拼法跑。

用法（仓库根目录执行；GitHub raw 本机直连即可，无需代理）：
    PYTHONUTF8=1 $PY -B "RLVR学习资料/examples/upstream_probe/reference_probe.py" \
        --out RLVR学习资料/examples/upstream_probe/doocs_snapshot
"""
import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "RLVR学习资料" / "examples" / "source-supplements"))
from supplement_env import load_reward_score  # noqa: E402

load_reward_score()  # 无 torch/numpy 环境加载真实判分包（verl 包桩见 supplement_env）
from verl.utils.reward_score.coder1_plus.sandbox import run_python  # noqa: E402

RAW_BASE = "https://raw.githubusercontent.com/doocs/leetcode/main"
HEADERS = {"User-Agent": "Mozilla/5.0 crawler-for-study/1.0 (reference probe)"}

# 默认四题覆盖 value / list / tree 三种题型：链表和树走前缀辅助函数路径
# （题须在本地 v2 数据里：merge-two-sorted-lists 不在 v2-rl/sft，用 add-two-numbers 替代）
DEFAULT_PROBLEMS = ["maximum-subarray", "two-sum", "add-two-numbers", "same-tree"]

# slug → 官方英文标题的启发式（连字符分段、首字母大写）覆盖不到的题写在这里
SPECIAL_TITLES = {
    "3sum": "3Sum", "powx-n": "Pow(x, n)", "n-queens": "N-Queens", "n-queens-ii": "N-Queens II",
    "sqrtx": "Sqrt(x)", "atoi": "String to Integer (atoi)",
}


def slug_to_title(slug):
    if slug in SPECIAL_TITLES:
        return SPECIAL_TITLES[slug]
    return " ".join(w[:1].upper() + w[1:] for w in slug.split("-"))


def doocs_dir(qid, title):
    lo = qid // 100 * 100
    return f"solution/{lo:04d}-{lo + 99:04d}/{qid:04d}.{title}"


def load_repo_index():
    """task_id → 原始记录（v2 RL 优先，补 v2 SFT；都要本地子模块已初始化）。"""
    index = {}
    for name in ["LeetCodeDataset-v2-rl-problems.jsonl", "LeetCodeDataset-v2-sft-problems.jsonl"]:
        path = REPO_ROOT / "LeetCodeDataset" / "data" / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                index.setdefault(rec["task_id"], rec)
    return index


def fetch_readme(relative_dir, cache_dir):
    """抓 doocs README.md，命中缓存零请求（断点重跑）。"""
    raw = cache_dir / f"{relative_dir.split('/')[-1]}.md"
    if raw.exists():
        return raw.read_text(encoding="utf-8"), "cached"
    url = f"{RAW_BASE}/{urllib.parse.quote(relative_dir)}/README.md"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 白名单域名
        text = resp.read().decode("utf-8")
    raw.write_text(text, encoding="utf-8")
    time.sleep(1.0)  # 限速
    return text, "fetched"


def extract_python_solutions(readme):
    """取每个「### 方法」标题下 #### Python3 的围栏块（须含 class Solution）。"""
    solutions, method, lang = [], None, None
    in_fence = False
    block, fence_len = [], 0
    for line in readme.split("\n"):
        fence = re.match(r"^ {0,3}(`{3,})[ \t]*(\S*)[ \t]*$", line)
        if in_fence:
            if fence and not fence.group(2) and len(fence.group(1)) >= fence_len:  # 闭合围栏无 info
                if lang == "Python3" and "class Solution" in "\n".join(block):
                    solutions.append({"method": method, "code": "\n".join(block).strip()})
                in_fence = False
            else:
                block.append(line)
            continue
        if fence:  # 开启围栏（info 串是语言名）
            in_fence, block, fence_len = True, [], len(fence.group(1))
        elif line.startswith("### "):
            method = line[4:].strip()
        elif line.startswith("#### "):
            lang = line[5:].strip()
    return solutions


def assemble_native(rec, solution_code):
    """原生评测器拼法（leetcote/execution.py:28-33）：前缀 + 解 + test + check(入口)。"""
    return "\n\n".join([rec["prompt"], solution_code, rec["test"], f"check({rec['entry_point']})"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*", default=DEFAULT_PROBLEMS)
    ap.add_argument("--out", default="RLVR学习资料/examples/upstream_probe/doocs_snapshot")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    repo_index = load_repo_index()
    results, report = [], []
    for slug in args.problems:
        rec = repo_index.get(slug)
        if rec is None:
            print(f"✗ {slug:26s} skip: 本地 v2 数据无此题（对照组与 test 都无从取）")
            report.append({"slug": slug, "status": "skip: 本地数据无此题"}); continue
        qid = int(rec["meta"]["question_id"])
        rel = doocs_dir(qid, slug_to_title(slug))
        try:
            readme, how = fetch_readme(rel, out / "raw")
        except Exception as error:  # noqa: BLE001
            report.append({"slug": slug, "doocs_dir": rel, "status": f"error: {error}"})
            continue

        solutions = extract_python_solutions(readme)
        # 对照组：仓库冻结的 completion 用同一拼法、同一沙箱跑
        control = run_python(assemble_native(rec, rec["completion"]))
        entry = {"slug": slug, "qid": qid, "doocs_dir": rel, "fetch": how,
                 "repo_completion": {"code_head": rec["completion"].splitlines()[0],
                                     "status": control.status, "duration": round(control.duration, 3)},
                 "methods": []}
        for sol in solutions:
            run = run_python(assemble_native(rec, sol["code"]))
            sim = difflib.SequenceMatcher(None, sol["code"], rec["completion"]).ratio()
            entry["methods"].append({
                "method": sol["method"], "status": run.status, "duration": round(run.duration, 3),
                "similarity_to_repo_completion": round(sim, 3),
                "stderr_tail": run.stderr.strip().splitlines()[-1][:120] if not run.ok else "",
                "code": sol["code"]})
        results.append(entry)
        mark = "✓" if all(m["status"] == "ok" for m in entry["methods"]) and control.ok else "✗"
        print(f"{mark} {slug:26s} {rel}")
        print(f"    对照组(仓库 completion): {control.status}    社区解: "
              + ", ".join(f"{m['method']}={m['status']}(sim {m['similarity_to_repo_completion']})"
                          for m in entry["methods"]))
        report.append({"slug": slug, "doocs_dir": rel, "status": "ok",
                       "methods": [(m["method"], m["status"]) for m in entry["methods"]],
                       "control": control.status})

    (out / "reference_solutions.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# 社区参考解获取报告（doocs/leetcode，{time.strftime('%Y-%m-%d %H:%M')}）", "",
             "| 题目 | doocs 目录 | 获取 | 对照组 | 社区解（方法=行为层验收） |", "|---|---|---|---|---|"]
    for entry, row in zip(results, [r for r in report if r.get("status") == "ok"]):
        lines.append(f"| {entry['slug']} | `{entry['doocs_dir']}` | {entry['fetch']} | "
                     f"{entry['repo_completion']['status']} | "
                     + "；".join(f"{m['method']} → **{m['status']}**（相似度 {m['similarity_to_repo_completion']}）"
                                 for m in entry["methods"]) + " |")
    for row in report:
        if row.get("status", "ok") != "ok":
            lines.append(f"| {row['slug']} | {row.get('doocs_dir', '-')} | - | - | {row['status']} |")
    ok_problems = sum(1 for e in results if e["repo_completion"]["status"] == "ok"
                      and all(m["status"] == "ok" for m in e["methods"]))
    lines += ["", f"**结果：{ok_problems}/{len(results)} 题的社区参考解全部通过行为层验收"
                  "（原生拼法跑仓库 test，沙箱隔离执行）；与仓库冻结 completion 的文本相似度见上表——"
                  "低相似度 + 双双通过 = 独立来源的行为一致，正是交叉核对想要的证据。**"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n产物：{out}/raw/*.md（缓存的题解原文）、reference_solutions.json、report.md")
    return 0 if ok_problems == len(results) and len(results) == len(args.problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
