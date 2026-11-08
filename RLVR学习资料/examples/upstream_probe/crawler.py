# -*- coding: utf-8 -*-
"""LeetCode 题目爬虫（学习/研究用途）v2：列表分页 → 逐题详情 → 六步解析 → 验收。

v2 修复实测暴露的三个问题：
1. entry_point 误取注释里的 `# def __init__` → 过滤注释行，只取 class Solution 内的 def；
2. 多参数输入靠"碰巧"语法正确 → 按顶层逗号分割参数对；
3. 链表/树题目返回值需包装与比较（`==` 会误判）→ 检测 ListNode/TreeNode，参数用
   list_node/tree_node 包装、比较用 is_same_list/is_same_tree，并在 prelude 补齐定义。

用法（仓库根目录，需代理）：
    PYTHONUTF8=1 HTTPS_PROXY=http://127.0.0.1:7890 python -B \
        "RLVR学习资料/examples/upstream_probe/crawler.py" --limit 10 --out tmp_crawl
"""
import argparse
import html as html_mod
import json
import re
import time
import urllib.request
from pathlib import Path

GRAPHQL = "https://leetcode.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com/problemset/all/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) crawler-for-study/1.0",
}
LIST_QUERY = """query ps($cat:String!,$limit:Int!,$skip:Int!){
  problemsetQuestionListV2(categorySlug:$cat, limit:$limit, skip:$skip){
    questions{ questionFrontendId titleSlug difficulty paidOnly }
  }
}"""
DETAIL_QUERY = """query q($titleSlug:String!){
  question(titleSlug:$titleSlug){
    questionId titleSlug difficulty isPaidOnly content codeSnippets{lang code}
  }
}"""

# 与 LeetCodeDataset 原版 prompt 对齐的标准前缀（链表/树的包装与比较）
PRELUDE_BASE = """from typing import *

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values):
    if not values:
        return None
    head = ListNode(values[0]); p = head
    for val in values[1:]:
        p.next = ListNode(val); p = p.next
    return head

def is_same_list(a, b):
    while a is not None and b is not None:
        if a.val != b.val:
            return False
        a, b = a.next, b.next
    return a is None and b is None

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def tree_node(values):
    if not values:
        return None
    root = TreeNode(values[0]); queue = [root]; i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i]); queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i]); queue.append(node.right)
        i += 1
    return root

def is_same_tree(a, b):
    if a is None or b is None:
        return a is b
    return a.val == b.val and is_same_tree(a.left, b.left) and is_same_tree(a.right, b.right)
"""


def gql(query, variables, retries=3):
    payload = json.dumps({"query": query, "variables": variables}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(GRAPHQL, data=payload, headers=HEADERS, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            if body.get("errors"):
                raise RuntimeError(body["errors"][0].get("message", "graphql error"))
            return body["data"]
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_list(limit, delay):
    rows, skip, page = [], 0, 50
    while len(rows) < limit:
        data = gql(LIST_QUERY, {"cat": "all-code-essentials", "limit": page, "skip": skip})
        batch = data["problemsetQuestionListV2"]["questions"]
        if not batch:
            break
        rows.extend(batch); skip += page; time.sleep(delay)
    return rows[:limit]


def html_to_text(content):
    text = html_mod.unescape(re.sub(r"<[^>]+>", "", content))
    return re.sub(r"\n\s*\n", "\n", text).replace("\xa0", " ").strip()


def parse_examples(text):
    return [{"input": i.strip(), "output": o.strip()}
            for i, o in re.findall(r"Input:\s*(.+?)\s*\nOutput:\s*(.+?)\s*(?:\n|$)", text)]


def find_entry(starter):
    """过滤注释行后，只取 class Solution 内的第一个 def（修复注释块误匹配）。"""
    src = "\n".join(line for line in (starter or "").splitlines() if not line.strip().startswith("#"))
    m = re.search(r"class Solution\b.*?def (\w+)\(", src, re.S)
    return f"Solution().{m.group(1)}" if m else None


def split_params(text):
    """按顶层逗号切分 'a = 1, b = [2,3]'（忽略括号/引号内的逗号）。"""
    parts, depth, quote, cur = [], 0, None, ""
    for ch in text:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(cur); cur = ""; continue
        cur += ch
    if cur.strip():
        parts.append(cur)
    out = []
    for part in parts:
        name, _, value = part.partition("=")
        out.append((name.strip(), value.strip()))
    return out


def examples_to_test(examples, kind):
    """kind: value | list | tree。链表/树用 list_node/tree_node 包装与 is_same_* 比较。"""
    asserts = []
    for e in examples:
        params = split_params(e["input"])
        if not params or any(not name or not val for name, val in params):
            return None, "unparsable_input"
        wrap = {"value": (lambda v: v), "list": (lambda v: f"list_node({v})"),
                "tree": (lambda v: f"tree_node({v})")}[kind]
        call = ", ".join(f"{name} = {wrap(val)}" for name, val in params)
        if kind == "list":
            asserts.append(f"assert is_same_list(candidate({call}), list_node({e['output']}))")
        elif kind == "tree":
            asserts.append(f"assert is_same_tree(candidate({call}), tree_node({e['output']}))")
        else:
            asserts.append(f"assert candidate({call}) == {e['output']}")
    return ("def check(candidate):\n    " + "\n    ".join(asserts) if asserts else None), None


def parse_one(slug, data, repo_index):
    q = data["question"]
    content = q.get("content") or ""
    text = html_to_text(content)
    examples = parse_examples(text)
    starter = next((s["code"] for s in (q.get("codeSnippets") or [])
                    if "python3" in (s.get("lang") or "").lower()), None)
    entry = find_entry(starter)
    kind = "list" if "ListNode" in (starter or "") else "tree" if "TreeNode" in (starter or "") else "value"
    test, test_error = examples_to_test(examples, kind)
    checks = {
        "struct": bool(q.get("questionId") and slug and content),
        "examples_expected": len(re.findall(r"Input:", text)), "examples_parsed": len(examples),
        "examples_match": len(re.findall(r"Input:", text)) == len(examples),
        "entry": bool(entry), "kind": kind, "test_ok": bool(test), "test_error": test_error,
        "behavior": "skip: 无参考解（官网题解在付费墙后）", "repo_same_task": repo_index.get(slug),
    }
    prelude = PRELUDE_BASE if kind in ("list", "tree") else "from typing import *"
    row = {
        "task_id": q.get("titleSlug"), "prompt": starter or "", "entry_point": entry,
        "test": test, "completion": None, "examples": examples,
        "meta": {
            "question_id": int(q["questionId"]) if q.get("questionId") else None,
            "difficulty": (q.get("difficulty") or "").lower() or None, "lang_code": starter,
            "question_title": text,
            "query": (text.split("\n")[0].strip() + "\n \nPlease complete the following python code precisely:\n```python\n"
                      + (starter or "").strip() + "\n```"),
            "response": None, "split": None,
        },
        "_prelude": prelude,
    }
    return row, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="tmp_crawl")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out); (out / "raw").mkdir(parents=True, exist_ok=True)

    repo_index = {}
    try:
        with open("LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                repo_index[r["task_id"]] = {"question_id": r["meta"]["question_id"],
                                            "difficulty": r["meta"]["difficulty"]}
    except FileNotFoundError:
        pass

    listing = fetch_list(args.limit, args.delay)
    print(f"列表取得 {len(listing)} 题（免费: {sum(1 for x in listing if not x.get('paidOnly'))}）")
    rows, reviews = [], []
    for item in listing:
        slug = item["titleSlug"]
        if item.get("paidOnly"):
            reviews.append({"slug": slug, "status": "skip_paid"}); continue
        raw_path = out / "raw" / f"{slug}.json"
        cached = raw_path.exists()
        try:
            data = json.loads(raw_path.read_text(encoding="utf-8")) if cached else gql(DETAIL_QUERY, {"titleSlug": slug})
            if not cached:
                raw_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                time.sleep(args.delay)
        except Exception as error:  # noqa: BLE001
            reviews.append({"slug": slug, "status": f"error:{error}"}); continue
        row, checks = parse_one(slug, data, repo_index)
        rows.append(row)
        reviews.append({"slug": slug, "status": "ok" if cached else "fetched", **checks})
        print(f"  [{'cached' if cached else 'fetch'}] {slug:40s} entry={row['entry_point']} "
              f"kind={checks['kind']} ex={checks['examples_parsed']} test={'ok' if checks['test_ok'] else checks['test_error']}")

    with open(out / "parsed.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = [r for r in reviews if r["status"] in ("ok", "fetched")]
    lines = [f"# 爬虫验收报告 v2（limit={args.limit}，{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"- 成功解析：{len(ok)} / {len(reviews)}",
             f"- 结构断言：{sum(1 for r in ok if r.get('struct'))}/{len(ok)}",
             f"- Example 对账：{sum(1 for r in ok if r.get('examples_match'))}/{len(ok)}",
             f"- entry_point：{sum(1 for r in ok if r.get('entry'))}/{len(ok)}",
             f"- 可生成 test：{sum(1 for r in ok if r.get('test_ok'))}/{len(ok)}",
             f"- 题型分布：value={sum(1 for r in ok if r.get('kind') == 'value')} "
             f"list={sum(1 for r in ok if r.get('kind') == 'list')} tree={sum(1 for r in ok if r.get('kind') == 'tree')}", "",
             "| slug | 状态 | kind | examples | 对账 | test | 仓库同题 |", "|---|---|---|---:|---|---|---|"]
    for r in reviews:
        lines.append(f"| {r['slug']} | {r['status']} | {r.get('kind', '-')} | {r.get('examples_parsed', '-')} | "
                     f"{'✓' if r.get('examples_match') else '✗'} | {'✓' if r.get('test_ok') else r.get('test_error', '-')} | "
                     f"{r.get('repo_same_task') or '—'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n产物：{out}/raw/*.json、{out}/parsed.jsonl、{out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
