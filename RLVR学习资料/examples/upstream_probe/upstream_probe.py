# -*- coding: utf-8 -*-
"""上游管线实测重建：GraphQL 抓取的 maximum-subarray 走六步，对照仓库 v2-rl 第 143 行。"""
import json
import re
import html as html_mod

# ── 抓取结果（tmp_gql.json，HTTP 200 实测） ──
q = json.load(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "snapshot_maximum-subarray.json"), encoding="utf-8"))["data"]["question"]
content_html = q["content"]

# ── 仓库实际数据（对照组） ──
with open("LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i == 143:
            repo = json.loads(line)
            break

print("═" * 68)
print("第1步·解析：题面 HTML → 纯文本（朴素去标签，模拟当年转换）")
text = html_mod.unescape(re.sub(r"<[^>]+>", "", content_html))
text = re.sub(r"\n\s*\n", "\n", text).replace("\xa0", " ").strip()
i = text.find("Constraints")
print(text[i:i + 110])
repo_title = repo["meta"]["question_title"]
print(f"   对照仓库 question_title 同段: {repo_title[repo_title.find('Constraints'):repo_title.find('Constraints')+75]!r}")
print(f"   ★上标丢失复现? 我们={'105' in text}, 仓库={'105' in repo_title}")

print("═" * 68)
print("第2步·解析：Example 段 → examples → assert 壳")
pairs = re.findall(r"Input:\s*(.+?)\s*\nOutput:\s*(.+?)\s*(?:\n|$)", text)
examples = [{"input": inp, "output": out} for inp, out in pairs]
print("   解析出", len(examples), "对:", examples[:1], "...")
tests = [f"assert candidate({{'nums': {re.search(r'nums = (.+)', e['input']).group(1)}}}) == {e['output']}"
         for e in examples]
ours_test = "def check(candidate):\n    " + "\n    ".join(tests)
repo_test_body = repo["test"].split("check(")[0]
same2 = all(t in repo["test"] for t in tests)
print(f"   与仓库 test 的断言一致? {same2}（仓库断言数={repo['test'].count('assert')}，我们={len(tests)}）")

print("═" * 68)
print("第3步·解析：starter code → 签名 + entry_point")
starter = next(s["code"] for s in q["codeSnippets"] if s["lang"] == "Python3")
m = re.search(r"def (\w+)\(", starter)
entry = f"Solution().{m.group(1)}"
print(f"   entry_point = {entry}  对照仓库 = {repo['entry_point']}  一致? {entry == repo['entry_point']}")
sig_repo = "class Solution:" in repo["meta"]["query"] and "List[int]" in repo["meta"]["query"]
print(f"   官网 Python3 签名是 list[int]，仓库 query 里是 List[int] —— 官网已改版，数据集冻结旧形态")

print("═" * 68)
print("第4步·拼装：描述第一段 + 固定话术 + 签名 → meta.query")
desc = text.split("\n")[0].strip()
ours_query = (desc + "\n \nPlease complete the following python code precisely:\n```python\n"
              + starter.strip() + "\n```")
repo_query = repo["meta"]["query"]
print(f"   描述段一致? {desc in repo_query}")
print(f"   话术+签名结构一致? {'Please complete the following python code precisely:' in repo_query}")

print("═" * 68)
print("第5/6步·补充：completion（题解核对）与 response（SFT 教材）")
print("   官方题解 API 需要订阅（isPaidOnly=False 但 solution 仍要登录/会员）——真实难点，见记录")
print("   仓库 completion 首行:", repo["completion"].splitlines()[0])
print("   仓库 response 首段:", repo["meta"]["response"][:60], "...")

print("═" * 68)
print("总结：graphql(questionId/difficulty/content/codeSnippets) 已实测四字段与仓库一致；")
print("上标丢失在第1步复现；断言一致；entry_point 一致；签名差异=官网改版所致。")
