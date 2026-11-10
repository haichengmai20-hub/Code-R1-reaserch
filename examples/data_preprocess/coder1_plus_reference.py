"""补强项 11：社区参考解获取与「参考解不可得 → 过滤」（定案 D12）。

参考解不参与判分（`extra_info.reference` 判分器不读），但它是行为层验收的必需品：
题面 ↔ 样例 ↔ 参考解的三角自洽少了参考解就无从做起。LeetCode 官方题解在付费墙后，
按可信度取社区题解开源库 doocs/leetcode（2026-11-09 实测：训练池 1868 道算法题
文件夹级覆盖 1868/1868，README 含 Python3 解抽样约 80%；3/4 题「方法一」与冻结
completion 相似度 ≥0.99——上游参考解的出处即 doocs）。

D12（RLVR学习资料/设计定案表）：参考解拿不到的题在过滤关整题丢弃，不用 LLM 兜底
生成参考解（宁缺毋滥）；丢弃不静默——记清单与原因；丢弃非随机（偏新题/冷门题），
要对比过滤前后的难度分布。与 TACO「solutions 跑不通测试就删题」同构：把
「过不了行为层验收的题不要」前移一格到「没有参考解就无从验收」。

用法（仓库根目录执行；GitHub raw 本机直连，无需代理；全部断点续跑，缓存命中零请求）：
    # 门①（D12）：覆盖普查 + 丢弃清单与难度对照（全量约 1900 次请求、限速 1s/次）
    python examples/data_preprocess/coder1_plus_reference.py coverage \
        --out_dir data/reference_coverage
    # 门②（D14）：参考解可信性——对缓存的每题做行为层验收，纯 CPU、读 coverage 的 raw 缓存
    python examples/data_preprocess/coder1_plus_reference.py verify --out_dir data/reference_coverage
    # 只查指定的题（调试/抽样）
    python examples/data_preprocess/coder1_plus_reference.py coverage \
        --problems maximum-subarray defanging-an-ip-address --out_dir data/reference_coverage
    # 获取参考解并做行为层验收（原生拼法在补强沙箱跑仓库 test）
    python examples/data_preprocess/coder1_plus_reference.py fetch --problems maximum-subarray two-sum

coverage 产物（out_dir 下）：
    drop_reference_missing.json   门①丢弃清单（JSON list），给 coder1_plus.py --drop_task_ids
    coverage_report.json          逐题结果（含丢弃原因）+ 汇总（含难度分布对照）
    buckets/NNNN-NNNN.json        doocs 目录分片清单缓存；raw/NNNN.Title.md  题解原文缓存
verify 产物（out_dir 下）：
    drop_reference_untrusted.json 门②丢弃清单（doocs 全部方法都跑不通样例且对照组通过）
    verify_report.json            逐题验收结果（哪份方法通过、对照组状态、env_suspect 分层）
两份清单一起传：coder1_plus.py --drop_task_ids a.json b.json（2026-11-09 起支持多文件合并）。
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_BASE = "https://raw.githubusercontent.com/doocs/leetcode/main"
TREE_BASE = "https://github.com/doocs/leetcode/tree/main/solution"
HEADERS = {"User-Agent": "Mozilla/5.0 coder1-plus/1.0 (reference supplement 11)"}

# slug → 官方英文标题的启发式（连字符分段、首字母大写）覆盖不到的题写在这里
SPECIAL_TITLES = {
    "3sum": "3Sum", "powx-n": "Pow(x, n)", "n-queens": "N-Queens", "n-queens-ii": "N-Queens II",
    "sqrtx": "Sqrt(x)", "atoi": "String to Integer (atoi)",
}

V2_FILES = ["LeetCodeDataset-v2-rl-problems.jsonl", "LeetCodeDataset-v2-sft-problems.jsonl"]


def slug_to_title(slug):
    if slug in SPECIAL_TITLES:
        return SPECIAL_TITLES[slug]
    return " ".join(w[:1].upper() + w[1:] for w in slug.split("-"))


def doocs_dir(qid, title):
    lo = qid // 100 * 100
    return f"solution/{lo:04d}-{lo + 99:04d}/{qid:04d}.{title}"


def http_get(url, retries=2, timeout=10, budget=20.0):
    """带重试的 GET；单次尝试有总墙钟预算。失败抛异常（调用方记 fetch_error、不缓存，重跑可续）。

    为什么除了 socket 超时还要总预算：socket 超时按**每次读**计时，一个每十几秒滴一个
    字节的慢连接（2026-11-09 实测本机遇到过：进程卡在 SSL read 的 poll 上，每题拖几分钟）
    每次读都赶在超时前到一点数据，能无限拖下去。按块读 + 总预算把它截成 fetch_error。
    """
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)  # noqa: S310 固定域名
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 固定域名
                chunks, started = [], time.monotonic()
                while True:
                    chunk = resp.read1(4096)  # 单次底层读：socket 超时管「多久没数据」，budget 管「总共多久」
                    chunks.append(chunk)
                    if not chunk:
                        break
                    if time.monotonic() - started > budget:
                        raise TimeoutError(f"response exceeded {budget}s budget")
                return b"".join(chunks).decode("utf-8")
        except Exception:  # noqa: BLE001
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def load_pool():
    """训练池：v2 RL + v2 SFT 合并、task_id 去重保留首次（与 leetcode2k 同口径）。"""
    pool = {}
    for name in V2_FILES:
        path = REPO_ROOT / "LeetCodeDataset" / "data" / name
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                pool.setdefault(rec["task_id"], {
                    "qid": int(rec["meta"]["question_id"]),
                    "difficulty": rec["meta"].get("difficulty"),
                    "record": rec,
                })
    return pool


def bucket_folders(bucket, cache_dir):
    """一个分片的 doocs 文件夹清单：{qid: 标题}。HTML 树页面解析，结果缓存。"""
    cache = cache_dir / "buckets" / f"{bucket:04d}-{bucket + 99:04d}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    html = http_get(f"{TREE_BASE}/{bucket:04d}-{bucket + 99:04d}")
    pairs = re.findall(rf'"path":"solution/{bucket:04d}-{bucket + 99:04d}/(\d{{4}})\.([^"]+)"', html)
    folders = {int(qid): title for qid, title in pairs}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(folders, ensure_ascii=False), encoding="utf-8")
    time.sleep(1.0)
    return folders


def fetch_readme(relative_dir, cache_dir):
    """doocs README.md，成功才缓存（失败留给重跑重试）。"""
    name = relative_dir.split("/")[-1] + ".md"
    cache = cache_dir / "raw" / name
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    text = http_get(f"{RAW_BASE}/{urllib.parse.quote(relative_dir)}/README.md")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    time.sleep(1.0)
    return text


def extract_python_solutions(readme):
    """每个「### 方法」标题下 #### Python3 的围栏块（须含 class Solution）。"""
    solutions, method, lang = [], None, None
    in_fence, block, fence_len = False, [], 0
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


def check_one(task_id, info, cache_dir, titles=None):
    """单题覆盖检查。返回 (has_reference, reason, 细节)。"""
    qid = info["qid"]
    if qid >= 100000:
        return False, "lcci_out_of_doocs_range", {"qid": qid}  # 面试金典题不在 doocs 编号范围
    folders = bucket_folders(qid // 100 * 100, cache_dir)
    title = folders.get(qid) or (titles or {}).get(task_id) or slug_to_title(task_id)
    readme = fetch_readme(doocs_dir(qid, title), cache_dir)
    solutions = extract_python_solutions(readme)
    if not solutions:
        return False, "no_python_solution", {"qid": qid, "folder": title}
    return True, "ok", {"qid": qid, "folder": title, "methods": [s["method"] for s in solutions]}


def cmd_coverage(args):
    out_dir = Path(args.out_dir)
    pool = load_pool()
    targets = args.problems or sorted(pool)
    if args.problems:
        missing = [p for p in targets if p not in pool]
        if missing:
            sys.exit(f"训练池里没有这些题：{missing}")
    # 先用 slug 启发式兜底标题，减少对树页面解析的依赖（解析到了会覆盖）
    titles = {p: slug_to_title(p) for p in targets}
    rows, done, fetched = [], 0, 0
    for task_id in targets:
        info = pool[task_id]
        try:
            has_ref, reason, detail = check_one(task_id, info, out_dir, titles)
        except Exception as error:  # noqa: BLE001  网络失败：丢，但不缓存失败、重跑可续
            has_ref, reason, detail = False, f"fetch_error:{type(error).__name__}", {"qid": info["qid"]}
        rows.append({"task_id": task_id, "difficulty": info["difficulty"],
                     "has_reference": has_ref, "reason": reason, **detail})
        done += 1
        if reason == "ok":
            fetched += 1
        if done % 50 == 0:
            print(f"  … {done}/{len(targets)}（有参考解 {fetched}）")

    drop = sorted(r["task_id"] for r in rows if not r["has_reference"])
    kept = [r for r in rows if r["has_reference"]]
    dist = lambda rs: {d: sum(1 for r in rs if r["difficulty"] == d) for d in ("Easy", "Medium", "Hard")}
    kept_d, drop_d, all_d = dist(kept), dist([r for r in rows if not r["has_reference"]]), dist(rows)
    reasons = {}
    for r in rows:
        if not r["has_reference"]:
            reasons[r["reason"].split(":")[0]] = reasons.get(r["reason"].split(":")[0], 0) + 1
    report = {
        "policy": "D12：参考解拿不到 → 过滤关整题丢弃（不用 LLM 兜底；丢弃清单与原因随数据版本发布）",
        "checked": len(rows), "kept": len(kept), "dropped": len(drop),
        "drop_reasons": reasons,
        "difficulty_all": all_d, "difficulty_kept": kept_d, "difficulty_dropped": drop_d,
        "dropped_share_by_difficulty": {d: (round(drop_d[d] / all_d[d], 3) if all_d.get(d) else None)
                                        for d in ("Easy", "Medium", "Hard")},
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drop_reference_missing.json").write_text(json.dumps(drop, ensure_ascii=False), encoding="utf-8")
    (out_dir / "coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检查 {len(rows)} 题：有参考解 {len(kept)}，丢弃 {len(drop)}（原因分布 {reasons}）")
    print(f"难度分布  全部 {all_d} → 保留 {kept_d} / 丢弃 {drop_d}"
          f"（各难度丢弃占比 {report['dropped_share_by_difficulty']}——丢弃非随机，过滤前后要对比）")
    print(f"产物：{out_dir}/drop_reference_missing.json（给 --drop_task_ids）、coverage_report.json")
    return 0


def _load_run_python():
    """尽量拿到补强沙箱的 run_python；无 numpy/torch 的环境走 supplement_env 的包桩。"""
    try:
        from verl.utils.reward_score.coder1_plus.sandbox import run_python
        return run_python
    except Exception:  # noqa: BLE001
        sys.path.insert(0, str(REPO_ROOT / "RLVR学习资料" / "examples" / "source-supplements"))
        from supplement_env import load_reward_score
        load_reward_score()
        from verl.utils.reward_score.coder1_plus.sandbox import run_python
        return run_python


def cmd_verify(args):
    """门②（D14）：对 coverage 缓存的每题做参考解行为层验收——原生拼法（前缀+解+test+check）跑仓库 test。

    any 语义：任一 doocs 方法通过 = 该题有可信参考解，并记录是哪份通过（供变异审计、扩容验收取用）。
    对照组（仓库冻结 completion）也失败时按 BC-24 铁律判 env_suspect：先查环境，不算参考解不可信、
    不自动丢弃。全部方法失败且对照组通过才进 untrusted 清单。
    """
    out_dir = Path(args.out_dir)
    run_python = _load_run_python()
    pool = load_pool()
    qid_index = {info["qid"]: (task_id, info) for task_id, info in pool.items()}
    results = []
    for path in sorted((out_dir / "raw").glob("*.md")):
        m = re.match(r"^(\d{4})\.", path.name)
        if not m:
            continue
        hit = qid_index.get(int(m.group(1)))
        if hit is None:
            continue  # 缓存里有但不在本仓训练池（如 v2 test 题）
        task_id, info = hit
        rec = info["record"]

        def assemble(code, rec=rec):
            return "\n\n".join([rec["prompt"], code, rec["test"], f"check({rec['entry_point']})"])

        solutions = extract_python_solutions(path.read_text(encoding="utf-8"))
        methods = [{"method": s["method"], "status": run_python(assemble(s["code"]), timeout=30).status}
                   for s in solutions]
        control = run_python(assemble(rec["completion"]), timeout=30).status
        trusted = any(x["status"] == "ok" for x in methods)
        verdict = "trusted" if trusted else ("env_suspect" if control != "ok" else "untrusted")
        results.append({"task_id": task_id, "difficulty": info["difficulty"], "verdict": verdict,
                        "passing_methods": [x["method"] for x in methods if x["status"] == "ok"],
                        "methods": methods, "repo_completion_control": control})
        mark = "✓" if trusted else ("⚠" if verdict == "env_suspect" else "✗")
        print(f"{mark} {task_id:34s} 对照={control:12s} "
              + ", ".join(f"{x['method']}={x['status']}" for x in methods))
    counts = {}
    for row in results:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    drop = sorted(row["task_id"] for row in results if row["verdict"] == "untrusted")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drop_reference_untrusted.json").write_text(
        json.dumps(drop, ensure_ascii=False), encoding="utf-8")
    (out_dir / "verify_report.json").write_text(
        json.dumps({"counts": counts, "drop": drop, "rows": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"验 {len(results)} 题：{counts}")
    print(f"产物：{out_dir}/drop_reference_untrusted.json（给 --drop_task_ids，与门①清单一起传）、"
          f"verify_report.json（env_suspect 不自动丢——先按 BC-24 铁律查环境）")
    return 0


def cmd_fetch(args):
    """获取参考解并做行为层验收：原生拼法（前缀+解+test+check）在补强沙箱跑仓库 test。"""
    run_python = _load_run_python() if not args.no_verify else None
    out_dir = Path(args.out_dir)
    pool = load_pool()
    results = []
    for task_id in args.problems:
        if task_id not in pool:
            print(f"✗ {task_id}: 训练池里没有"); continue
        rec = pool[task_id]["record"]
        try:
            readme = fetch_readme(doocs_dir(pool[task_id]["qid"], slug_to_title(task_id)), out_dir)
        except Exception as error:  # noqa: BLE001
            print(f"✗ {task_id}: 抓取失败 {type(error).__name__}"); continue
        solutions = extract_python_solutions(readme)
        entry = {"task_id": task_id, "methods": []}
        program = lambda code: "\n\n".join([rec["prompt"], code, rec["test"], f"check({rec['entry_point']})"])
        control = None
        if run_python:
            control = run_python(program(rec["completion"])).status  # 冻结 completion 对照组
        for sol in solutions:
            status = run_python(program(sol["code"])).status if run_python else "not_verified"
            entry["methods"].append({"method": sol["method"], "status": status, "code": sol["code"]})
        entry["repo_completion_control"] = control
        results.append(entry)
        print(("✓" if all(m["status"] == "ok" for m in entry["methods"]) and control in (None, "ok") else "✗")
              + f" {task_id}: 对照组={control} " + ", ".join(f"{m['method']}={m['status']}" for m in entry["methods"]))
    (out_dir / "fetched_solutions.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物：{out_dir}/fetched_solutions.json")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cov = sub.add_parser("coverage", help="覆盖普查 + D12 丢弃清单")
    cov.add_argument("--problems", nargs="*", help="只查这些 task_id（默认全池）")
    cov.add_argument("--out_dir", default="data/reference_coverage")
    cov.set_defaults(func=cmd_coverage)
    ver = sub.add_parser("verify", help="门②（D14）：缓存参考解的行为层验收 → untrusted 清单")
    ver.add_argument("--out_dir", default="data/reference_coverage")
    ver.set_defaults(func=cmd_verify)
    fet = sub.add_parser("fetch", help="获取参考解 + 行为层验收")
    fet.add_argument("--problems", nargs="*", required=True)
    fet.add_argument("--out_dir", default="data/reference_coverage")
    fet.add_argument("--no_verify", action="store_true", help="只抓取不跑沙箱验收")
    fet.set_defaults(func=cmd_fetch)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
