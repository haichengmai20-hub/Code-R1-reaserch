"""补强项 4：受保护的判题协议。

原版把「候选代码 + 测试断言」拼成一个程序，按退出码判通过，于是：
- 候选代码在测试前 sys.exit(0) 就能拿 1.1；
- 期望答案写在同一个进程的断言里，候选代码理论上能读到；
- 只有整体通过或失败，不知道哪一条错、错成什么；
- 题目允许任意顺序时，断言仍按列表顺序比较，正确答案也可能被判错。

这里改成判分方与执行方分开：
1. 判分方把 check(candidate) 里的每条断言拆成「调用表达式」和「期望答案」。LeetCodeDataset 的断言只有
   三种写法：candidate(...) == 字面量、is_same_tree(candidate(...), tree_node(字面量))、
   is_same_list(candidate(...), list_node(字面量))，2090 题全部覆盖；拆不开时由调用方退回旧契约。
2. 子进程只拿到调用表达式，不拿期望答案。它执行候选代码，逐条调用，把输出的 repr 带着随机标记打印出来，
   最后打印完成事件。树和链表先转成层序列表或普通列表。
3. 判分方自己比较：缺少完成事件、缺少某一条结果、某一条报错或答案不同，都判失败。
   compare 为 unordered_outer 时，列表答案忽略外层顺序。
候选代码即使伪造了标记和完成事件，也不知道期望答案，伪造不出正确输出；提前退出则表现为结果缺失。
"""

import ast
import json

NONCE_PREFIX = "@@CODER1_PLUS@@"


class UnsupportedTests(ValueError):
    """断言写法超出协议支持的三种形式。"""


def _literal(node):
    return ast.literal_eval(node)


def parse_cases(functional: str):
    """返回 (入口表达式, 用例列表)。每个用例含调用表达式、输出种类、期望答案和标签。"""
    module = ast.parse(functional)
    check = next((node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "check"), None)
    final = module.body[-1] if module.body else None
    if check is None or not (isinstance(final, ast.Expr) and isinstance(final.value, ast.Call)
                             and getattr(final.value.func, "id", None) == "check" and len(final.value.args) == 1):
        raise UnsupportedTests("expected `def check(candidate)` followed by `check(<entry_point>)`")
    entry = ast.unparse(final.value.args[0])
    cases = []
    for index, statement in enumerate(check.body):
        if not isinstance(statement, ast.Assert):
            raise UnsupportedTests(f"statement {index} is not an assert")
        label = statement.msg.value if isinstance(statement.msg, ast.Constant) else f"assert-{index + 1}"
        test = statement.test
        try:
            if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                    and isinstance(test.left, ast.Call) and getattr(test.left.func, "id", None) == "candidate"):
                cases.append({"call": ast.unparse(test.left), "kind": "value",
                              "expected": _literal(test.comparators[0]), "label": label})
                continue
            if (isinstance(test, ast.Call) and getattr(test.func, "id", None) in ("is_same_tree", "is_same_list")
                    and len(test.args) == 2 and isinstance(test.args[0], ast.Call)
                    and getattr(test.args[0].func, "id", None) == "candidate"
                    and isinstance(test.args[1], ast.Call)
                    and getattr(test.args[1].func, "id", None) in ("tree_node", "list_node")
                    and len(test.args[1].args) == 1):
                kind = "tree" if test.func.id == "is_same_tree" else "list"
                expected = _literal(test.args[1].args[0])
                if kind == "tree":
                    while expected and expected[-1] is None:
                        expected.pop()
                cases.append({"call": ast.unparse(test.args[0]), "kind": kind, "expected": expected, "label": label})
                continue
        except (ValueError, SyntaxError, TypeError) as error:
            raise UnsupportedTests(f"statement {index}: expected value is not a literal ({error})") from error
        raise UnsupportedTests(f"statement {index}: unsupported assert form")
    if not cases:
        raise UnsupportedTests("no test cases")
    return entry, cases


_HARNESS = '''

import json as __coder1_json


def __coder1_emit(record):
    print({nonce!r} + " " + __coder1_json.dumps(record), flush=True)


def __coder1_tree(root):
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        if node is None:
            out.append(None)
            continue
        out.append(node.val)
        queue.extend((node.left, node.right))
    while out and out[-1] is None:
        out.pop()
    return out


def __coder1_list(head):
    out = []
    while head is not None and len(out) < 100000:
        out.append(head.val)
        head = head.next
    return out


__coder1_emit({{"event": "start", "cases": {count}}})
try:
    exec(compile({solution!r}, "<solution>", "exec"), globals())
except BaseException as __coder1_error:  # 包括 SystemExit：提前退出是失败，不是通过
    __coder1_emit({{"event": "solution_error", "error": type(__coder1_error).__name__}})
else:
    for __coder1_index, (__coder1_call, __coder1_kind) in enumerate({calls!r}):
        try:
            __coder1_out = eval(__coder1_call, globals(), {{"candidate": eval({entry!r}, globals())}})
            if __coder1_kind == "tree":
                __coder1_out = __coder1_tree(__coder1_out)
            elif __coder1_kind == "list":
                __coder1_out = __coder1_list(__coder1_out)
            __coder1_emit({{"case": __coder1_index, "status": "ok", "repr": repr(__coder1_out)}})
        except BaseException as __coder1_error:
            __coder1_emit({{"case": __coder1_index, "status": "error", "error": type(__coder1_error).__name__}})
    __coder1_emit({{"event": "done"}})
'''


def build_harness(prelude: str, solution_code: str, entry: str, cases, nonce: str) -> str:
    """子进程程序：前缀、执行候选代码、逐条调用并报告输出。期望答案不进入这段程序。"""
    calls = [(case["call"], case["kind"]) for case in cases]
    return prelude.rstrip() + _HARNESS.format(nonce=NONCE_PREFIX + nonce, count=len(cases), solution=solution_code,
                                              calls=calls, entry=entry)


def _same(output, expected, compare):
    if output == expected:
        return True
    if compare == "unordered_outer" and isinstance(output, list) and isinstance(expected, list):
        return sorted(map(repr, output)) == sorted(map(repr, expected))
    return False


def judge(stdout: str, cases, nonce: str, compare: str = "exact"):
    """判分方解析子进程输出并逐条比较，返回结构化结果。"""
    marker = NONCE_PREFIX + nonce + " "
    events, results, solution_error = [], {}, None
    for line in stdout.splitlines():
        if not line.startswith(marker):
            continue
        try:
            record = json.loads(line[len(marker):])
        except json.JSONDecodeError:
            continue
        if "case" in record:
            results.setdefault(record["case"], record)  # 同一条只认第一次报告
        else:
            events.append(record.get("event"))
            if record.get("event") == "solution_error" and solution_error is None:
                solution_error = record.get("error")
    details = []
    for index, case in enumerate(cases):
        record = results.get(index)
        if record is None:
            verdict = "missing"
        elif record.get("status") != "ok":
            verdict = f"runtime_error:{record.get('error')}"
        else:
            try:
                output = ast.literal_eval(record["repr"])
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                verdict = "unparseable_output"
            else:
                verdict = "pass" if _same(output, case["expected"], compare) else "wrong_answer"
        details.append({"label": case["label"], "verdict": verdict})
    completed = "start" in events and "done" in events and "solution_error" not in events
    passed = completed and all(item["verdict"] == "pass" for item in details)
    return {"passed": passed, "completed": completed, "events": events, "solution_error": solution_error,
            "n_pass": sum(1 for item in details if item["verdict"] == "pass"), "n_cases": len(cases),
            "cases": details}
