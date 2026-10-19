"""在没有 torch、numpy、firejail 的 CPU 学习环境里，加载真实的 Code-R1 判分代码做验证。

做法：
1. 只为 ``verl``、``verl.utils`` 放两个空的包对象，避免执行它们会拉起 torch 的 ``__init__``；
   ``verl.utils.reward_score`` 用真实的 ``__init__.py`` 加载，所以路由是真实路由。
2. numpy、requests 没装时放最小桩：原版判分只用 ``np.ndarray`` 做类型判断，``requests`` 只在远程后端用。
3. 把原版 ``coder1.code_exec`` 换成本地子进程执行器，成功与失败的返回格式与 firejail 后端一致。

本地子进程不是沙箱：只允许执行数据集自带的参考解和本课固定的候选，不能执行未知的模型代码。
"""
import ast
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REWARD_DIR = REPO / "verl" / "utils" / "reward_score"
BUILDER_PATH = REPO / "examples" / "data_preprocess" / "coder1_plus.py"
TEST_HELPERS = ("tree_node", "is_same_tree", "list_node", "is_same_list")


def _ensure_stub(name, **attrs):
    if name in sys.modules or importlib.util.find_spec(name) is not None:
        return
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


def _ensure_package(name, path):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package


def load_reward_score():
    """返回 (真实路由模块, 原版 coder1, 补强版 coder1_plus)。"""
    os.environ.setdefault("CODER1_EXEC", "firejail")  # 只决定导入哪个后端模块；执行器随后被替换
    _ensure_stub("numpy", ndarray=type("ndarray", (), {}))
    _ensure_stub("requests", get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("requests stub")))
    _ensure_package("verl", REPO / "verl")
    _ensure_package("verl.utils", REPO / "verl" / "utils")
    if "verl.utils.reward_score" not in sys.modules:
        spec = importlib.util.spec_from_file_location("verl.utils.reward_score", REWARD_DIR / "__init__.py",
                                                      submodule_search_locations=[str(REWARD_DIR)])
        router = importlib.util.module_from_spec(spec)
        sys.modules["verl.utils.reward_score"] = router
        spec.loader.exec_module(router)
    router = sys.modules["verl.utils.reward_score"]
    coder1 = importlib.import_module("verl.utils.reward_score.coder1")
    coder1_plus = importlib.import_module("verl.utils.reward_score.coder1_plus")
    coder1.code_exec = local_exec
    return router, coder1, coder1_plus


def local_exec(code, stdin=None, timeout=10, pytest=None, hash_seed=None):
    """本地子进程执行器，返回值格式与 firejail_exec.code_exec_firejail 相同。

    hash_seed 用来固定 PYTHONHASHSEED，检查结果是否依赖 set 和 dict 的迭代顺序。
    """
    if pytest is not None:
        raise NotImplementedError("local_exec does not run pytest ground truths")
    prefix = "Failed to execute program: "
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "program.py"
        script.write_text(code, encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        if hash_seed is not None:
            env["PYTHONHASHSEED"] = str(hash_seed)
        try:
            result = subprocess.run([sys.executable, "-I", "-B", str(script)], cwd=workdir, env=env,
                                    input=stdin.encode() if stdin else None, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, prefix + f"STDOUT:\n\n\nSTDERR:\nTimeout after {timeout}s"
    stdout, stderr = result.stdout.decode(), result.stderr.decode().strip()
    if result.returncode == 0:
        return True, stdout
    return False, prefix + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"


def load_builder():
    spec = importlib.util.spec_from_file_location("coder1_plus_preprocess", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def self_contained_candidate(prelude: str, completion: str) -> str:
    """模拟一个认真的模型回答：写全了 imports 和 TreeNode、ListNode 类，

    但不可能知道数据集私有的测试辅助函数，所以去掉 tree_node、is_same_tree、list_node、is_same_list。
    """
    tree = ast.parse(prelude)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name in TEST_HELPERS)]
    return ast.unparse(tree) + "\n\n" + completion


def wrap_response(code: str) -> str:
    return "<think>Fixed reference candidate; no model generation.</think><answer>```python\n" + code.rstrip() + "\n```</answer>"
