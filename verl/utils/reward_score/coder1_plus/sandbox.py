"""补强项 5：结构化执行层与隔离后端。

原版 firejail 后端只返回「退出码是否为 0」和一段混在一起的输出；超时后靠 firejail 自己结束，
奖励管理器那一层的 asyncio.wait_for 只是不再等待，不保证子进程被清理。学习资料主框架第三章与第五章
要求：执行结果分成可区分的状态、记录耗时、超时后确实清理进程、基础设施故障与候选失败分开。

本模块提供：
- ExecResult：status 取 ok、compile_error、runtime_error、memory_error、timeout、killed、infra_error 之一，
  另有退出码、标准输出、标准错误、耗时和所用后端。
- 三个后端：firejail（Linux 训练环境，参数与原版相同）、sandbox-exec（macOS 本机开发：禁网、只允许写
  临时目录、禁止读取受保护目录，默认保护本仓库）、local（不隔离，只能显式选择，用于执行可信代码）。
- 超时后杀掉整个进程组，并回收子进程。
后端由环境变量 CODER1_PLUS_SANDBOX 选择：auto（默认）、firejail、sandbox-exec、local。auto 在 Linux 上用
firejail，在 macOS 上用 sandbox-exec；都不可用时返回 infra_error，不会退回到不隔离的执行。
"""

import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import resource
except ImportError:  # 非 Unix 平台
    resource = None

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_LIMIT = 1 << 20  # 标准输出与标准错误各最多保留 1 MiB
_ERROR_MSG_PREFIX = "Failed to execute program: "


@dataclass
class ExecResult:
    status: str
    returncode: Optional[int]
    stdout: str
    stderr: str
    duration: float
    backend: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _classify(returncode, stderr, timed_out):
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "ok"
    if returncode is not None and returncode < 0:
        return "timeout" if -returncode == getattr(signal, "SIGXCPU", -1) else "killed"
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    if tail.startswith(("SyntaxError", "IndentationError", "TabError")):
        return "compile_error"
    if "MemoryError" in tail:
        return "memory_error"
    return "runtime_error"


def choose_backend(requested=None):
    name = (requested or os.environ.get("CODER1_PLUS_SANDBOX", "auto")).strip().lower()
    if name != "auto":
        return name
    system = platform.system()
    if system == "Linux" and shutil.which("firejail"):
        return "firejail"
    if system == "Darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    return "unavailable"


def _sandbox_profile(writable_dir, protected_paths):
    allow_write = " ".join(f'(subpath "{path}")' for path in [writable_dir])
    deny_read = "\n".join(f'(deny file-read* (subpath "{path}"))' for path in protected_paths)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* {allow_write} (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") (regex #"^/dev/tty"))
{deny_read}
"""


def _limits(timeout):
    """子进程的资源上限。

    RLIMIT_CPU 只在 Linux 上设置：2026-10-14 在 macOS 14.2 实测，通过 preexec_fn 设置 RLIMIT_CPU 后，
    无论上限多少秒，子进程都会立刻收到 SIGXCPU。macOS 上靠墙钟超时加整组清理兜底。
    """
    limits = [(resource.RLIMIT_FSIZE, 2 << 20), (resource.RLIMIT_NOFILE, 64)] if resource else []
    if resource is not None and platform.system() == "Linux":
        limits.append((resource.RLIMIT_CPU, int(timeout) + 1))

    def apply():
        for limit, value in limits:
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass
        try:  # 地址空间上限在 macOS 上通常不生效，失败就跳过
            resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
        except (ValueError, OSError):
            pass
    return apply


def _command(backend, script, workdir, timeout, protected_paths):
    if backend == "firejail":  # 参数与原版 firejail_exec 相同
        return ["firejail", "--private", "--quiet", "--seccomp=socket", "--profile=pip", f"--whitelist={workdir}",
                "--rlimit-nproc=32", "--rlimit-nofile=32", "--rlimit-fsize=2m", "--rlimit-as=4096m",
                f"--timeout=00:00:{int(timeout)}", "python3", "-I", "-B", script]
    if backend == "sandbox-exec":
        profile = _sandbox_profile(workdir, protected_paths)
        return ["sandbox-exec", "-p", profile, sys.executable, "-I", "-B", script]
    if backend == "local":
        return [sys.executable, "-I", "-B", script]
    raise ValueError(f"unknown sandbox backend: {backend}")


def run_python(code, stdin=None, timeout=30, backend=None, protected_paths=None) -> ExecResult:
    """在隔离后端里运行一段 Python 代码，返回结构化结果。"""
    backend = choose_backend(backend)
    if backend == "unavailable":
        return ExecResult("infra_error", None, "", "no sandbox backend available; refusing to run unisolated",
                          0.0, backend)
    if protected_paths is None:
        protected_paths = [os.path.realpath(REPO_ROOT)]
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as raw_workdir:
        workdir = os.path.realpath(raw_workdir)
        script = os.path.join(workdir, "program.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(code)
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env["OPENBLAS_NUM_THREADS"] = "1"
        try:
            process = subprocess.Popen(_command(backend, script, workdir, timeout, protected_paths), cwd=workdir,
                                       env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True,
                                       # 2026-11-01：Windows 的 Popen 不支持 preexec_fn（传入直接 ValueError），
                                       # 与 macOS 的 RLIMIT_CPU 同理，靠墙钟超时加整组清理兜底、不设资源上限。
                                       preexec_fn=(_limits(timeout) if (backend != "firejail" and resource is not None
                                                                        and platform.system() != "Windows") else None))
        except OSError as error:
            return ExecResult("infra_error", None, "", f"failed to start {backend}: {error}",
                              time.monotonic() - started, backend)
        timed_out = False
        try:
            raw_out, raw_err = process.communicate(stdin.encode() if stdin else None, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)  # 杀掉整个进程组，包括候选代码派生的子进程
            except ProcessLookupError:
                pass
            raw_out, raw_err = process.communicate()
    stdout = raw_out[:OUTPUT_LIMIT].decode(errors="replace")
    stderr = raw_err[:OUTPUT_LIMIT].decode(errors="replace")
    if backend == "sandbox-exec" and stderr.startswith("sandbox-exec:"):
        return ExecResult("infra_error", process.returncode, stdout, stderr, time.monotonic() - started, backend)
    if backend == "firejail" and "Error: cannot" in stderr and process.returncode not in (0, None):
        return ExecResult("infra_error", process.returncode, stdout, stderr, time.monotonic() - started, backend)
    return ExecResult(_classify(process.returncode, stderr, timed_out), process.returncode, stdout, stderr,
                      time.monotonic() - started, backend)


def code_exec(code, stdin=None, timeout=30, pytest=None):
    """与原版 code_exec 相同的返回约定：(是否成功, 标准输出或错误信息)。"""
    if pytest is not None:
        raise NotImplementedError("coder1_plus sandbox does not run pytest ground truths")
    result = run_python(code, stdin=stdin, timeout=timeout)
    if result.ok:
        return True, result.stdout
    return False, (_ERROR_MSG_PREFIX + f"[{result.status}] STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
