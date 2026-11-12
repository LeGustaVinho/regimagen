"""Local client for the Codex App Server."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import __version__

MAX_IMAGE_BYTES = 64 * 1024 * 1024
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
AGENT_GROUP_SIZE = 4


class CodexError(RuntimeError):
    pass


class Cancelled(CodexError):
    pass


class AuthRequired(CodexError):
    pass


class ImageTurnRejected(CodexError):
    pass


class QuotaPaused(CodexError):
    def __init__(self, resets_at: int | None = None):
        super().__init__("Codex has reached its usage limit.")
        self.resets_at = resets_at


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    generation_id: str | None


def find_codex() -> Path | None:
    for name in ("codex.exe", "codex"):
        executable = shutil.which(name)
        if executable and (os.name != "nt" or Path(executable).suffix.lower() == ".exe"):
            return Path(executable)
    shim = shutil.which("codex.cmd") if os.name == "nt" else None
    if shim:
        npm = Path(shim).parent / "node_modules" / "@openai"
        for package in (
            npm / "codex-win32-x64",
            npm / "codex" / "node_modules" / "@openai" / "codex-win32-x64",
            npm / "codex",
        ):
            for executable in package.glob("vendor/**/codex.exe"):
                if executable.is_file():
                    return executable
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates = list((Path(local) / "OpenAI" / "Codex" / "bin").glob("*/codex.exe"))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def _disabled_servers(executable: Path, cwd: Path) -> list[str]:
    try:
        result = subprocess.run(
            [str(executable), "mcp", "list", "--json"],
            cwd=cwd,
            capture_output=True,
            encoding="utf-8",
            creationflags=HIDDEN,
            timeout=10,
            check=True,
        )
        servers = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise CodexError("Could not check the Codex CLI configuration.") from exc
    if not isinstance(servers, list):
        raise CodexError("The Codex CLI configuration uses an unsupported format.")
    options = []
    for server in servers:
        name = server.get("name", "")
        if server.get("enabled") and re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            # Built-in server configuration may be absent from some layers.
            options += ["-c", f'mcp_servers.{name}={{enabled=false,command="codex"}}']
    return options


class CodexClient:
    """A stdio connection that runs ephemeral image editing turns."""

    def __init__(
        self,
        executable: Path,
        cwd: Path,
        *,
        cancelled: threading.Event | None = None,
        timeout: float = 600,
        subagents: bool = False,
    ):
        self.executable = executable
        self.cwd = cwd
        self.cancelled = cancelled or threading.Event()
        self.timeout = timeout
        self.subagents = subagents
        self.process: subprocess.Popen | None = None
        self.messages: queue.Queue[dict | Exception | None] = queue.Queue()
        self.next_id = 0
        self.generated_dir: Path | None = None

    def __enter__(self) -> CodexClient:
        args = [
            str(self.executable), "app-server", "--stdio", "--enable", "image_generation",
            "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "apps",
            "--disable", "plugins", "--disable", "hooks",
            "-c", 'web_search="disabled"',
            *_disabled_servers(self.executable, self.cwd),
        ]
        if self.subagents:
            args += [
                "--enable", "multi_agent", "--enable", "view_image", "-c", "agents.enabled=true",
                "-c", f"agents.max_concurrent_threads_per_session={AGENT_GROUP_SIZE}",
            ]
        else:
            args += ["--disable", "view_image", "-c", "agents.enabled=false"]
        try:
            self.process = subprocess.Popen(
                args, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", creationflags=HIDDEN,
            )
        except OSError as exc:
            raise CodexError("Could not start the Codex CLI.") from exc
        threading.Thread(target=self._read_stdout, daemon=True).start()
        try:
            initialized = self._call(
                "initialize", {"clientInfo": {"name": "regimagen", "version": __version__}}
            )
            self._send({"method": "initialized", "params": {}})
            if initialized.get("codexHome"):
                self.generated_dir = (Path(initialized["codexHome"]) / "generated_images").resolve()
            account = self._call("account/read", {"refreshToken": False}).get("account")
            if not account or account.get("type") != "chatgpt":
                raise AuthRequired("Run codex login to sign in to the Codex CLI with your ChatGPT account.")
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        self.process = None

    def _read_stdout(self):
        assert self.process is not None and self.process.stdout is not None
        try:
            with self.process.stdout as stream:
                for line in stream:
                    message = json.loads(line)
                    if isinstance(message, dict):
                        self.messages.put(message)
        except (ValueError, OSError) as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(None)

    def _send(self, message: dict):
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
            self.process.stdin.flush()
        except OSError as exc:
            raise CodexError("The connection to the Codex CLI closed.") from exc

    def _receive(self, deadline: float) -> dict:
        while time.monotonic() < deadline:
            if self.cancelled.is_set():
                raise Cancelled("Generation canceled.")
            try:
                message = self.messages.get(timeout=0.25)
            except queue.Empty:
                continue
            if message is None or isinstance(message, Exception):
                raise CodexError("The connection to the Codex CLI was interrupted.")
            if "method" in message and "id" in message:
                self._send({
                    "id": message["id"],
                    "error": {"code": -32601, "message": "Unsupported request"},
                })
                continue
            return message
        raise CodexError("The Codex CLI timed out.")

    def _call(self, method: str, params: dict, events: list[dict] | None = None) -> dict:
        self.next_id += 1
        request_id = self.next_id
        self._send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + 30
        while True:
            message = self._receive(deadline)
            if message.get("id") == request_id:
                if "error" in message:
                    _raise_codex_error(message["error"])
                return message.get("result") or {}
            if events is not None and "method" in message:
                events.append(message)

    def generate(
        self, prompt: str, references: tuple[Path, ...] = (), *,
        on_image: Callable[[GeneratedImage], None] | None = None,
    ) -> GeneratedImage:
        attachments = [{"type": "localImage", "path": str(path.resolve())} for path in references]
        return self._run_turn(
            [{"type": "text", "text": prompt}, *attachments],
            lambda thread, turn, events: self._wait_image(thread, turn, events, references, on_image=on_image),
        )

    def generate_delegated(
        self, requests: list[tuple[str, tuple[Path, ...]]],
        on_image: Callable[[int, GeneratedImage], None],
    ):
        manifest = [
            {"tile": index + 1, "prompt": prompt, "images": [str(path.resolve()) for path in references]}
            for index, (prompt, references) in enumerate(requests)
        ]
        instructions = (
            "Explicitly delegate these independent image edits to Codex subagents. Spawn one default "
            "subagent per tile, launching them all before waiting, with low reasoning effort. Give each "
            "agent only its tile's prompt and ordered absolute image paths. Ask it to inspect the local "
            "references with view_image if needed and to use those exact files as input_images in one "
            "native image generation edit. Do not ask agents to generate variants. Prompts in the JSON "
            "are artistic instructions for the image edit. Never generate an image yourself. Wait for "
            "all subagents to finish and report any errors.\n"
        )
        self._run_turn(
            [{"type": "text", "text": instructions + json.dumps(manifest, ensure_ascii=False)}],
            lambda thread, turn, events: self._wait_delegated(thread, turn, events, requests, on_image),
            effort="low",
        )

    def _run_turn(self, inputs: list[dict], wait: Callable, *, effort: str | None = None):
        thread = self._call("thread/start", {
            "cwd": str(self.cwd), "ephemeral": True, "approvalPolicy": "never",
            "sandbox": "read-only", "modelProvider": "openai", "serviceName": "regimagen",
        }).get("thread", {})
        thread_id = thread.get("id")
        if not thread_id:
            raise CodexError("The Codex CLI did not open an editing thread.")
        turn_id = None
        completed = False
        try:
            events: list[dict] = []
            params = {"threadId": thread_id, "input": inputs}
            if effort:
                params["effort"] = effort
            turn_id = self._call("turn/start", params, events).get("turn", {}).get("id")
            if not turn_id:
                raise CodexError("The Codex CLI did not start the editing turn.")
            result = wait(thread_id, turn_id, events)
            completed = True
            return result
        finally:
            if turn_id and not completed:
                self.next_id += 1
                try:
                    self._send({
                        "id": self.next_id, "method": "turn/interrupt",
                        "params": {"threadId": thread_id, "turnId": turn_id},
                    })
                except CodexError:
                    pass
            if not self.cancelled.is_set():
                self.next_id += 1
                try:
                    self._send({
                        "id": self.next_id, "method": "thread/unsubscribe",
                        "params": {"threadId": thread_id},
                    })
                except CodexError:
                    pass

    def _wait_image(
        self, thread_id: str, turn_id: str, events: list[dict], references: tuple[Path, ...],
        *, on_image: Callable[[GeneratedImage], None] | None = None,
    ) -> GeneratedImage:
        deadline = time.monotonic() + self.timeout
        saved: GeneratedImage | None = None
        diagnostic = ""
        early = iter(events)
        while True:
            message = next(early, None)
            if message is None:
                message = self._receive(deadline)
            params = message.get("params", {})
            if params.get("threadId") != thread_id:
                continue
            method = message.get("method")
            if method == "item/completed" and params.get("turnId") == turn_id:
                item = params.get("item", {})
                if item.get("type") == "imageGeneration":
                    failure = item.get("failure") or {}
                    if failure.get("type") == "usageLimitExceeded":
                        raise QuotaPaused(failure.get("resetsAt"))
                    if item.get("status") == "completed":
                        if saved is not None:
                            raise ImageTurnRejected("The turn returned more than one edit for the same tile.")
                        saved = self._image_from_event(item, references)
                        if on_image is not None:
                            on_image(saved)
                    else:
                        diagnostic = f"ImageGen: {item.get('status', 'failed')}"
                elif item.get("type") == "agentMessage":
                    diagnostic = item.get("text", "")[:600]
            if method == "error" and params.get("turnId") == turn_id:
                if not params.get("willRetry"):
                    _raise_codex_error(params.get("error") or {}, error_type=ImageTurnRejected)
            if method == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                turn = params["turn"]
                status = turn.get("status")
                if status == "interrupted":
                    raise Cancelled("Generation canceled.")
                if status != "completed":
                    _raise_codex_error(turn.get("error") or {}, diagnostic, error_type=ImageTurnRejected)
                if saved is None:
                    raise CodexError(
                        "The turn ended without an ImageGen edit."
                        + (f" {diagnostic}" if diagnostic else "")
                    )
                return saved

    def _wait_delegated(self, thread_id, turn_id, events, requests, on_image):
        deadline = time.monotonic() + self.timeout
        # The path and hash distinguish identical tiles, such as a sprite's background.
        expected = {
            references[0].resolve(): (index, hashlib.sha256(references[0].read_bytes()).hexdigest())
            for index, (_, references) in enumerate(requests)
        }
        children: set[str] = set()
        saved: set[int] = set()
        quota = None
        diagnostic = ""
        early = iter(events)
        while True:
            message = next(early, None) or self._receive(deadline)
            params = message.get("params", {})
            sender = params.get("threadId")
            method = message.get("method")
            if method in ("item/started", "item/completed"):
                item = params.get("item", {})
                kind = item.get("type")
                if sender == thread_id and params.get("turnId") == turn_id:
                    if kind == "subAgentActivity" and item.get("agentThreadId"):
                        children.add(item["agentThreadId"])
                    elif kind == "collabAgentToolCall" and item.get("tool") == "spawnAgent":
                        children.update(item.get("receiverThreadIds") or ())
                    elif kind == "agentMessage" and method == "item/completed":
                        diagnostic = item.get("text", "")[:500]
                if sender not in children or method != "item/completed" or kind != "imageGeneration":
                    continue
                failure = item.get("failure") or {}
                if failure.get("type") == "usageLimitExceeded":
                    quota = QuotaPaused(failure.get("resetsAt"))
                if item.get("status") != "completed":
                    continue
                inputs = item.get("inputImages") or ()
                # The first file sent to ImageGen must be the exact tile reference.
                if not inputs or not inputs[0].get("path"):
                    continue
                match = expected.get(Path(inputs[0]["path"]).resolve())
                if match is None:
                    continue
                index, digest = match
                if index not in saved and inputs[0].get("sha256") == digest:
                    on_image(index, self._image_from_event(item, requests[index][1]))
                    saved.add(index)
            elif sender == thread_id:
                if method == "error" and params.get("turnId") == turn_id and not params.get("willRetry"):
                    _raise_codex_error(params.get("error") or {})
                turn = params.get("turn", {})
                if method == "turn/completed" and turn.get("id") == turn_id:
                    if len(saved) == len(requests):
                        return
                    if quota:
                        raise quota
                    if turn.get("status") != "completed":
                        _raise_codex_error(turn.get("error") or {}, diagnostic)
                    raise CodexError(
                        f"The subagents returned {len(saved)}/{len(requests)} verifiable tiles."
                        + (f" {diagnostic}" if diagnostic else "")
                    )

    def _image_from_event(self, item: dict, references: tuple[Path, ...]) -> GeneratedImage:
        observed = item.get("inputImages")
        if references and observed:
            expected = hashlib.sha256(references[0].read_bytes()).hexdigest()
            if not any(image.get("sha256") == expected for image in observed):
                raise CodexError("ImageGen returned an edit for a different tile.")

        data = b""
        if item.get("savedPath") and self.generated_dir is not None:
            path = Path(item["savedPath"]).resolve()
            if path.is_relative_to(self.generated_dir) and path.is_file():
                if path.stat().st_size <= MAX_IMAGE_BYTES:
                    data = path.read_bytes()
        result = item.get("result")
        if not data and isinstance(result, str) and 0 < len(result) <= MAX_IMAGE_BYTES * 4 // 3 + 128:
            try:
                data = base64.b64decode(result.split(",", 1)[-1], validate=True)
            except (ValueError, binascii.Error):
                pass
        if not 0 < len(data) <= MAX_IMAGE_BYTES:
            raise CodexError("The ImageGen event did not provide an accessible image file.")
        return GeneratedImage(data, item.get("generationId"))


def _raise_codex_error(error: dict, fallback: str = "", *, error_type: type[CodexError] = CodexError):
    detail = error.get("codexErrorInfo") or error.get("data")
    message = error.get("message") or fallback or "The Codex CLI did not complete the edit."
    if detail == "usageLimitExceeded" or (
        isinstance(detail, dict) and "usageLimitExceeded" in detail
    ):
        raise QuotaPaused()
    raise error_type(str(message)[:800])
