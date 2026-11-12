"""Persistent CLI operations and the background worker lifecycle."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

from .codex import Cancelled
from .workspace import WorkspaceBusy, exclusive_workspace


ACTIVE = ("starting", "running")
RESUMABLE = ("paused", "canceled", "interrupted", "quota_limited", "failed")
START_TIMEOUT = 30
DEFAULT_CONFIG = {"codex": "auto", "subagents": False, "scales": None, "input_mode": "original"}


class CommandError(Exception):
    def __init__(self, code: str, message: str, *, status: int = 2, details: dict | None = None):
        super().__init__(message)
        self.code, self.status, self.details = code, status, details or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


class Operations:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "automation.sqlite3"

    @contextmanager
    def connection(self, *, write: bool = False):
        if write:
            self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path if self.path.is_file() or write else ":memory:", timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                if write:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("""
                        CREATE TABLE IF NOT EXISTS operations (
                            id TEXT PRIMARY KEY, request_id TEXT UNIQUE, request TEXT NOT NULL,
                            token TEXT NOT NULL, state TEXT NOT NULL, action TEXT, pid INTEGER,
                            created_at REAL NOT NULL, updated_at REAL NOT NULL,
                            progress TEXT NOT NULL DEFAULT '{}', error TEXT, quota_reset INTEGER
                        )
                    """)
                    db.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
                yield db
        finally:
            db.close()

    def config(self) -> dict:
        with self.connection() as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='settings'").fetchone():
                return DEFAULT_CONFIG.copy()
            return DEFAULT_CONFIG | {row["name"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM settings")}

    def set_config(self, values: dict):
        with self.connection(write=True) as db:
            db.executemany(
                "INSERT OR REPLACE INTO settings VALUES (?, ?)",
                ((key, json.dumps(value)) for key, value in values.items()),
            )

    def rows(self, *, operation_id: str | None = None, request_id: str | None = None) -> list[dict]:
        with self.connection() as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='operations'").fetchone():
                return []
            rows = db.execute(
                "SELECT * FROM operations WHERE (? IS NULL OR id=?) AND (? IS NULL OR request_id=?) "
                "ORDER BY rowid DESC", (operation_id, operation_id, request_id, request_id),
            ).fetchall()
        return [self.decode(row) for row in rows]

    @staticmethod
    def decode(row) -> dict:
        item = dict(row)
        for key in ("request", "progress", "error"):
            if item[key] is not None:
                item[key] = json.loads(item[key])
        return item

    def get(self, operation_id: str) -> dict:
        rows = self.rows(operation_id=operation_id)
        if not rows:
            raise CommandError("NOT_FOUND", "Operation not found.", status=3)
        return rows[0]

    def change(self, operation_id: str, current_token: str, *, expected_state=None, before_commit=None, **values):
        values["updated_at"] = time.time()
        for key in ("progress", "error"):
            if key in values and values[key] is not None:
                values[key] = json.dumps(values[key])
        fields = ", ".join(f"{key}=?" for key in values)
        with self.connection(write=True) as db:
            db.execute(f"UPDATE operations SET {fields} WHERE id=? AND token=? AND (? IS NULL OR state=?)",
                       (*values.values(), operation_id, current_token, expected_state, expected_state))
            if before_commit is not None:
                before_commit()

    def recover(self, *, locked: bool = False):
        active = next((item for item in self.rows() if item["state"] in ACTIVE), None)
        if not active or (active["state"] == "starting" and time.time() - active["updated_at"] < START_TIMEOUT):
            return
        if not locked:
            try:
                with exclusive_workspace(self.root):
                    self.recover(locked=True)
            except WorkspaceBusy:
                pass
            return
        self.change(
            active["id"], active["token"], expected_state=active["state"], state="interrupted", action=None,
            error={"code": "WORKER_INTERRUPTED", "message": "The worker exited before reporting a result."},
        )

    def require_idle(self):
        self.recover(locked=True)
        if any(item["state"] in ACTIVE for item in self.rows()):
            raise WorkspaceBusy("A Regimagen operation is starting in this workspace.")

    def replay(self, request: dict, key: str | None) -> dict | None:
        rows = self.rows(request_id=key) if key is not None else []
        if not rows:
            return None
        if rows[0]["request"] != request:
            raise CommandError("REQUEST_ID_CONFLICT", "This request ID already has different parameters.", status=4)
        return rows[0]

    def submit(self, request: dict, key: str | None, *, validate=lambda: None) -> dict:
        if previous := self.replay(request, key):
            self.recover()
            return self.get(previous["id"])
        with exclusive_workspace(self.root):
            if previous := self.replay(request, key):
                return previous
            self.require_idle()
            validate()
            operation_id, token = uuid4().hex, uuid4().hex
            now = time.time()
            with self.connection(write=True) as db:
                db.execute(
                    "INSERT INTO operations (id, request_id, request, token, state, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 'starting', ?, ?)",
                    (operation_id, key, json.dumps(request), token, now, now),
                )
            self.launch(operation_id, token)
        return self.get(operation_id)

    def launch(self, operation_id: str, token: str):
        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command += ["-m", "regimagen.cli"]
        command += ["--data-dir", str(self.root), "_worker", operation_id, token]
        options = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.name == "nt":
            options["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
            )
        else:
            options["start_new_session"] = True
        try:
            process = subprocess.Popen(command, **options)
        except OSError as exc:
            error = CommandError("WORKER_START_FAILED", str(exc), status=1, details={"operation_id": operation_id})
            self.change(operation_id, token, state="failed", error=error.as_dict())
            raise error from exc
        self.change(operation_id, token, pid=process.pid)

    def control(self, operation_id: str, action: str) -> dict:
        self.recover()
        if action == "resume":
            with exclusive_workspace(self.root):
                self.require_idle()
                item = self.get(operation_id)
                if item["state"] not in RESUMABLE or item["request"]["kind"] not in ("generate", "batch"):
                    raise CommandError("INVALID_STATE", "This operation cannot be resumed.", status=4)
                token = uuid4().hex
                progress = item["progress"]
                if item["request"]["kind"] == "batch":
                    progress = progress | {"outputs": [], "completed_stages": 0}
                self.change(operation_id, item["token"], state="starting", token=token, action=None, error=None,
                            quota_reset=None, pid=None, progress=progress)
                self.launch(operation_id, token)
        else:
            with self.connection(write=True) as db:
                row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
                if row is None:
                    raise CommandError("NOT_FOUND", "Operation not found.", status=3)
                if row["state"] not in ACTIVE:
                    if (action, row["state"]) in (("pause", "paused"), ("cancel", "canceled")):
                        return self.decode(row)
                    raise CommandError("INVALID_STATE", "This operation is no longer running.", status=4)
                if action == "pause" and json.loads(row["request"])["kind"] not in ("generate", "batch"):
                    raise CommandError("INVALID_STATE", "Only generation and scale batches can be paused.", status=4)
                if row["action"] != "cancel":
                    db.execute("UPDATE operations SET action=? WHERE id=?", (action, operation_id))
        return self.get(operation_id)

    def worker(self, operation_id: str, token: str, run):
        try:
            with ExitStack() as workspace:
                workspace.enter_context(exclusive_workspace(self.root, timeout=START_TIMEOUT * 1000))
                item = self.get(operation_id)
                if item["token"] != token or item["state"] != "starting":
                    return
                self.change(operation_id, token, state="running")
                result = self.run_worker(item, run)
                # Release the filesystem lock before publishing the final state.
                # The write transaction keeps recovery from racing the update.
                self.change(operation_id, token, before_commit=workspace.close, **result)
        except WorkspaceBusy as exc:
            item = self.get(operation_id)
            if item["token"] == token and item["state"] == "starting":
                self.change(operation_id, token, expected_state="starting", state="failed",
                            error={"code": "WORKSPACE_BUSY", "message": str(exc)})

    def run_worker(self, item: dict, run):
        operation_id, token = item["id"], item["token"]
        pause, cancel, finished = Event(), Event(), Event()
        progress = item["progress"]

        def report(**changes):
            progress.update(changes)
            self.change(operation_id, token, progress=progress)

        def poll():
            action = self.get(operation_id)["action"]
            if action == "cancel":
                cancel.set()
            elif action == "pause":
                pause.set()

        def watch():
            while not finished.wait(0.2):
                poll()

        poll()
        watcher = Thread(target=watch, daemon=True)
        watcher.start()
        try:
            outcome = run(item, pause, cancel, report)
            state = outcome.state
            error = {"code": outcome.code, "message": outcome.message} if outcome.code else None
            quota_reset = outcome.quota_reset
        except Cancelled:
            state, error, quota_reset = "canceled", None, None
        except Exception as exc:
            code = "OUTPUT_EXISTS" if isinstance(exc, FileExistsError) else "OPERATION_FAILED"
            error = exc.as_dict() if isinstance(exc, CommandError) else {"code": code, "message": str(exc)}
            state, quota_reset = "failed", None
        finally:
            finished.set()
            watcher.join()
        if "active_tile" in progress:
            progress["active_tile"] = None
        return dict(state=state, action=None, error=error, quota_reset=quota_reset, progress=progress)
