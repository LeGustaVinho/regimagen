import io
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from regimagen import automation, cli, jobs
from regimagen.automation import Operations
from regimagen.cli_images import render_reconstruction
from regimagen.codex import AuthRequired, GeneratedImage, QuotaPaused
from regimagen.composite import Compositor, STRIPE_HEIGHT
from regimagen.geometry import FRAME, Grid, Rect
from regimagen.imaging import prepare_frame
from regimagen.storage import Store
from regimagen.workspace import exclusive_workspace


@pytest.fixture
def command(tmp_path, capsys):
    root = tmp_path / "workspace ç"

    def invoke(*args, error=None):
        status = cli.main(["--data-dir", str(root), *map(str, args)])
        response = json.loads(capsys.readouterr().out)
        assert response["schema_version"] == 1
        assert response["ok"] == (status == 0)
        if error:
            assert response["error"]["code"] == error
            return response["error"]
        assert status == 0, response
        return response["result"]

    return root, invoke


@pytest.fixture
def sprite(tmp_path):
    path = tmp_path / "Nave 龍.png"
    image = Image.new("RGBA", (40, 24))
    ImageDraw.Draw(image).rectangle((5, 4, 35, 20), fill=(100, 120, 140, 210))
    image.save(path)
    return path


def import_job(invoke, sprite, grid=(2, 2)):
    source = invoke("source", "import", sprite)
    return source, invoke("job", "create", source["source_id"], "--grid", *grid)


def seed_tile(store, job, cell, *, warning=None):
    attempt = f"fixture-{cell.column}-{cell.row}"
    store.begin_tile(job, cell, attempt)
    Image.new("RGB", (FRAME, FRAME), (112, 131, 147)).save(job.folder / f"{attempt}.png")
    store.finish_tile(job, cell, attempt, f"gen-{attempt}", None, warning or {})


def test_discover_import_plan_and_saved_jobs(command, sprite, tmp_path):
    root, invoke = command
    capabilities = {item["command"]: item for item in invoke("capabilities")["commands"]}
    assert set(capabilities[""]["subcommands"]) == {
        "capabilities", "doctor", "config", "source", "plan", "job", "prompt", "generate",
        "batch", "tile", "inspect", "export", "operation",
    }
    assert capabilities["prompt set"]["mutually_exclusive"] == [
        {"required": True, "arguments": ["text", "file", "stdin"]},
    ]
    assert invoke("source", "list") == {"sources": []}
    planned = invoke("plan", "--file", sprite, "--grid", 16, 6)["layout"]
    assert (planned["width"], planned["height"], planned["edits"]) == (8960, 5376, 96)
    assert not root.exists()
    source, job = import_job(invoke, sprite, (4, 4))
    sid, jid = source["source_id"], job["job_id"]
    assert invoke("source", "show", sid)["name"] == sprite.name
    for extension in ("jpg", "webp"):
        image = tmp_path / f"photo.{extension}"
        Image.new("RGB", (8, 4)).save(image)
        assert invoke("source", "import", image)["width"] == 8
    assert len(invoke("source", "list")["sources"]) == 3
    assert len(invoke("job", "show", jid)["tiles"]) == 16
    assert invoke("job", "create", sid, "--grid", 4, 4)["job_id"] == jid
    assert invoke("job", "create", sid, "--grid", 2, 2)["job_id"] != jid
    assert len(invoke("job", "list", "--source", sid)["jobs"]) == 2
    assert invoke("job", "show", "missing", error="NOT_FOUND")
    assert invoke("job", "create", sid, "--grid", 17, 6, error="INVALID_ARGUMENT")
    assert invoke("plan", sid, "--scales", "2x,4x", "--input-mode", "previous")["stages"][1]["input_scale"] == 2
    assert invoke("plan", sid, "--scales", 4, 2, "--input-mode", "previous", error="INVALID_ARGUMENT")

    large = tmp_path / "large.png"
    Image.new("RGB", (1850, 100)).save(large)
    layout = invoke("plan", "--file", large, "--grid", 2, 2)
    assert not layout["layout"]["enlarges"] and layout["suggested"]["enlarges"]


def test_prompts_settings_dry_run_and_doctor(command, sprite, tmp_path, monkeypatch):
    _, invoke = command
    source, job = import_job(invoke, sprite, (4, 4))
    jid = job["job_id"]
    settings = invoke("config", "show")
    assert settings["codex"] == "auto" and settings["subagents"] is False
    assert invoke("config", "set", "subagents", "true")["subagents"] is True
    assert invoke("config", "set", "scales", "2x, 4x")["scales"] == [2, 4]
    assert invoke("config", "set", "input_mode", "previous")["input_mode"] == "previous"
    fake_exe = tmp_path / "codex.exe"
    fake_exe.touch()
    assert invoke("config", "set", "codex", fake_exe)["codex"] == str(fake_exe)
    assert invoke("doctor")["authenticated"] is None
    monkeypatch.setattr(cli, "CodexClient", lambda *_: pytest.MonkeyPatch.context())
    assert invoke("doctor", "--check-login")["authenticated"] is True

    def no_login(*_):
        raise AuthRequired("Sign in first.")

    monkeypatch.setattr(cli, "CodexClient", no_login)
    assert invoke("doctor", "--check-login", error="AUTH_REQUIRED")
    assert invoke("config", "reset", "codex")["codex"] == "auto"
    assert invoke("config", "reset") == settings
    plan = invoke("generate", jid, "--preview", "--anchor", 4, 4, "--dry-run")
    assert plan["max_edits"] == 4 and {tile["column"] for tile in plan["selected"]} == {3, 4}
    assert invoke("generate", jid, "--tile", 1, 1, "--tile", 1, 1, "--regenerate", "--dry-run")["max_edits"] == 1
    assert invoke("generate", jid, "--anchor", 4, 4, "--dry-run", error="INVALID_ARGUMENT")
    assert invoke("generate", jid, "--regenerate", error="INVALID_ARGUMENT")
    assert invoke("generate", jid, "--tile", 0, 2, error="INVALID_ARGUMENT")
    assert invoke("operation", "list") == {"operations": []}
    assert invoke("prompt", "set", jid, "--text", "Brushed copper")["global_prompt"] == "Brushed copper"
    notes = tmp_path / "prompt.txt"
    notes.write_text("Gravüre 龍", encoding="utf-8-sig")
    assert invoke("prompt", "set", jid, "--tile", 1, 1, "--file", notes)["tile_prompt"] == "Gravüre 龍"
    references = invoke("tile", "references", jid, 1, 1, "--output-dir", tmp_path / "refs")
    assert len(references["images"]) == 2
    assert references["prompt"].index("Brushed copper") < references["prompt"].index("Gravüre 龍")
    assert Path(references["prompt_path"]).read_text(encoding="utf-8") == references["prompt"]
    assert invoke("tile", "references", jid, 1, 1, "--output-dir", tmp_path / "refs", error="OUTPUT_EXISTS")
    monkeypatch.setattr(sys, "stdin", io.StringIO("Rivets"))
    assert invoke("prompt", "set", jid, "--tile", 1, 1, "--stdin")["tile_prompt"] == "Rivets"
    assert invoke("prompt", "clear", jid, "--tile", 1, 1)["tile_prompt"] == ""
    assert invoke("prompt", "get", jid, "--tile", 1, 1)["precedence"] == ["tile", "global"]
    assert invoke("prompt", "clear", jid)["global_prompt"] == ""
    assert invoke("plan", source["source_id"])["suggested"]["enlarges"]


def test_inspection_seams_stale_and_workspace_lock(command, sprite, tmp_path):
    root, invoke = command
    source, _ = import_job(invoke, sprite)
    store = Store(root)
    job = store.ensure_job(store.list_sources(source["source_id"])[0], Grid(2, 2), 2)
    first, second, *_ = job.layout.cells
    seed_tile(store, job, first)
    seed_tile(store, job, second, warning={first: "The contour is displaced."})
    result = invoke("job", "show", job.id)
    assert result["seams"][0]["warning"] == "The contour is displaced."
    assert invoke("tile", "show", job.id, 1, 1)["generation_id"] == "gen-fixture-0-0"
    repair = invoke("tile", "references", job.id, 1, 1, "--output-dir", tmp_path / "repair")
    assert len(repair["images"]) == 3
    invoke("prompt", "set", job.id, "--tile", 1, 1, "--text", "Rust")
    destination = tmp_path / "compare.png"
    view = invoke("inspect", job.id, "--output", destination, "--viewport", 80, 48, "--zoom", 100, "--grid")
    assert (view["width"], view["height"]) == (160, 48)
    assert view["stale_tiles"] == [{"column": 1, "row": 1}]
    assert len(view["missing_tiles"]) == 2
    assert invoke("export", job.id, "--format", "png", "--output", tmp_path / "full.png", error="NOT_READY")
    for focus in (("--tile", 1, 1), ("--region", 2, 3, 18, 13), ("--seam", 2, 1, 1, 1)):
        assert invoke("inspect", job.id, "--view", "input", "--output", destination, "--overwrite", *focus)["path"]
    crop = invoke("inspect", job.id, "--output", destination, "--overwrite", "--region", 2, 3, 18, 13, "--zoom", 100)
    assert crop["region"] == {"x": 2, "y": 3, "width": 18, "height": 13}
    assert crop["panel_size"] == {"width": 18, "height": 13}
    crop = invoke("inspect", job.id, "--output", destination, "--overwrite", "--region", 2, 3, 18, 13,
                  "--zoom", 100, "--viewport", 8, 6, "--center", 70, 40)
    assert crop["region"] == {"x": 12, "y": 10, "width": 8, "height": 6}
    assert invoke("inspect", job.id, "--output", destination, "--overwrite", "--zoom", "1e-308", error="INVALID_ARGUMENT")
    assert invoke("inspect", job.id, "--output", destination, "--overwrite", "--zoom", 100,
                  "--center", "inf", 0, error="INVALID_ARGUMENT")
    assert invoke("inspect", job.id, "--output", destination, "--overwrite", "--viewport", 1, 1, "--grid")["height"] == 1
    with exclusive_workspace(root):
        assert invoke("job", "show", job.id)["ready"] == 1
        assert invoke("prompt", "clear", job.id, error="WORKSPACE_BUSY")


@pytest.fixture
def controlled_worker(command, monkeypatch):
    root, invoke = command
    ops = Operations(root)
    queued = []
    monkeypatch.setattr(Operations, "launch", lambda _self, operation_id, token: queued.append((operation_id, token)))

    def run_next(run=None):
        ops.worker(*queued.pop(0), run or (lambda *args: cli.run_operation(root, *args)))

    return ops, invoke, run_next


def test_pause_cancel_interrupt_and_resume(controlled_worker, sprite):
    ops, invoke, run_next = controlled_worker
    _, job = import_job(invoke, sprite)
    item = invoke("generate", job["job_id"], "--preview", "--request-id", "retryable")
    oid = item["operation_id"]
    assert invoke("generate", job["job_id"], "--preview", "--request-id", "retryable")["operation_id"] == oid
    assert invoke("generate", job["job_id"], "--tile", 1, 1, "--request-id", "retryable", error="REQUEST_ID_CONFLICT")
    assert invoke("operation", "wait", oid, "--timeout", 0)["timed_out"]
    for action, state in (("pause", "paused"), ("cancel", "canceled")):
        running = Event()

        def work(_item, pause, cancel, report):
            report(active_tile={"column": 1, "row": 1})
            running.set()
            assert (pause if action == "pause" else cancel).wait(5)
            return jobs.RunOutcome(state)

        worker = Thread(target=run_next, args=(work,))
        worker.start()
        try:
            assert running.wait(5)
            assert invoke("operation", action, oid)["requested_action"] == action
        finally:
            worker.join(10)
        assert not worker.is_alive()
        result = invoke("operation", "wait", oid)
        assert result["state"] == state and result["progress"]["active_tile"] is None
        assert invoke("operation", "resume", oid)["operation_id"] == oid

    item = ops.get(oid)
    ops.change(oid, item["token"], state="running")
    assert invoke("operation", "status", oid)["state"] == "interrupted"
    run_next()  # A superseded worker cannot claim the operation.
    invoke("operation", "resume", oid)
    run_next(lambda *_: jobs.RunOutcome())
    assert invoke("operation", "list")["operations"][0]["state"] == "completed"


def test_wait_reports_completion_after_workspace_release(controlled_worker, sprite, monkeypatch):
    _, invoke, run_next = controlled_worker
    _, job = import_job(invoke, sprite)
    operation = invoke("generate", job["job_id"], "--preview")["operation_id"]
    releasing, release = Event(), Event()

    @contextmanager
    def delayed_release(root, *, timeout=0):
        with exclusive_workspace(root, timeout=timeout):
            yield
            if timeout:
                releasing.set()
                assert release.wait(5)

    monkeypatch.setattr(automation, "exclusive_workspace", delayed_release)
    worker = Thread(target=run_next, args=(lambda *_: jobs.RunOutcome(),))
    worker.start()
    try:
        assert releasing.wait(5)
        result = invoke("operation", "wait", operation, "--timeout", 0)
        assert result["state"] == "running" and result["timed_out"]
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    result = invoke("operation", "wait", operation)
    assert result["state"] == "completed" and not result["timed_out"]
    assert invoke("generate", job["job_id"], "--preview")["operation_id"] != operation


def test_recovery_preserves_concurrent_completion(controlled_worker, sprite, monkeypatch):
    ops, invoke, run_next = controlled_worker
    _, job = import_job(invoke, sprite)
    operation = invoke("generate", job["job_id"], "--preview")["operation_id"]
    unlocked, recovering, commit = Event(), Event(), Event()
    change = ops.change

    def delay_change(*args, before_commit=None, **kwargs):
        def delay_commit():
            before_commit()
            unlocked.set()
            assert commit.wait(5)

        if kwargs.get("state") == "interrupted":
            recovering.set()
        change(*args, before_commit=delay_commit if before_commit else None, **kwargs)

    monkeypatch.setattr(ops, "change", delay_change)
    worker = Thread(target=run_next, args=(lambda *_: jobs.RunOutcome(),))
    recovery = Thread(target=ops.recover)
    worker.start()
    try:
        assert unlocked.wait(5)
        recovery.start()
        assert recovering.wait(5)
    finally:
        commit.set()
        worker.join(10)
        if recovery.ident is not None:
            recovery.join(10)
    assert not (worker.is_alive() or recovery.is_alive())
    assert invoke("operation", "status", operation)["state"] == "completed"


def test_large_inspection_resamples_across_stripes(command):
    root, _ = command
    store = Store(root)
    source = store.import_source(Image.new("RGB", (320, 320)), "pattern.png")
    job = store.ensure_job(source, Grid(2, 2))
    cell = job.layout.cells[0]
    frame = job.layout.frame(cell)
    image = Image.new("RGB", (FRAME, FRAME), "black")
    ImageDraw.Draw(image).rectangle((0, STRIPE_HEIGHT - frame.top, FRAME - 1, FRAME - 1), fill="white")
    store.begin_tile(job, cell, "pattern")
    image.save(job.folder / "pattern.png")
    store.finish_tile(job, cell, "pattern", None, None, {})
    snapshot = store.snapshot(job)
    area = Rect(0, 0, 1600, 1600)
    size = (2873, 2873)
    reference = Image.fromarray(Compositor(job.layout, snapshot.artifacts).render(area)).convert("RGBA")
    expected = reference.resize(size, Image.Resampling.LANCZOS)
    rendered = render_reconstruction(job, snapshot, None, area, size, fit=False)
    assert rendered.tobytes() == expected.tobytes()


def test_refresh_previews_recovers_from_failed_tile_commit(controlled_worker, monkeypatch, tmp_path):
    ops, invoke, run_next = controlled_worker
    store = Store(ops.root)
    original = Image.new("RGB", (320, 320), (120, 120, 120))
    source = store.import_source(original, "source.png")
    job = store.ensure_job(source, Grid(2, 2))
    first, second, *_ = job.layout.cells
    seed_tile(store, job, first)
    jobs.refresh_thumbnails(store, job)
    thumbnail = store.snapshot(job).tiles[first].thumbnail
    original_preview = thumbnail.read_bytes()
    generated = Image.new("RGB", (FRAME, FRAME), (220, 220, 220))
    stage = tmp_path / "staged.png"
    generated.save(stage)
    frame = prepare_frame(np.asarray(original), job.layout, second)
    store.begin_tile(job, second, "next")

    def failed(*args, **kwargs):
        raise OSError("Simulated write failure.")

    with monkeypatch.context() as patch:
        patch.setattr(store, "finish_tile", failed)
        with pytest.raises(OSError, match="Simulated write failure"):
            jobs._commit_tile(store, job, second, "next", stage, frame, generated, None, "imagegen")
    assert store.snapshot(job).tiles[second].artifact is None
    assert thumbnail.read_bytes() == original_preview

    with monkeypatch.context() as patch:
        patch.setattr(jobs, "refresh_thumbnails", failed)
        with pytest.raises(OSError, match="Simulated write failure"):
            jobs._commit_tile(store, job, second, "next", stage, frame, generated, None, "imagegen")
    pending = store.snapshot(job)
    assert pending.previews_dirty and set(pending.current_artifacts) == {first, second}
    assert pending.tiles[first].thumbnail is None
    operation = invoke("job", "refresh-previews", job.id)["operation_id"]
    run_next()
    assert invoke("operation", "status", operation)["state"] == "completed"
    refreshed = store.snapshot(job)
    assert not refreshed.previews_dirty and all(refreshed.tiles[cell].thumbnail for cell in (first, second))
    assert thumbnail.read_bytes() != original_preview
    correct_preview = thumbnail.read_bytes()
    thumbnail.write_bytes(b"damaged preview")
    invoke("job", "refresh-previews", job.id)
    run_next()
    assert thumbnail.read_bytes() == correct_preview


class ImageProvider:
    def __init__(self):
        image = io.BytesIO()
        Image.new("RGB", (FRAME, FRAME), (120, 140, 160)).save(image, format="PNG")
        self.image = image.getvalue()
        self.calls = []
        self.limit = None

    def __call__(self, *_args, **_options):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def generate(self, prompt, references):
        if self.limit is not None and len(self.calls) == self.limit:
            raise QuotaPaused(1_798_000_000)
        self.calls.append((prompt, len(references)))
        return GeneratedImage(self.image, f"simulated-{len(self.calls)}")

    def generate_delegated(self, requests, on_image):
        for index, (prompt, references) in enumerate(requests):
            on_image(index, self.generate(prompt, references))


@pytest.fixture
def provider(monkeypatch, tmp_path):
    executable = tmp_path / "simulated-codex.exe"
    executable.touch()
    monkeypatch.setattr(cli, "find_codex", lambda: executable)
    image_provider = ImageProvider()
    run_tiles = jobs.run_tiles

    def generate(*args, **kwargs):
        return run_tiles(*args, **kwargs, client_factory=image_provider)

    monkeypatch.setattr(cli, "run_tiles", generate)
    monkeypatch.setattr(jobs, "run_tiles", generate)
    return image_provider


def test_preview_quota_reuse_and_selected_regeneration(controlled_worker, sprite, provider, tmp_path):
    _, invoke, run_next = controlled_worker
    _, job = import_job(invoke, sprite)
    jid = job["job_id"]
    invoke("prompt", "set", jid, "--text", "Copper")
    invoke("prompt", "set", jid, "--tile", 1, 1, "--text", "Brass")
    provider.limit = 2
    oid = invoke("generate", jid, "--preview", "--subagents")["operation_id"]
    run_next()
    limited = invoke("operation", "status", oid)
    assert limited["state"] == "quota_limited" and limited["quota_reset"] == 1_798_000_000
    assert limited["error"]["code"] == "QUOTA_EXCEEDED"
    assert limited["progress"]["completed"] == 2
    assert invoke("generate", jid, "--dry-run")["max_edits"] == 2
    provider.limit = None
    invoke("operation", "resume", oid)
    run_next()
    assert invoke("operation", "status", oid)["state"] == "completed"
    assert invoke("generate", jid, "--dry-run")["max_edits"] == 0
    before = [tile["generation_id"] for tile in invoke("job", "show", jid)["tiles"]]
    request = invoke("tile", "references", jid, 1, 1, "--output-dir", tmp_path / "references")
    oid = invoke("generate", jid, "--tile", 1, 1, "--regenerate", "--no-subagents")["operation_id"]
    run_next()
    assert invoke("operation", "status", oid)["state"] == "completed"
    after = [tile["generation_id"] for tile in invoke("job", "show", jid)["tiles"]]
    assert before[0] != after[0] and before[1:] == after[1:]
    assert len(provider.calls) == 5 and provider.calls[-1] == (request["prompt"], len(request["images"]))
    assert request["prompt"].index("Copper") < request["prompt"].index("Brass")


def test_scale_batch_modes_and_resume(controlled_worker, sprite, provider, tmp_path):
    _, invoke, run_next = controlled_worker
    source, job = import_job(invoke, sprite)
    sid = source["source_id"]
    invoke("prompt", "set", job["job_id"], "--text", "Mottled steel")
    provider.limit = 5
    oid = invoke("batch", "start", sid, "--scales", 2, 4, "--input-mode", "previous", "--output-dir", tmp_path,
                 "--prompt-from-job", job["job_id"])["operation_id"]
    run_next()
    limited = invoke("operation", "status", oid)
    assert limited["state"] == "quota_limited"
    first, = limited["progress"]["outputs"]
    first_time = Path(first["path"]).stat().st_mtime_ns
    assert first["scale"] == 2 and first["input_path"] == source["path"]
    provider.limit = None
    invoke("operation", "resume", oid)
    run_next()
    done = invoke("operation", "status", oid)
    assert done["state"] == "completed", done
    first, second = done["progress"]["outputs"]
    assert Path(first["path"]).stat().st_mtime_ns == first_time
    assert second["scale"] == 4 and second["input_revision"] and second["input_path"] != source["path"]
    assert Path(second["input_path"]).read_bytes() == Path(first["path"]).read_bytes()
    assert len(provider.calls) == 8 and all("Mottled steel" in prompt for prompt, _ in provider.calls)
    with Image.open(second["path"]) as image:
        assert image.size == (160, 96) and image.mode == "RGBA"

    prompt_file = tmp_path / "batch prompt.txt"
    prompt_file.write_text("Gunmetal", encoding="utf-8")
    oid = invoke("batch", "start", sid, "--scales", 2, 4, "--input-mode", "original", "--output-dir", tmp_path,
                 "--prompt-file", prompt_file)["operation_id"]
    run_next()
    done = invoke("operation", "status", oid)
    assert done["state"] == "completed"
    stages = done["progress"]["outputs"]
    assert [stage["input_path"] for stage in stages] == [source["path"], source["path"]]
    assert stages[1]["job_id"] != second["job_id"]
    assert all("Gunmetal" in prompt for prompt, _ in provider.calls[8:])


@pytest.mark.parametrize(("mode", "damaged"), [("original", "export"), ("previous", "export"), ("previous", "cache")])
def test_batch_resume_restores_damaged_png(controlled_worker, sprite, provider, tmp_path, mode, damaged):
    ops, invoke, run_next = controlled_worker
    source = invoke("source", "import", sprite)
    provider.limit = 4
    oid = invoke("batch", "start", source["source_id"], "--scales", 2, 4, "--input-mode", mode,
                 "--output-dir", tmp_path)["operation_id"]
    run_next()
    stopped = invoke("operation", "status", oid)
    assert stopped["state"] == "quota_limited"
    first, = stopped["progress"]["outputs"]
    output = Path(first["path"])
    expected = output.read_bytes()
    if damaged == "cache":
        revision, _ = ops.get(oid)["progress"]["publications"]["2"]
        target = Store(ops.root).list_sources(source["source_id"])[0].scale_result_path(revision)
    else:
        target = output
    target.write_bytes(b"corrupt PNG")

    provider.limit = None
    invoke("operation", "resume", oid)
    run_next()
    finished = invoke("operation", "status", oid)
    assert finished["state"] == "completed", finished
    first, second = finished["progress"]["outputs"]
    assert target.read_bytes() == output.read_bytes() == expected
    if mode == "previous":
        assert Path(second["input_path"]).read_bytes() == expected
    assert len(provider.calls) == 8


def test_new_batch_verifies_shared_cache(controlled_worker, sprite, provider, tmp_path):
    _, invoke, run_next = controlled_worker
    source = invoke("source", "import", sprite)

    def run_batch():
        oid = invoke("batch", "start", source["source_id"], "--scales", 2, 4, "--input-mode", "previous",
                     "--output-dir", tmp_path)["operation_id"]
        run_next()
        result = invoke("operation", "status", oid)
        assert result["state"] == "completed", result
        return result["progress"]["outputs"]

    first, second = run_batch()
    expected = Path(first["path"]).read_bytes()
    cache = Path(second["input_path"])
    checksum = cache.with_suffix(".sha256")
    assert checksum.is_file()
    for has_checksum in (True, False):
        if not has_checksum:
            checksum.unlink()
        Image.new("RGB", (first["width"], first["height"]), "red").save(cache)
        rebuilt, following = run_batch()
        assert Path(rebuilt["path"]).read_bytes() == cache.read_bytes() == expected
        assert following["job_id"] == second["job_id"]
        assert checksum.is_file()
    assert len(provider.calls) == 8


def test_chained_batch_resume_hides_outdated_downstream_output(controlled_worker, sprite, provider, tmp_path):
    _, invoke, run_next = controlled_worker
    source = invoke("source", "import", sprite)
    provider.limit = 8
    oid = invoke("batch", "start", source["source_id"], "--scales", 2, 4, 5, "--input-mode", "previous",
                 "--output-dir", tmp_path)["operation_id"]
    run_next()
    stopped = invoke("operation", "status", oid)
    assert stopped["state"] == "quota_limited"
    first, second = stopped["progress"]["outputs"]
    invoke("prompt", "set", first["job_id"], "--text", "Rebuild this stage.")
    provider.limit = 12
    resumed = invoke("operation", "resume", oid)
    assert resumed["progress"]["outputs"] == []
    run_next()
    stopped = invoke("operation", "status", oid)
    assert stopped["state"] == "quota_limited"
    assert stopped["progress"]["scale"] == 4
    assert stopped["progress"]["job_id"] != second["job_id"]
    assert stopped["progress"]["completed_stages"] == 1
    assert [entry["scale"] for entry in stopped["progress"]["outputs"]] == [2]
    assert Path(second["path"]).is_file()


def test_detached_cli_exports_and_previews(tmp_path, sprite):
    root = tmp_path / "Données 龍"
    executable = os.environ.get("REGIMAGEN_CLI_TEST_EXE")
    prefix = [executable] if executable else [sys.executable, "-m", "regimagen.cli"]
    environment = os.environ.copy()
    if executable:
        environment["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)

    def invoke(*args, error=None):
        process = subprocess.run(
            [*prefix, "--data-dir", str(root), *map(str, args)], cwd=tmp_path, env=environment,
            capture_output=True, encoding="utf-8", timeout=45,
        )
        assert process.stderr == ""
        response = json.loads(process.stdout)
        if error:
            assert process.returncode != 0 and response["error"]["code"] == error
            return response["error"]
        assert process.returncode == 0 and response["ok"], response
        return response["result"]

    def wait(operation):
        result = invoke("operation", "wait", operation["operation_id"], "--timeout", 30)
        assert result["state"] == "completed", result
        return result["progress"]

    assert invoke("capabilities")["commands"]
    source = invoke("source", "import", sprite)
    store = Store(root)
    job = store.ensure_job(store.list_sources(source["source_id"])[0], Grid(2, 2), 2)
    first = job.layout.cells[0]
    seed_tile(store, job, first)
    with exclusive_workspace(root):
        assert invoke("job", "show", job.id)["ready"] == 1
        assert invoke("job", "refresh-previews", job.id, error="WORKSPACE_BUSY")
    references = invoke("job", "refresh-previews", job.id, "--request-id", "refresh")
    assert Path(wait(references)["outputs"][0]["path"]).is_file()
    archive = tmp_path / "pièces.zip"
    zip_export = invoke("export", job.id, "--format", "tiles-zip", "--output", archive, "--request-id", "partial")
    assert wait(zip_export)["outputs"][0]["manifest"] == "positions.json"
    assert invoke("export", job.id, "--format", "tiles-zip", "--output", archive,
                  "--request-id", "partial")["operation_id"] == zip_export["operation_id"]
    assert invoke("export", job.id, "--format", "tiles-zip", "--output", archive, error="OUTPUT_EXISTS")
    with ZipFile(archive) as parts:
        manifest = json.loads(parts.read("positions.json"))
        assert (manifest["width"], manifest["height"]) == (80, 48)
        entry, = manifest["tiles"]
        tile_data = parts.read(entry["file"])
    part = tmp_path / "tile.png"
    single = wait(invoke("export", job.id, "--format", "tile-png", "--tile", 1, 1, "--output", part))["outputs"][0]
    assert single["core"] == entry["core"]
    with Image.open(part) as image, Image.open(io.BytesIO(tile_data)) as expected:
        assert image.mode == "RGBA" and image.tobytes() == expected.tobytes()
    recovered = job.layout.cells[1]
    store.begin_tile(job, recovered, "received")
    Image.new("RGB", (FRAME, FRAME), (112, 131, 147)).save(job.folder / "received.image", format="PNG")
    operation = invoke("generate", job.id, "--tile", 2, 1, "--codex", tmp_path / "missing.exe")
    assert wait(operation)["completed"] == 1
    assert not (job.folder / "received.image").exists()
    for cell in job.layout.cells[2:]:
        seed_tile(store, job, cell)
    full = tmp_path / "resultado 龍.png"
    wait(invoke("export", job.id, "--format", "png", "--output", full))
    view = tmp_path / "pixels.png"
    invoke("inspect", job.id, "--view", "reconstruction", "--viewport", 80, 48, "--zoom", 100, "--output", view)
    with Image.open(full) as image, Image.open(view) as rendered:
        assert image.size == (80, 48) and image.mode == "RGBA" and image.info["srgb"] == 0
        assert image.getpixel((0, 0))[3] == 0 and image.tobytes() == rendered.tobytes()
    region = invoke("inspect", job.id, "--view", "reconstruction", "--zoom", 100, "--output", view,
                    "--overwrite", "--region", 12, 6, 24, 16)
    assert region["region"] == {"x": 12, "y": 6, "width": 24, "height": 16}
    with Image.open(full) as image, Image.open(view) as cropped:
        assert image.crop((12, 6, 36, 22)).tobytes() == cropped.tobytes()
    invoke("inspect", job.id, "--zoom", "1e-308", "--output", view, "--overwrite", error="INVALID_ARGUMENT")
