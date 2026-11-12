import hashlib
import sqlite3

import pytest
from PIL import Image

from regimagen.codex import Cancelled
from regimagen.geometry import Grid
from regimagen.storage import Store


def test_fill_versions_reopen_and_reconstruct_from_their_own_source(tmp_path):
    store = Store(tmp_path)
    base_image = Image.new("RGBA", (64, 48), (15, 25, 35, 0))
    base = store.import_source(base_image, "sprite.png")
    base_hash = hashlib.sha256(base.path.read_bytes()).hexdigest()
    old_job = store.ensure_job(base, Grid(2, 2))
    image = base_image.copy()
    image.paste((180, 95, 30, 220), (10, 10, 30, 30))
    mask = Image.new("L", image.size)
    mask.paste(255, (10, 10, 30, 30))
    candidate = Image.new("RGBA", (96, 96), (180, 95, 30, 255))
    parameters = {"crop": [-16, -24, 96], "shift": [2, -1], "color": [1, -2, 0]}
    first = store.save_fill(
        base, image, mask, candidate, prompt="Copper antenna", edit_alpha=True,
        metadata=parameters, base_sha256=base_hash,
    )
    mask.paste(0, (10, 10, 20, 20))
    image.paste(base_image.crop((10, 10, 20, 20)), (10, 10))
    second = store.save_fill(
        base, image, mask, candidate, prompt="Copper antenna", edit_alpha=True,
        metadata=parameters, base_sha256=base_hash, derived_from=first.id,
    )
    reopened = Store(tmp_path)
    assert reopened.fill_sources() == [second, first, base]
    assert reopened.find_job(base, Grid(2, 2)) == old_job
    assert reopened.find_job(second, Grid(2, 2)) is None
    new_job = reopened.ensure_job(second, Grid(2, 2), 2)
    assert new_job.source == second and new_job.input_path == second.path
    assert new_job.layout.size == (128, 96)
    edit = reopened.get_fill(second)
    assert edit.base == base and edit.base_sha256 == base_hash and edit.derived_from == first.id
    assert edit.prompt == "Copper antenna" and edit.edit_alpha and edit.metadata == parameters
    assert edit.candidate_path != reopened.get_fill(first).candidate_path
    for path, expected in [(second.path, image), (edit.mask_path, mask), (edit.candidate_path, candidate)]:
        with Image.open(path) as saved:
            assert saved.mode == expected.mode and saved.tobytes() == expected.tobytes()
    third = reopened.save_fill(
        second, image, mask, candidate, prompt="Copper antenna", edit_alpha=True,
        metadata=parameters, base_sha256=hashlib.sha256(second.path.read_bytes()).hexdigest(),
    )
    assert reopened.get_fill(third).base == second
    assert reopened.fill_sources() == [third, second, first, base]
    assert third.name == "sprite — Fill 3.png"
    assert hashlib.sha256(base.path.read_bytes()).hexdigest() == base_hash


def test_cancelled_fill_never_appears_in_history(tmp_path):
    store = Store(tmp_path)
    image = Image.new("RGB", (64, 48))
    base = store.import_source(image, "image.png")
    mask = Image.new("L", image.size, 255)
    candidate = Image.new("RGB", (96, 96))

    def cancel_when_files_are_written():
        return any(store.sources.glob("*-fill-candidate.png"))

    with pytest.raises(Cancelled):
        store.save_fill(
            base, image, mask, candidate, prompt="", edit_alpha=False, metadata={},
            base_sha256=hashlib.sha256(base.path.read_bytes()).hexdigest(),
            cancelled=cancel_when_files_are_written,
        )
    assert Store(tmp_path).list_sources() == [base]
    assert Store(tmp_path).fill_sources() == []
    assert list(store.sources.iterdir()) == [base.path]


def test_fill_rejects_changed_base_and_supports_legacy_workspace(tmp_path):
    with sqlite3.connect(tmp_path / "projetos.sqlite3") as db:
        db.execute("CREATE TABLE sources (id TEXT PRIMARY KEY, name TEXT, width INT, height INT)")
    assert Store(tmp_path, readonly=True).fill_sources() == []
    store = Store(tmp_path)
    image = Image.new("RGB", (64, 48))
    base = store.import_source(image, "image.png")
    base_hash = hashlib.sha256(base.path.read_bytes()).hexdigest()
    image.putpixel((0, 0), (255, 255, 255))
    image.save(base.path)
    with pytest.raises(ValueError, match="base has changed"):
        store.save_fill(
            base, image, Image.new("L", image.size, 255), Image.new("RGB", (96, 96)),
            prompt="", edit_alpha=False, metadata={}, base_sha256=base_hash,
        )
    assert store.list_sources() == [base] and store.fill_sources() == []
    assert list(store.sources.iterdir()) == [base.path]
