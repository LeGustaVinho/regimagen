from pathlib import Path
from threading import Event

import pytest
from PIL import Image

from regimagen.codex import Cancelled, CodexError
from regimagen.geometry import Rect
from regimagen.remove_detection import RemovalTarget, _guide, detect_removal, detect_removal_effects


def _outline(label, bounds, *, view="overview", uncertain=False):
    left, top, right, bottom = bounds
    return {
        "label": label,
        "view": view,
        "contours": [[[left, top], [right, top], [right, bottom], [left, bottom]]],
        "uncertain": uncertain,
    }


class _FakeAnalysis:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.prompt = ""

    def __call__(self, _executable, _cwd, *, cancelled, image_generation):
        assert not image_generation and not cancelled.is_set()
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def analyze_images(self, prompt, references, schema, *, effort=None):
        self.calls += 1
        self.prompt = prompt
        effect_scan = set(schema["properties"]) == {"effects"}
        names = ["detail.png", "object-outline.png"] if effect_scan else ["detail.png", "mark.png", "overview.png"]
        if effect_scan and len(references) == 3:
            names.append("possible-effects.png")
        assert [path.name for path in references] == names
        with Image.open(references[0]) as clean, Image.open(references[1]) as marked:
            assert max(clean.size) == 1024 and clean.mode == "RGB"
            assert clean.size == marked.size and clean.tobytes() != marked.tobytes()
        return self.result


def test_codex_selection_keeps_the_returned_contours_and_review_roles():
    # An untextured source cannot produce these shapes from pixel boundaries.
    base = Image.new("RGB", (200, 150), (25, 75, 110))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (86, 51, 101, 81))
    lasso.paste(255, (146, 56, 153, 72))
    subject = _outline("Whole object", [375, 240, 630, 690])
    shadow = _outline("Cast shadow", [580, 675, 735, 740])
    nearby = _outline("Nearby vase", [690, 305, 845, 620], uncertain=True)
    distant = _outline("Distant object", [0, 0, 100, 100])
    foreground = _outline("Foreground rail", [465, 420, 484, 550])
    client = _FakeAnalysis({
        "targets": [{**subject, "effects": [shadow]}, {**nearby, "effects": []}, {**distant, "effects": []}],
        "protected": [foreground],
    })

    targets = detect_removal(base, lasso, Path("codex"), cancelled=Event(), client_factory=client)

    assert client.calls == 1 and "The user used a lasso." in client.prompt
    assert [target.label for target in targets] == ["Whole object", "Cast shadow", "Nearby vase", "Foreground rail"]
    assert targets[0].origin == (75, 36)
    assert targets[0].mask.size == (52, 68)
    assert targets[0].mask.getextrema() == (255, 255)
    assert targets[1].parent == 0
    assert [target.selected for target in targets] == [True, True, False, False]
    assert [target.protected for target in targets] == [False, False, False, True]


def test_codex_selection_keeps_multipart_wire_and_the_adjacent_gap():
    base = Image.new("RGB", (100, 100), (34, 70, 110))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (20, 19, 62, 65))
    proposal = _outline("Hanging object", [240, 220, 550, 360], view="detail")
    proposal["contours"].extend(
        _outline("", bounds)["contours"][0]
        for bounds in ([240, 480, 550, 590], [340, 350, 360, 490])
    )
    client = _FakeAnalysis({"targets": [{**proposal, "effects": []}], "protected": []})

    (target,) = detect_removal(base, lasso, Path("codex"), cancelled=Event(), client_factory=client)

    x, y = target.origin
    assert target.selected and target.mask.getpixel((35 - x, 42 - y)) == 255
    assert target.mask.getpixel((28 - x, 42 - y)) == 0


def test_direct_lasso_requests_object_effect_and_protection_contours():
    base = Image.new("RGB", (100, 100), (34, 70, 110))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (20, 19, 62, 65))
    proposal = _outline("Lamp", [240, 220, 550, 360], view="detail")
    shadow = _outline("Shadow", [310, 380, 600, 450], view="detail")
    protected = _outline("Vase", [615, 500, 700, 560], view="detail")

    class DirectAnalysis(_FakeAnalysis):
        def analyze_images(self, prompt, references, schema, *, effort):
            assert effort == "medium" and "their visible effects, including outside it" in prompt
            assert "fine wire or cord continuously" in prompt and "the whole cast shadow" in prompt
            return super().analyze_images(prompt, references, schema)

    client = DirectAnalysis({"targets": [{**proposal, "effects": [shadow]}], "protected": [protected]})
    targets = detect_removal(base, lasso, Path("codex"), direct=True, cancelled=Event(), client_factory=client)
    assert [target.label for target in targets] == ["Lamp", "Shadow", "Vase"] and client.calls == 1
    assert targets[0].origin == (21, 20)
    assert targets[1].parent == 0 and targets[2].protected


def test_detection_ignores_contours_in_the_gap_between_lasso_components():
    base = Image.new("RGB", (200, 100), (26, 74, 105))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (5, 10, 25, 40))
    lasso.paste(255, (175, 10, 195, 40))
    client = _FakeAnalysis({
        "targets": [{**_outline("Unselected object", [450, 200, 550, 350]), "effects": []}],
        "protected": [],
    })

    assert not detect_removal(base, lasso, Path("codex"), cancelled=Event(), client_factory=client)


def test_click_uses_codex_and_clips_the_contour_to_visible_pixels():
    base = Image.new("RGBA", (100, 100))
    base.paste((200, 80, 10, 255), (24, 22, 56, 60))
    lasso = Image.new("L", base.size)
    object_outline = _outline("Clicked object", [200, 190, 620, 650], view="detail")
    client = _FakeAnalysis({"targets": [{**object_outline, "effects": []}], "protected": []})

    assert not detect_removal(base, lasso, Path("codex"), point=(10, 10), cancelled=Event(), client_factory=client)
    assert client.calls == 0
    (target,) = detect_removal(
        base, lasso, Path("codex"), point=(30, 31), cancelled=Event(), client_factory=client,
    )
    assert client.calls == 1 and "The user used a click." in client.prompt
    assert target.origin == (24, 22) and target.mask.size == (32, 38)


def test_effect_pass_rechecks_an_uncertain_shadow_beyond_the_selected_object():
    base = Image.new("RGB", (120, 100), (84, 112, 148))
    caster = RemovalTarget("Lamp", Image.new("L", (26, 32), 255), (24, 20))
    protected = RemovalTarget("Foreground rail", Image.new("L", (5, 15), 255), (60, 42), False, True)
    tentative = RemovalTarget("Possible shadow", Image.new("L", (12, 12), 255), (54, 50), False, parent=0)
    shadow = _outline("Cast shadow", [400, 490, 710, 720], view="detail")
    shading = _outline("Shading on the lamp", [250, 260, 330, 380], view="detail")
    client = _FakeAnalysis({"effects": [shadow, shading]})

    targets = detect_removal_effects(
        base, (caster, tentative, protected), Path("codex"), cancelled=Event(), client_factory=client,
    )
    assert client.calls == 1 and "Lamp" in client.prompt and targets[:2] == (caster, protected)
    effect = targets[2]
    assert effect.label == "Cast shadow" and effect.selected and effect.parent == 0
    assert effect.origin == (45, 49) and effect.mask.getpixel((22, 10)) == 255


def test_effect_pass_for_direct_lasso_rechecks_confirmed_effects_outside_the_lasso():
    base = Image.new("RGB", (200, 160), (84, 112, 148))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (30, 40, 90, 105))
    caster = RemovalTarget("Lamp", Image.new("L", (30, 40), 255), (34, 50))
    outside = RemovalTarget("Shadow outside", Image.new("L", (10, 10), 255), (97, 64), parent=0)
    inside = _outline("Cast shadow", [630, 530, 740, 690], view="detail")
    other = _outline("Farther cast shadow", [840, 530, 920, 690], view="detail")
    extension = _outline("Far edge of the shadow", [895, 575, 970, 735], view="detail")

    class Recheck(_FakeAnalysis):
        def analyze_images(self, prompt, references, schema, *, effort):
            if self.calls:
                assert "Include any missed extensions" in prompt and len(references) == 3
                self.result = {"effects": [extension]}
            return super().analyze_images(prompt, references, schema)

    client = Recheck({"effects": [inside, other]})

    targets = detect_removal_effects(
        base, (caster, outside), Path("codex"), cancelled=Event(), lasso=lasso, client_factory=client,
    )
    assert client.calls == 2 and targets[:2] == (caster, outside)
    assert [target.label for target in targets[2:]] == ["Cast shadow", "Farther cast shadow", "Far edge of the shadow"]
    assert targets[-1].origin[0] >= 90 and targets[-1].parent == 0


def test_direct_lasso_checks_for_offset_shadows_even_with_a_confirmed_contact_shadow():
    base = Image.new("RGB", (200, 160), (84, 112, 148))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (30, 40, 90, 105))
    caster = RemovalTarget("Lamp", Image.new("L", (30, 40), 255), (34, 50))
    contact = RemovalTarget("Contact shadow", Image.new("L", (10, 6), 255), (40, 90), parent=0)
    proposal = _outline("Possible pan shadow", [830, 530, 925, 690], view="detail", uncertain=True)
    revision = _outline("Pan shadow", [815, 500, 940, 720], view="detail")

    class Recheck(_FakeAnalysis):
        def analyze_images(self, prompt, references, schema, *, effort):
            assert effort == "high"
            if self.calls:
                assert "Image 3 marks areas" in prompt
                assert references[2].read_bytes() != references[0].read_bytes()
                self.result = {"effects": [revision]}
            return super().analyze_images(prompt, references, schema)

    client = Recheck({"effects": [proposal]})
    targets = detect_removal_effects(
        base, (caster, contact), Path("codex"), cancelled=Event(), lasso=lasso, client_factory=client,
    )
    assert client.calls == 2 and targets[:2] == (caster, contact)
    assert targets[2].label == "Pan shadow" and targets[2].selected and targets[2].parent == 0


def test_detection_guide_preserves_isolated_pixels_during_downsample():
    base = Image.new("RGB", (8192, 256), (40, 60, 70))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (8, 10, 256, 256))
    lasso.putpixel((8000, 67), 1)
    guide = _guide(base, lasso, Rect(0, 0, *base.size))
    assert guide.size == (1024, 32)
    assert guide.getpixel((1000, 8)) != (40, 60, 70)
    assert guide.getpixel((1, 1)) != (40, 60, 70)


def test_invalid_contours_are_reported_and_cancellation_skips_analysis():
    base = Image.new("RGB", (100, 100))
    lasso = Image.new("L", base.size, 255)
    client = _FakeAnalysis({
        "targets": [{**_outline("Invalid", [-5, 2, 900, 950]), "effects": []}],
        "protected": [],
    })
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        detect_removal(base, lasso, Path("codex"), cancelled=cancel, client_factory=client)
    assert client.calls == 0
    cancel.clear()
    with pytest.raises(CodexError, match="invalid selection"):
        detect_removal(base, lasso, Path("codex"), cancelled=cancel, client_factory=client)
