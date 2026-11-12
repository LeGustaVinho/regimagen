from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image, ImageDraw

from regimagen.codex import Cancelled, CodexError
from regimagen.geometry import Rect
from regimagen.remove_detection import _guide, detect_removal, segment_removal_click, segment_removal_lasso


def _description(label, bounds, *, uncertain=False):
    left, top, right, bottom = bounds
    return {
        "label": label,
        "view": "overview",
        "bounds": bounds,
        "foreground": [[(left + right) / 2, (top + bottom) / 2]],
        "background": [],
        "contours": [[[left, top], [right, top], [right, bottom], [left, bottom]]],
        "uncertain": uncertain,
    }


class _FakeAnalysis:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, _executable, _cwd, *, cancelled, image_generation):
        assert not image_generation and not cancelled.is_set()
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def analyze_images(self, _prompt, references, schema):
        self.calls += 1
        assert {"targets", "protected"} == set(schema["properties"])
        assert [path.name for path in references] == ["detail.png", "overview.png"]
        for path in references:
            with Image.open(path) as guide:
                assert max(guide.size) == 1024
                assert guide.mode == "RGB"
        return self.result


class _BoundsSegmenter:
    def segment(self, image, bounds, foreground, background):
        assert foreground
        result = np.zeros(image.shape[:2], dtype=np.uint8)
        result[bounds.slices] = 1
        return result


def test_detection_expands_targets_beyond_lasso_and_retains_review_roles():
    base = Image.new("RGB", (200, 150), (25, 75, 110))
    lasso = Image.new("L", base.size)
    draw = ImageDraw.Draw(lasso)
    draw.rectangle((86, 51, 101, 81), fill=255)
    draw.rectangle((146, 56, 153, 72), fill=255)
    first = _description("Lamp", [375, 240, 630, 690])
    shadow = _description("Lamp shadow", [580, 675, 735, 740])
    second = _description("Nearby vase", [690, 305, 845, 620], uncertain=True)
    distant = _description("Unselected corner", [0, 0, 100, 100])
    foreground = _description("Foreground rail", [465, 420, 484, 550])
    client = _FakeAnalysis({
        "targets": [{**first, "effects": [shadow]}, {**second, "effects": []}, {**distant, "effects": []}],
        "protected": [foreground],
    })

    targets = detect_removal(
        base, lasso, Path("codex"), cancelled=Event(), client_factory=client, segmenter_factory=_BoundsSegmenter,
    )

    assert client.calls == 1
    assert [target.label for target in targets] == ["Lamp", "Lamp shadow", "Nearby vase", "Foreground rail"]
    assert targets[0].origin == (75, 36)
    assert targets[0].mask.size == (51, 68)
    assert targets[0].mask.mode == "L"
    assert targets[1].parent == 0
    assert [target.selected for target in targets] == [True, True, False, False]
    assert [target.protected for target in targets] == [False, False, False, True]


def test_detection_ignores_targets_in_the_gap_between_lasso_components():
    base = Image.new("RGB", (200, 100), (26, 74, 105))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (5, 10, 25, 40))
    lasso.paste(255, (175, 10, 195, 40))
    client = _FakeAnalysis({
        "targets": [{**_description("Not selected", [450, 200, 550, 350]), "effects": []}],
        "protected": [],
    })

    assert not detect_removal(
        base, lasso, Path("codex"), cancelled=Event(), client_factory=client, segmenter_factory=_BoundsSegmenter,
    )


def test_missing_segmenter_uses_local_contour_refinement():
    base = Image.new("RGB", (100, 100), (15, 120, 140))
    base.paste((215, 70, 35), (32, 28, 68, 75))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (41, 40, 57, 64))
    client = _FakeAnalysis({
        "targets": [{**_description("Object", [270, 240, 730, 790]), "effects": []}],
        "protected": [],
    })

    def missing_model():
        raise RuntimeError("Missing model")

    (target,) = detect_removal(
        base, lasso, Path("codex"), cancelled=Event(), client_factory=client, segmenter_factory=missing_model,
    )
    assert target.selected
    assert target.origin == (32, 28)
    assert target.mask.size == (36, 47)
    assert target.mask.getextrema() == (255, 255)


def test_detection_guide_preserves_isolated_pixels_during_downsample():
    base = Image.new("RGB", (8192, 256), (40, 60, 70))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (8, 10, 256, 256))
    lasso.putpixel((8000, 67), 1)
    guide = _guide(base, lasso, Rect(0, 0, *base.size))

    assert guide.size == (1024, 32)
    assert guide.getpixel((1000, 8)) != (40, 60, 70)
    assert guide.getpixel((1, 1)) != (40, 60, 70)
    assert guide.getpixel((600, 5)) == (40, 60, 70)


def test_invalid_locations_are_reported_and_cancel_skips_analysis():
    base = Image.new("RGB", (100, 100))
    lasso = Image.new("L", base.size, 255)
    client = _FakeAnalysis({
        "targets": [{**_description("Invalid", [-5, 2, 900, 950]), "effects": []}],
        "protected": [],
    })
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        detect_removal(base, lasso, Path("codex"), cancelled=cancel, client_factory=client)
    assert client.calls == 0
    cancel.clear()
    with pytest.raises(CodexError, match="invalid object locations"):
        detect_removal(base, lasso, Path("codex"), cancelled=cancel, client_factory=client)


class _LocalSegmenter:
    def candidates(self, image, _foreground):
        red, green, _blue = np.moveaxis(image, 2, 0)
        yield red > green, 0.94


def test_click_and_tight_lasso_locate_same_object_locally():
    base = Image.new("RGB", (120, 100), (30, 80, 90))
    base.paste((220, 55, 25), (36, 24, 90, 68))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (29, 18, 96, 73))
    options = {"cancelled": Event(), "segmenter_factory": _LocalSegmenter}

    (clicked,) = segment_removal_click(base, (65.25, 43.75), **options)
    (selected,) = segment_removal_lasso(base, lasso, **options)

    assert clicked.selected and selected.selected
    assert clicked.origin == selected.origin == (36, 24)
    assert clicked.mask.size == selected.mask.size == (54, 44)
    assert clicked.mask.tobytes() == selected.mask.tobytes()


def test_lasso_with_unexplained_object_defers_even_with_a_confident_main_mask():
    base = Image.new("RGB", (460, 300), (124, 139, 153))
    base.paste((175, 55, 35), (70, 64, 271, 231))
    base.putpixel((303, 138), (75, 173, 57))
    lasso = Image.new("L", base.size)
    lasso.paste(255, (63, 58, 344, 236))
    options = {"cancelled": Event(), "segmenter_factory": _LocalSegmenter}
    (main,) = segment_removal_lasso(base, lasso, **options)
    assert main.origin == (70, 64)

    # This second object is inside the same continuous lasso, but the point
    # segmenter still returns only the large red object under the prompt.
    base.paste((75, 173, 57), (292, 110, 338, 162))
    assert segment_removal_lasso(base, lasso, **options) == ()


def test_local_detection_defers_ambiguous_point_and_disconnected_lasso():
    class Ambiguous:
        def candidates(self, image, _foreground):
            large = np.zeros(image.shape[:2], dtype=bool)
            small = large.copy()
            large[22:78, 25:91] = True
            small[37:66, 41:72] = True
            yield large, 0.92
            yield small, 0.89

    base = Image.new("RGB", (120, 100))
    options = {"cancelled": Event(), "segmenter_factory": Ambiguous}
    assert segment_removal_click(base, (52, 50), **options) == ()
    mask = Image.new("L", base.size)
    mask.paste(255, (2, 7, 24, 40))
    mask.paste(255, (66, 24, 100, 66))
    assert segment_removal_lasso(base, mask, **options) == ()

    tight = Image.new("L", base.size)
    tight.paste(255, (38, 35, 76, 67))
    (selected,) = segment_removal_lasso(base, tight, **options)
    assert selected.origin == (41, 37)
    assert selected.mask.size == (31, 29)


def test_local_detection_requires_agreement_for_lower_confidence():
    class OneProposal:
        def candidates(self, image, _foreground):
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[22:78, 25:91] = True
            yield mask, 0.61

    class MatchingProposals(OneProposal):
        def candidates(self, image, foreground):
            yield from super().candidates(image, foreground)
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[23:79, 24:90] = True
            yield mask, 0.55

    base = Image.new("RGB", (120, 100))
    point = (52, 50)
    assert segment_removal_click(base, point, cancelled=Event(), segmenter_factory=OneProposal) == ()
    (result,) = segment_removal_click(base, point, cancelled=Event(), segmenter_factory=MatchingProposals)
    assert result.origin == (25, 22)


def test_local_detection_defers_potential_translucent_shadow():
    base = Image.new("RGBA", (120, 100))
    base.paste((220, 55, 25, 255), (36, 24, 90, 68))
    options = {"cancelled": Event(), "segmenter_factory": _LocalSegmenter}
    base.paste((210, 51, 22, 100), (36, 68, 90, 69))
    assert segment_removal_click(base, (65, 43), **options)

    base.paste((15, 15, 15, 100), (36, 71, 90, 74))
    assert segment_removal_click(base, (65, 43), **options) == ()


def test_local_detection_drops_specks_but_defers_separate_regions():
    base = Image.new("RGB", (120, 100), (30, 80, 90))
    base.paste((220, 55, 25), (36, 24, 90, 68))
    base.putpixel((2, 2), (220, 55, 25))
    options = {"cancelled": Event(), "segmenter_factory": _LocalSegmenter}
    (result,) = segment_removal_click(base, (65, 43), **options)
    assert result.origin == (36, 24)
    assert result.mask.size == (54, 44)

    base.paste((220, 55, 25), (2, 2, 18, 14))
    assert segment_removal_click(base, (65, 43), **options) == ()


def test_local_detection_skips_transparent_click_and_propagates_cancel():
    class CancelAfterStart:
        def candidates(self, image, _foreground):
            cancel.set()
            yield np.ones(image.shape[:2], dtype=bool), 0.95

    def should_not_load():
        pytest.fail("Transparent or out-of-bounds clicks should not load the segmenter")

    base = Image.new("RGBA", (120, 100))
    base.paste((210, 85, 0, 255), (10, 10, 22, 22))
    cancel = Event()
    options = {"cancelled": cancel, "segmenter_factory": should_not_load}
    assert segment_removal_click(base, (48, 30), **options) == ()
    assert segment_removal_click(base, (-1, 30), **options) == ()
    with pytest.raises(Cancelled):
        segment_removal_click(base, (15, 15), cancelled=cancel, segmenter_factory=CancelAfterStart)
