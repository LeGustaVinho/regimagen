import io
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image

from regimagen.codex import Cancelled, CodexError, GeneratedImage
from regimagen.fill import _match_pattern_context, _pattern_edge_weight
from regimagen.geometry import FRAME, Rect
from regimagen.pattern import (
    _alpha_seam_mask, _polish_join, _work_windows, check_pattern_joins, detect_pattern_seams, repair_pattern,
)
from regimagen.periodic import periodic_crop, wrap_offset
from regimagen.pattern_courses import match_course_widths


def test_periodic_crop_and_offset_preserve_rgba_with_odd_dimensions():
    pixels = np.arange(5 * 7 * 4, dtype=np.uint8).reshape(5, 7, 4)
    tile = Image.fromarray(pixels)
    area = Rect(-17, -4, 12, 24)
    expected = pixels[np.arange(area.top, area.bottom)[:, None] % 5, np.arange(area.left, area.right) % 7]
    np.testing.assert_array_equal(np.asarray(periodic_crop(tile, area)), expected)
    shifted = wrap_offset(tile, -9, 12)
    np.testing.assert_array_equal(np.asarray(shifted), np.roll(pixels, (12, -9), axis=(0, 1)))
    assert wrap_offset(shifted, 9, -12).tobytes() == tile.tobytes()


def test_detection_uses_offset_and_repetition_and_rejects_invalid_segments():
    original = Image.new("RGB", (31, 25), "#123456")
    result = {"segments": [
        {"axis": "vertical", "start": 0, "end": 400, "half_width": 70},
        {"axis": "horizontal", "start": 300, "end": 1000, "half_width": 40},
    ]}

    class Client:
        def __init__(self, *_args, image_generation, **_kwargs):
            assert not image_generation

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def analyze_images(self, _prompt, references, _schema, *, subject):
            assert subject == "seam detection"
            clean, marked, tiled = references
            with Image.open(clean) as ref:
                assert ref.size == original.size
            with Image.open(marked) as ref:
                assert ref.getpixel((15, 2)) == (255, 69, 214)
            with Image.open(tiled) as ref:
                assert ref.size == (93, 75)
                assert ref.getpixel((40, 26)) == original.getpixel((9, 1))
            return result

    def detect():
        return detect_pattern_seams(original, original, (15, 12), Path("codex"),
                                    cancelled=Event(), client_factory=Client)

    mask = detect()
    assert mask.getpixel((15, 8)) == mask.getpixel((28, 12)) == 255
    assert mask.getpixel((15, 19)) == mask.getpixel((4, 12)) == 0
    result["segments"][0]["half_width"] = float("nan")
    with pytest.raises(CodexError, match="invalid seam coordinates"):
        detect()
    result["segments"].clear()
    assert detect().getbbox() is None


@pytest.mark.parametrize("mode,size", [("RGB", (57, 41)), ("RGBA", (1401, 73))])
def test_repair_preserves_unselected_pixels_and_reuses_earlier_windows(mode, size):
    alpha = (190,) if mode == "RGBA" else ()
    expected = (90, 120, 150, *alpha)
    base = Image.new(mode, size, expected)
    shifted = base.copy()
    dx, dy = (extent // 2 for extent in size)
    mask = Image.new("L", size)
    mask.paste(255, (dx - 6, 0, dx + 6, size[1]))
    mask.paste(255, (0, dy - 6, size[0], dy + 6))
    shifted.putpixel((dx, dy), (8, 9, 11, *alpha))
    windows = tuple(_work_windows(size))
    anchor = (windows[1].left + 40, dy) if len(windows) > 1 else None
    if anchor is not None:
        shifted.putpixel(anchor, (8, 9, 11, *alpha))
    calls = []
    data = io.BytesIO()
    Image.new(mode, (FRAME, FRAME), expected).save(data, format="PNG")

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def generate(self, prompt, refs, *, verify_references):
            assert "Repair the marked discontinuities" in prompt
            assert "classify the requested transparency" not in prompt
            assert verify_references and len(refs) == 3
            if anchor is not None and len(calls) == 1:
                window = windows[1]
                extent = max(window.width, window.height)
                side = extent + 2 * min(128, max(8, extent // 4))
                left = window.left - (side - window.width) // 2
                top = window.top - (side - window.height) // 2
                # The previous window has already repaired the periodic copy of this pixel.
                with Image.open(refs[0]) as context:
                    scale = FRAME / side
                    x, y = anchor[0] - left, anchor[1] - top
                    pixel = context.getpixel((int((x + 0.5) * scale), int((y + 0.5) * scale)))
                    np.testing.assert_allclose(pixel, expected, atol=1)
            calls.append(refs)
            return GeneratedImage(data.getvalue(), "pattern", transparent_background=mode == "RGBA")

    progress = []
    fixed, _ = repair_pattern(shifted, mask, Path("codex"), cancelled=Event(),
                              progress=lambda a, b: progress.append((a, b)), client_factory=Client)
    assert fixed.tobytes() == base.tobytes()
    assert progress[-1] == (len(calls), len(calls))
    assert len(calls) == (1 if mode == "RGB" else 3)


def test_cancellation_discards_incomplete_repair():
    tile = Image.new("RGB", (700, 90), "#123456")
    mask = Image.new("L", tile.size)
    mask.paste(255, (0, 40, 700, 50))
    cancel = Event()

    def progress(*_):
        cancel.set()

    with pytest.raises(Cancelled):
        repair_pattern(tile, mask, Path("codex"), cancelled=cancel, progress=progress,
                       client_factory=lambda *_args, **_kwargs: pytest.fail("Started ImageGen after cancellation"))


def test_validation_checks_both_axes_and_ignores_hidden_rgb():
    tile = Image.new("RGBA", (32, 32))
    tile.putpixel((0, 0), (255, 255, 255, 0))
    assert check_pattern_joins(tile, cancelled=Event()) == ()
    tile.paste((20, 50, 65, 254), (0, 0, 10, 10))
    warnings = check_pattern_joins(tile, cancelled=Event())
    assert {(item.axis, item.start, item.end) for item in warnings} == {
        ("vertical", 0, 10), ("horizontal", 0, 10),
    }


def test_alpha_detection_includes_a_missed_cut_and_leaves_continuous_silhouettes_alone():
    original = Image.new("RGBA", (128, 94))
    original.paste((0, 100, 230, 230), (0, 30, 18, 46))
    for x0, x1 in ((0, 9), (119, 128)):
        original.paste((32, 83, 120, 185), (x0, 5, x1, 15))
    dx, dy = 64, 47
    mask = _alpha_seam_mask(wrap_offset(original, dx, dy), (dx, dy))
    assert mask.getpixel((dx - 12, 30 + dy)) == 255
    assert mask.getpixel((dx + 20, 35 + dy)) == 255
    assert mask.getpixel((dx, 10 + dy)) == 0

    across = Image.new("RGBA", original.size)
    across.paste((40, 70, 195, 255), (0, 46, 12, 50))
    wrapped = _alpha_seam_mask(wrap_offset(across, dx, dy), (dx, dy))
    assert wrapped.getpixel((dx, 0)) == wrapped.getpixel((dx, 93)) == 255


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_polish_repairs_a_residual_join_and_leaves_protected_pixels(mode):
    tile = Image.new(mode, (45, 36))
    left, right = ((100, 70, 30), (210, 110, 80)) if mode == "RGB" else ((0, 0, 0, 0), (14, 80, 60, 185))
    tile.paste(left, (0, 0, 22, 36))
    tile.paste(right, (22, 0, 45, 36))
    mask = Image.new("L", tile.size)
    mask.paste(255, (17, 0, 27, 33))
    original = np.array(tile)
    _polish_join(tile, mask, 0, 22, Event())
    after = np.asarray(tile).astype(np.int16)
    assert np.max(abs(after[10, 21] - after[10, 22])) < 24
    assert np.abs(after[10, 18] - after[10, 19]).max() > 0
    assert np.array_equal(after[np.asarray(mask) == 0], original[np.asarray(mask) == 0])


def test_alpha_feather_is_narrow_where_two_silhouettes_disagree():
    distance = np.full((25, 25), 12, np.float32)
    distance[12, 12] = 4
    base = np.zeros((25, 25, 4), np.uint8)
    alpha = np.ones((25, 25, 1), np.float32)
    assert _pattern_edge_weight(distance, base, alpha)[12, 12] == 1
    base[..., 3] = 255
    assert 0 < _pattern_edge_weight(distance, base, alpha)[12, 12] < 0.5


def test_pattern_color_matching_preserves_material_contrast():
    size = (128, 96)
    source = np.full((96, 128, 3), (161, 62, 39), dtype=np.uint8)
    source[np.arange(size[1]) % 24 < 5] = (170, 157, 139)
    generated = source.astype(np.int16)
    generated += np.where((np.arange(size[1]) % 24 < 5)[:, None, None],
                          np.array([-14, -19, -18]), np.array([19, -7, -8]))
    generated = generated.astype(np.uint8)
    selection = Image.new("L", size)
    selection.paste(255, (53, 0, 75, size[1]))
    matched = np.asarray(_match_pattern_context(Image.fromarray(source), selection,
                                               Image.fromarray(generated), (0, 0))).astype(np.int16)
    for y in (2, 11):
        assert abs(matched[y, 61] - source[y, 61]).mean() < abs(generated[y, 61].astype(int) - source[y, 61]).mean()
    assert np.mean(matched[2, 53:75, 1]) - np.mean(matched[11, 53:75, 1]) > 85


def test_pattern_geometry_alignment_continues_offset_courses():
    y, x = np.mgrid[:160, :160]
    rng = np.random.default_rng(13)
    courses = (y + (2 * np.sin(x / 19)).astype(int)) % 23 < 4
    source = np.where(courses[..., None], (177, 156, 130), (144, 52, 30))
    source = np.clip(source + rng.normal(0, 5, (160, 160, 1)), 0, 255).astype(np.uint8)
    patch = np.roll(source, 4, axis=0)
    selection = Image.new("L", (160, 160))
    selection.paste(255, (68, 0, 92, 160))
    corrected = np.asarray(_match_pattern_context(Image.fromarray(source), selection,
                                                  Image.fromarray(patch), (0, 0)))
    area = np.s_[12:-12, 68:74, 1]
    error = lambda pixels: np.mean(abs(pixels[area].astype(np.int16) - source[area]))
    assert error(corrected) < error(patch) * 0.6


@pytest.mark.parametrize("radius", (2, 6))
@pytest.mark.parametrize("transposed", (False, True))
def test_pattern_course_width_uses_untouched_joints(radius, transposed):
    y, x = np.mgrid[:240, :240]
    if transposed:
        y, x = x, y
    selected = (x >= 92) & (x < 148)
    distance = abs((y - 16) % 31 - 15)
    mortar = (185, 173, 159)
    brick = (168, 66, 39)
    source = np.uint8(np.where((distance <= 3)[..., None], mortar, brick))
    candidate = np.uint8(np.where((distance <= np.where(selected, radius, 3))[..., None], mortar, brick))
    corrected = match_course_widths(source, candidate, selected)

    def error(pixels):
        joints = pixels[..., 0].astype(np.int16) - pixels[..., 1] < 55
        return np.count_nonzero(joints != (distance <= 3))

    assert error(corrected) < error(candidate) * 0.25
    np.testing.assert_array_equal(corrected[~selected], candidate[~selected])


def test_pattern_course_width_does_not_warp_nonrepeating_texture():
    rng = np.random.default_rng(18)
    reference = rng.integers(0, 256, (180, 180, 3), dtype=np.uint8)
    selected = np.zeros(reference.shape[:2], dtype=bool)
    selected[:, 80:100] = True
    candidate = np.roll(reference, 2, axis=0)
    assert match_course_widths(reference, candidate, selected) is candidate


def test_pattern_course_width_uses_adjacent_courses_for_a_full_width_repair():
    y, _ = np.mgrid[:260, :220]
    distance = abs(y % 32 - 16)
    selected = (y >= 101) & (y < 124)
    mortar, brick = (185, 173, 159), (168, 66, 39)
    source = np.uint8(np.where((distance <= 3)[..., None], mortar, brick))
    candidate = np.uint8(np.where((distance <= np.where(selected, 6, 3))[..., None], mortar, brick))
    corrected = match_course_widths(source, candidate, selected)
    received = corrected[:, 110, 0] > 177
    assert np.count_nonzero(received & selected[:, 110]) == 7
    np.testing.assert_array_equal(corrected[~selected], candidate[~selected])
