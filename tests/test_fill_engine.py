import io
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from regimagen.codex import Cancelled, GeneratedImage
from regimagen.fill import SELECTION_MARK, FillCandidate, generate_fill, render_fill
from regimagen.geometry import FRAME


def _png(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class FillClient:
    def __init__(self, response, *, references_verified=True):
        self.response = response
        self.references_verified = references_verified
        self.calls = []

    def __call__(self, executable, directory, *, cancelled):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def generate(self, prompt, references, *, verify_references=False):
        assert verify_references
        source, selection, context = references
        with Image.open(source) as target, Image.open(selection) as guide, Image.open(context) as overview:
            self.calls.append((prompt, target.copy(), guide.copy(), overview.copy()))
        return GeneratedImage(self.response(*self.calls[-1]), "test-image", self.references_verified)


def _generate(base, mask, client, edit_alpha=False, *, cancelled=None):
    return generate_fill(base, mask, "Add a blue panel", edit_alpha, Path("codex"),
                         cancelled=cancelled if cancelled is not None else Event(), client_factory=client)


def _candidate(image, *, crop=None, feather=1):
    return FillCandidate(image, {
        "crop": crop or [0, 0, *image.size], "shift": [0, 0], "color": [0, 0, 0], "feather": feather,
    })


def test_frame_is_square_and_first_input_is_source_even_at_canvas_edge():
    base = Image.new("RGB", (140, 120), (55, 30, 25))
    mask = Image.new("L", base.size)
    ImageDraw.Draw(mask).rectangle((0, 35, 19, 64), fill=255)
    client = FillClient(lambda *_: _png(Image.new("RGB", (80, 80), "blue")))
    candidate = _generate(base, mask, client)
    prompt, reference, guide, context = client.calls[0]
    assert len(client.calls) == 1
    assert "Add a blue panel" in prompt
    assert reference.size == guide.size == (FRAME, FRAME)
    assert reference.mode == guide.mode == "RGB"
    assert context.mode == "RGB" and context.size == (FRAME, FRAME // 2)
    assert reference.getpixel((0, FRAME // 2)) == (55, 30, 25)
    assert guide.getpixel((0, FRAME // 2)) == SELECTION_MARK
    assert guide.getpixel((FRAME - 1, FRAME - 1)) == reference.getpixel((FRAME - 1, FRAME - 1))
    assert candidate.metadata["references_verified"] is True
    assert candidate.metadata["crop"] == [0, 19, 62, 81]
    assert candidate.metadata["shift"] == [0, 0]
    assert candidate.metadata["color"] == [0.0, 0.0, 0.0]

    preview = render_fill(base, mask, candidate, edit_alpha=False)
    assert preview.origin == (0, 35)
    assert preview.patch.size == (20, 30)
    result = np.asarray(preview.compose(base))
    outside = np.asarray(mask) == 0
    np.testing.assert_array_equal(result[outside], np.asarray(base)[outside])
    assert tuple(result[50, 6]) == (0, 0, 255)


def test_sprite_context_shows_neighborhood_and_global_subject():
    base = Image.new("RGBA", (500, 200), (231, 2, 97, 0))
    base.paste((17, 143, 70, 255), (94, 80, 110, 120))
    base.paste((15, 39, 200, 255), (300, 80, 320, 120))
    mask = Image.new("L", base.size)
    mask.paste(255, (60, 95, 70, 105))
    client = FillClient(lambda *_: _png(Image.new("RGBA", (80, 80), (0, 0, 0, 0))),
                        references_verified=None)
    candidate = _generate(base, mask, client, edit_alpha=True)
    _, reference, _, context = client.calls[0]
    assert reference.getchannel("A").getbbox() is None

    pixels = np.asarray(context)
    near, whole = np.split(pixels, 2, axis=1)
    for panel in (near, whole):
        assert np.any(np.all(panel == SELECTION_MARK, axis=2))
        assert np.any(np.all(panel == (17, 143, 70), axis=2))
        assert not np.any(np.all(panel == (231, 2, 97), axis=2))
    assert not np.any(np.all(near == (15, 39, 200), axis=2))
    assert np.any(np.all(whole == (15, 39, 200), axis=2))

    assert candidate.metadata["references_verified"] is None
    preview = render_fill(base, mask, candidate, edit_alpha=True)
    assert "Could not verify that ImageGen used the context references." in preview.warning


def test_selection_guide_retains_small_distant_region_in_virtual_crop():
    base = Image.new("RGB", (8192, 256), (28, 34, 42))
    mask = Image.new("L", base.size)
    mask.paste(255, (25, 25, 150, 150))
    mask.paste(255, (8000, 61, 8005, 66))
    client = FillClient(lambda *_: _png(Image.new("RGB", (80, 80), "blue")))
    _generate(base, mask, client)
    _, _, guide, _ = client.calls[0]

    left, right = np.split(np.asarray(guide), 2, axis=1)
    assert np.any(np.all(left == SELECTION_MARK, axis=2))
    assert np.any(np.all(right == SELECTION_MARK, axis=2))


def test_selection_mark_contrasts_with_magenta_art_in_all_guides():
    base = Image.new("RGB", (512, 512), SELECTION_MARK)
    mask = Image.new("L", base.size)
    mask.paste(255, (190, 190, 320, 320))
    client = FillClient(lambda *_: _png(Image.new("RGB", (80, 80), "blue")))
    _generate(base, mask, client)
    _, reference, guide, context = client.calls[0]

    assert reference.getextrema() == tuple((channel, channel) for channel in SELECTION_MARK)
    near, whole = np.split(np.asarray(context), 2, axis=1)
    for panel in (np.asarray(guide), near, whole):
        centers = panel[20:-20, 20:-20]
        assert np.any(np.max(np.abs(centers.astype(int) - SELECTION_MARK), axis=2) > 160)
        assert np.any(np.all(centers == SELECTION_MARK, axis=2))


def test_accepts_encoded_opaque_alpha_and_rejects_missing_alpha_or_wrong_shape():
    base = Image.new("RGBA", (64, 64))
    mask = Image.new("L", base.size, 255)
    opaque = FillClient(lambda *_: _png(Image.new("RGBA", (64, 64), (25, 30, 40, 255))))
    assert _generate(base, mask, opaque, edit_alpha=True).image.mode == "RGBA"
    assert "transparent_background=true" in opaque.calls[0][0]
    assert opaque.calls[0][1].getchannel("A").getextrema() == (0, 0)

    missing = FillClient(lambda *_: _png(Image.new("RGB", (64, 64))))
    with pytest.raises(ValueError, match="without transparency"):
        _generate(base, mask, missing, edit_alpha=True)
    malformed = FillClient(lambda *_: _png(Image.new("RGBA", (64, 65))))
    with pytest.raises(ValueError, match="non-square"):
        _generate(base, mask, malformed, edit_alpha=True)


def test_locked_transparent_selection_does_not_call_generator():
    base = Image.new("RGBA", (64, 64))
    mask = Image.new("L", base.size)
    mask.paste(255, (10, 10, 30, 30))
    base.putpixel((40, 40), (30, 50, 70, 255))
    client = FillClient(lambda *_: _png(Image.new("RGBA", (64, 64))))
    with pytest.raises(ValueError, match="turn on Edit Transparency"):
        _generate(base, mask, client)
    assert not client.calls

    base.putpixel((15, 15), (30, 50, 70, 1))
    mask.putpixel((15, 15), 1)
    _generate(base, mask, client)
    assert len(client.calls) == 1


def test_alpha_edit_adds_coverage_and_cuts_a_hole_while_lock_retains_base():
    base = Image.new("RGBA", (32, 32), (10, 25, 75, 0))
    ImageDraw.Draw(base).rectangle((12, 0, 31, 31), fill=(150, 20, 30, 255))
    generated = Image.new("RGBA", base.size, (0, 170, 240, 180))
    ImageDraw.Draw(generated).rectangle((14, 0, 31, 31), fill=(255, 0, 255, 0))
    mask = Image.new("L", base.size)
    ImageDraw.Draw(mask).rectangle((3, 5, 27, 26), fill=255)
    candidate = _candidate(generated, feather=2)
    result = np.asarray(render_fill(base, mask, candidate, edit_alpha=True).compose(base))
    original = np.asarray(base)
    outside = np.asarray(mask) == 0
    np.testing.assert_array_equal(result[outside], original[outside])
    assert tuple(result[12, 8]) == (0, 170, 240, 180)
    assert tuple(result[12, 23]) == (0, 0, 0, 0)

    locked = np.asarray(render_fill(base, mask, candidate, edit_alpha=False).compose(base))
    np.testing.assert_array_equal(locked[..., 3], original[..., 3])
    np.testing.assert_array_equal(locked[:, :12], original[:, :12])
    np.testing.assert_array_equal(locked[:, 14:], original[:, 14:])


def test_half_coverage_blends_in_linear_rgb_and_reduced_mask_restores_source():
    base = Image.new("RGB", (24, 24), "red")
    candidate = _candidate(Image.new("RGB", base.size, "blue"))
    mask = Image.new("L", base.size)
    ImageDraw.Draw(mask).rectangle((3, 4, 20, 19), fill=255)
    mask.putpixel((10, 10), 128)
    output = render_fill(base, mask, candidate, edit_alpha=False).compose(base)
    assert output.getpixel((10, 10)) == (187, 0, 188)

    reduced = mask.copy()
    ImageDraw.Draw(reduced).rectangle((3, 4, 10, 19), fill=0)
    first = render_fill(base, reduced, candidate, edit_alpha=False).compose(base)
    second = render_fill(base, reduced, candidate, edit_alpha=False).compose(base)
    assert first.getpixel((10, 10)) == base.getpixel((10, 10))
    assert first.getpixel((16, 12)) == output.getpixel((16, 12))
    assert first.tobytes() == second.tobytes()


def test_registers_matching_context_and_skips_transparent_context():
    rng = np.random.default_rng(1821)
    noise = rng.integers(25, 231, size=(1024, 1024, 3), dtype=np.uint8)
    base = Image.fromarray(noise)
    mask = Image.new("L", base.size)
    ImageDraw.Draw(mask).rectangle((320, 320, 703, 703), fill=255)

    def shifted(_prompt, frame, _guide, _context):
        pixels = cv2.warpAffine(np.asarray(frame), np.float32([[1, 0, 4], [0, 1, -3]]),
                                frame.size, borderMode=cv2.BORDER_REPLICATE)
        return _png(Image.fromarray(pixels))

    candidate = _generate(base, mask, FillClient(shifted))
    assert candidate.metadata["shift"] == [4, -3]

    alpha = mask
    base.putalpha(alpha)
    candidate = _generate(base, mask, FillClient(shifted), edit_alpha=True)
    assert candidate.metadata["shift"] == [0, 0]
    assert candidate.metadata["color"] == [0.0] * 3


def test_enlargement_warning_and_cancellation():
    base = Image.new("RGB", (90, 200), "red")
    mask = Image.new("L", base.size, 255)
    candidate = _candidate(Image.new("RGB", (20, 20), "blue"), crop=[0, 0, 200, 200])
    result = render_fill(base, mask, candidate, edit_alpha=False)
    assert "enlarged" in result.warning
    assert result.compose(base).getpixel((89, 199)) == (0, 0, 255)

    cancelled = Event()
    cancelled.set()
    client = FillClient(lambda *_: b"")
    with pytest.raises(Cancelled):
        _generate(base, mask, client, cancelled=cancelled)
    assert not client.calls


def test_reports_displaced_contour_through_the_feather():
    base = Image.new("RGB", (512, 512), (70, 70, 70))
    ImageDraw.Draw(base).line((0, 250, 511, 250), fill="white", width=12)
    mask = Image.new("L", base.size)
    ImageDraw.Draw(mask).ellipse((100, 80, 400, 440), fill=255)
    moved = Image.new("RGB", base.size, (70, 70, 70))
    ImageDraw.Draw(moved).line((0, 265, 511, 265), fill="white", width=12)

    aligned = render_fill(base, mask, _candidate(base, feather=12), edit_alpha=False)
    displaced = render_fill(base, mask, _candidate(moved, feather=12), edit_alpha=False)
    assert not aligned.warning
    assert "visible seam" in displaced.warning


def test_reports_new_alpha_seam_and_respects_locked_alpha():
    base = Image.new("RGBA", (512, 512), (126, 126, 126, 160))
    mask = Image.new("L", base.size)
    mask.paste(255, (120, 80, 330, 360))
    erased = _candidate(Image.new("RGBA", base.size), feather=5)

    assert "visible seam" in render_fill(base, mask, erased, edit_alpha=True).warning
    assert not render_fill(base, mask, erased, edit_alpha=False).warning
    assert not render_fill(base, mask, _candidate(base, feather=5), edit_alpha=True).warning
