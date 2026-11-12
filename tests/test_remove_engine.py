import io
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image, ImageDraw

from regimagen.codex import Cancelled, GeneratedImage
from regimagen.fill import FillCandidate, generate_remove, removal_alpha_mask, render_remove


class _Capture:
    def __init__(self):
        self.requests = []

    def __call__(self, executable, directory, *, cancelled):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def generate(self, prompt, references, *, verify_references):
        assert verify_references
        images = []
        for path in references:
            with Image.open(path) as image:
                images.append(image.copy())
        self.requests.append((prompt, images))
        data = io.BytesIO()
        images[0].save(data, format="PNG")
        return GeneratedImage(data.getvalue(), "fixture", references_verified=True)


def _candidate(image, *, feather=3):
    return FillCandidate(image, {
        "crop": [0, 0, *image.size], "shift": [0, 0], "color": [0, 0, 0], "feather": feather,
    })


def test_prepared_references_hide_object_including_repeated_virtual_edges():
    base = Image.new("RGB", (80, 40), (24, 117, 81))
    base.paste((255, 0, 0), (0, 0, 12, 19))
    core = Image.new("L", base.size)
    core.paste(255, (0, 0, 12, 19))
    footprint = Image.new("L", base.size)
    footprint.paste(255, (0, 0, 16, 23))
    capture = _Capture()

    for prepare in (False, True):
        generate_remove(base, core, footprint, ("Red object",), Path("codex"), cancelled=Event(),
                        client_factory=capture, prepare=prepare)

    def shows_red(image):
        rgb = np.asarray(image)
        return bool(((rgb[..., 0] > 200) & (rgb[..., 1] < 40) & (rgb[..., 2] < 40)).any())

    raw_prompt, raw_images = capture.requests[0]
    prepared_prompt, prepared_images = capture.requests[1]
    assert "original square crop" in raw_prompt
    assert "background approximation" in prepared_prompt
    assert all(shows_red(image) for image in raw_images)
    assert not any(shows_red(image) for image in prepared_images)
    assert prepared_images[1].tobytes() != prepared_images[0].tobytes()


def test_approved_footprint_and_protected_hole_are_exact_at_source_resolution():
    base = Image.new("RGB", (48, 48), "red")
    core = Image.new("L", base.size)
    core.paste(255, (15, 12, 35, 31))
    footprint = Image.new("L", base.size)
    footprint.paste(255, (13, 10, 37, 33))
    for mask in (core, footprint):
        mask.paste(0, (22, 16, 25, 25))
    result = render_remove(base, core, footprint, _candidate(Image.new("RGB", base.size, "blue"), feather=7))
    output = np.asarray(result.compose(base))
    approved = np.asarray(footprint) != 0
    target = np.asarray(core) != 0
    np.testing.assert_array_equal(output[~approved], np.asarray(base)[~approved])
    assert np.all(output[target] == (0, 0, 255))
    assert 0 < output[13, 13, 0] < 255
    assert 0 < output[13, 13, 2] < 255


def test_isolated_target_restores_clear_alpha_even_with_opaque_generated_patch():
    base = Image.new("RGBA", (48, 48), (120, 200, 42, 0))
    base.paste((220, 100, 42, 225), (16, 16, 32, 32))
    core = Image.new("L", base.size)
    core.paste(255, (16, 16, 32, 32))
    footprint = Image.new("L", base.size)
    footprint.paste(255, (14, 14, 34, 34))
    generated = _candidate(Image.new("RGBA", base.size, (32, 50, 90, 255)))
    output = np.asarray(render_remove(base, core, footprint, generated).compose(base))

    assert not output[..., 3].any()
    protected = np.asarray(footprint) == 0
    np.testing.assert_array_equal(output[protected], np.asarray(base)[protected])

    # One real nontransparent neighbor is enough to make it ambiguous.
    base.putpixel((13, 20), (110, 110, 110, 1))
    mixed = render_remove(base, core, footprint, generated).compose(base)
    assert mixed.getpixel((16, 16)) == (32, 50, 90, 255)
    assert mixed.getpixel((13, 20)) == (110, 110, 110, 1)


def test_fully_clear_removal_never_looks_up_or_calls_imagegen():
    base = Image.new("RGBA", (64, 64), (90, 70, 40, 0))
    base.paste((160, 100, 90, 255), (15, 16, 29, 32))
    core = Image.new("L", base.size)
    core.paste(255, (15, 16, 29, 32))
    footprint = Image.new("L", base.size)
    footprint.paste(127, (12, 13, 32, 35))
    capture = _Capture()

    def resolve():
        pytest.fail("A clear background does not require Codex")

    candidate = generate_remove(base, core, footprint, (), resolve, cancelled=Event(), client_factory=capture)
    preview = render_remove(base, core, footprint, candidate)
    output = preview.compose(base)

    assert candidate.metadata["generation_backend"] == "alpha"
    assert not capture.requests
    assert not preview.warning
    assert output.getchannel("A").getbbox() is None
    outside = np.asarray(footprint) == 0
    np.testing.assert_array_equal(np.asarray(output)[outside], np.asarray(base)[outside])


def test_attached_target_is_cleared_before_imagegen_but_visible_mount_is_reconstructed():
    base = Image.new("RGBA", (112, 112), (60, 50, 40, 0))
    base.paste((40, 95, 55, 255), (8, 78, 94, 95))
    base.paste((10, 20, 160, 255), (42, 36, 54, 83))
    base.paste((255, 0, 0, 255), (42, 18, 54, 36))
    core = Image.new("L", base.size)
    core.paste(255, (42, 18, 54, 83))
    footprint = Image.new("L", base.size)
    footprint.paste(255, (38, 14, 58, 87))
    capture = _Capture()
    preparations = []

    def prepare_crop(reference, coverage, crop, size):
        row, column = ((coord - origin) * reference.width // crop.width
                       for coord, origin in ((25, crop.top), (48, crop.left)))
        assert reference.getpixel((column, row))[3] == 0
        assert coverage[row, column] == 0
        row = (80 - crop.top) * reference.height // crop.height
        assert reference.getpixel((column, row)) == (10, 20, 160, 255)
        assert coverage[row, column] != 0
        preparations.append(size)
        return reference

    generate_remove(base, core, footprint, ("antenna",), Path("codex"), cancelled=Event(),
                    client_factory=capture, prepare_crop=prepare_crop)
    assert preparations == [base.size]
    assert len(capture.requests) == 1
    prompt, references = capture.requests[0]
    assert "already transparent in the references" in prompt
    for reference in references:
        r, g, b = np.moveaxis(np.asarray(reference)[..., :3], -1, 0)
        assert not np.any((r > 240) & (g < 25) & (b < 25))

    replacement = (32, 90, 43, 255)
    candidate = _candidate(Image.new("RGBA", base.size, replacement))
    output = render_remove(base, core, footprint, candidate).compose(base)
    assert output.getpixel((48, 25)) == (0, 0, 0, 0)
    assert output.getpixel((48, 80)) == replacement
    outside = np.asarray(footprint) == 0
    np.testing.assert_array_equal(np.asarray(output)[outside], np.asarray(base)[outside])


@pytest.mark.parametrize("start,end", [((55, 8), (55, 104)), ((8, 8), (104, 104))])
def test_transparent_sides_do_not_erase_a_bridge_between_opaque_endpoints(start, end):
    base = Image.new("RGBA", (112, 112))
    ImageDraw.Draw(base).line((start, end), (220, 180, 120, 255), width=9)
    footprint = Image.new("L", base.size)
    footprint.paste(255, (30, 30, 80, 80))
    assert removal_alpha_mask(base, footprint) is None


def test_closed_frame_is_reconstructed_between_separate_contacts():
    frame = Image.new("RGBA", (180, 180))
    ImageDraw.Draw(frame).line(
        [(90, 12), (139, 58), (83, 117), (30, 72), (90, 12)],
        fill=(200, 200, 200, 255), width=5, joint="curve",
    )
    base = frame.copy()
    ImageDraw.Draw(base).ellipse((47, 30, 73, 56), fill=(255, 40, 30, 255))
    footprint = Image.new("L", base.size)
    footprint.paste(255, (40, 26, 81, 63))
    assert removal_alpha_mask(base, footprint) is None

    output = render_remove(base, footprint, footprint, _candidate(frame)).compose(base)
    assert output.getpixel((62, 38)) == frame.getpixel((62, 38)) == (200, 200, 200, 255)


def test_separate_contacts_at_one_end_still_allow_the_protruding_tip_to_clear():
    base = Image.new("RGBA", (130, 150))
    color = (200, 200, 200, 255)
    base.paste(color, (49, 20, 64, 111))
    base.paste(color, (35, 108, 48, 135))
    base.paste(color, (65, 108, 78, 135))
    base.paste(color, (35, 125, 78, 135))
    footprint = Image.new("L", base.size)
    footprint.paste(255, (42, 17, 71, 116))

    clear = removal_alpha_mask(base, footprint)
    assert clear is not None and clear.getpixel((55, 27)) == 255
    assert clear.getpixel((55, 110)) == 0


def test_isolated_component_clears_even_when_another_target_needs_reconstruction():
    base = Image.new("RGBA", (400, 240))
    base.paste((200, 40, 40, 255), (20, 30, 181, 201))
    base.paste((180, 170, 150, 255), (230, 20, 380, 220))
    footprint = Image.new("L", base.size)
    footprint.paste(255, (18, 28, 183, 203))
    footprint.paste(255, (250, 80, 310, 150))
    clear = removal_alpha_mask(base, footprint)
    assert clear is not None
    assert clear.getbbox() == (18, 28, 183, 203)


def test_invalid_footprint_and_cancellation_never_generate():
    base = Image.new("RGB", (40, 40))
    core = Image.new("L", base.size)
    core.paste(255, (15, 15, 22, 22))
    footprint = core.copy()
    footprint.putpixel((16, 16), 0)
    client = _Capture()
    with pytest.raises(ValueError, match="entire removal mask"):
        generate_remove(base, core, footprint, (), Path("codex"), cancelled=Event(), client_factory=client)
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        generate_remove(base, core, core, (), Path("codex"), cancelled=cancel, client_factory=client)
    assert not client.requests
