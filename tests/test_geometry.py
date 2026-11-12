import pytest

from regimagen.geometry import CORE, FRAME, MARGIN, Cell, Grid, Layout, suggested_grid


@pytest.mark.parametrize(("source", "grid", "output"), [
    ((1024, 1024), Grid(2, 2), (1792, 1792)),
    ((1024, 1024), Grid(6, 6), (5376, 5376)),
    ((1920, 1080), Grid(11, 6), (9557, 5376)),
    ((1080, 1920), Grid(4, 6), (3024, 5376)),
    ((3072, 1152), Grid(16, 6), (14336, 5376)),
    ((1200, 800), Grid(30, 20), (26880, 17920)),
])
def test_export_dimensions(source, grid, output):
    assert Layout(*source, grid).size == output


@pytest.mark.parametrize(("source", "expected"), [
    ((1024, 1024), Grid(3, 3)), ((1920, 1080), Grid(5, 3)),
    ((1080, 1920), Grid(3, 5)), ((3072, 1152), Grid(7, 3)),
])
def test_suggest_smallest_grid_that_fits_double_the_source(source, expected):
    assert suggested_grid(*source) == expected
    assert all(actual >= 2 * original for actual, original in zip(Layout(*source, expected).size, source))


def test_insufficient_grid_still_shows_compatible_resolution():
    original = (2048, 2048)
    assert not Layout(*original, Grid(2, 2)).enlarges
    assert Layout(*original, suggested_grid(*original)).enlarges
    assert Layout(8000, 8000, suggested_grid(8000, 8000)).grid == Grid(18, 18)


def test_grid_cannot_split_an_axis_into_subpixel_tiles():
    assert Layout(40, 40, Grid(1792, 2)).grid_error is None
    assert Layout(40, 40, Grid(1793, 2)).grid_error


@pytest.mark.parametrize("grid", [Grid(2, 2), Grid(7, 3), Grid(16, 6)])
def test_odd_aspect_cores_cover_output_exactly(grid):
    layout = Layout(1003, 877, grid)
    assert sum(layout.core(cell).width * layout.core(cell).height for cell in layout.cells) == (
        layout.bounds.width * layout.bounds.height
    )
    for cell in layout.cells:
        core = layout.core(cell)
        frame = layout.frame(cell)
        assert 0 < core.width <= CORE and 0 < core.height <= CORE
        assert frame.width == frame.height == FRAME
        assert all(value >= MARGIN for value in (
            core.left - frame.left, frame.right - core.right,
            core.top - frame.top, frame.bottom - core.bottom,
        ))
        for other in layout.neighbors(cell):
            assert core.intersect(layout.core(other)) is None


def test_preview_is_clamped_to_grid_edges():
    layout = Layout(1000, 1000, Grid(16, 6))
    assert set(layout.preview(Cell(15, 5))) == {Cell(14, 4), Cell(15, 4), Cell(14, 5), Cell(15, 5)}
    smallest = Layout(1000, 1000, Grid(2, 2))
    assert smallest.preview() == smallest.cells
