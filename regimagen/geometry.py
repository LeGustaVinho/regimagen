"""Shared geometry for tiling, display, and export."""

import re
from dataclasses import dataclass
from itertools import product

FRAME = 1024
MARGIN = 64
CORE = FRAME - 2 * MARGIN
MIN_COLUMNS, MAX_COLUMNS = 2, 16
MIN_ROWS, MAX_ROWS = 2, 6


@dataclass(frozen=True)
class Grid:
    columns: int
    rows: int

    def __post_init__(self):
        if not MIN_COLUMNS <= self.columns <= MAX_COLUMNS or not MIN_ROWS <= self.rows <= MAX_ROWS:
            raise ValueError("The grid must have 2 to 16 columns and 2 to 6 rows.")

    @property
    def count(self) -> int:
        return self.columns * self.rows


@dataclass(frozen=True)
class Cell:
    column: int
    row: int

    @property
    def label(self) -> str:
        return f"Column {self.column + 1}, row {self.row + 1}"


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.top, self.bottom), slice(self.left, self.right)

    def intersect(self, other: "Rect") -> "Rect | None":
        result = Rect(
            max(self.left, other.left), max(self.top, other.top),
            min(self.right, other.right), min(self.bottom, other.bottom),
        )
        return result if result.width > 0 and result.height > 0 else None

    def relative_to(self, origin: "Rect") -> "Rect":
        return Rect(
            self.left - origin.left, self.top - origin.top,
            self.right - origin.left, self.bottom - origin.top,
        )

    def expand(self, horizontal: int, vertical: int) -> "Rect":
        return Rect(
            self.left - horizontal, self.top - vertical,
            self.right + horizontal, self.bottom + vertical,
        )


def output_size(width: int, height: int, grid: Grid) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    limit_width, limit_height = CORE * grid.columns, CORE * grid.rows
    if limit_width * height <= limit_height * width:
        return limit_width, max(1, limit_width * height // width)
    return max(1, limit_height * width // height), limit_height


def scale_limit(width: int, height: int) -> int:
    return min(CORE * MAX_COLUMNS // width, CORE * MAX_ROWS // height)


def scaled_grid(width: int, height: int, scale: int) -> Grid:
    if not 2 <= scale <= scale_limit(width, height):
        raise ValueError(f"The {scale}× scale exceeds the available grid size for this image.")
    return Grid(
        max(MIN_COLUMNS, (width * scale + CORE - 1) // CORE),
        max(MIN_ROWS, (height * scale + CORE - 1) // CORE),
    )


def parse_scales(text: str) -> tuple[int, ...]:
    normalized = re.sub(r"(?<=\d)\s+(?=[x×])", "", text.lower())
    values = re.split(r"[,;\s]+", normalized.strip().strip(",;").strip())
    if not all(re.fullmatch(r"[0-9]+[x×]?", value) for value in values):
        raise ValueError("Enter whole-number scales such as 2×, 3×, and 4×.")
    return tuple(dict.fromkeys(int(value.rstrip("x×")) for value in values))


def plan_scales(width: int, height: int, text: str, *, chained: bool) -> list["Layout"]:
    layouts = []
    input_scale = 1
    for scale in parse_scales(text):
        if chained and scale <= input_scale:
            raise ValueError("Enter scales in ascending order when using the previous stage result.")
        grid = scaled_grid(width, height, scale)
        layouts.append(Layout(width * input_scale, height * input_scale, grid, scale, input_scale))
        if chained:
            input_scale = scale
    return layouts


def suggested_grid(width: int, height: int) -> Grid:
    target = output_size(width, height, Grid(MAX_COLUMNS, MAX_ROWS))
    choices = (
        Grid(columns, rows)
        for columns, rows in product(
            range(MIN_COLUMNS, MAX_COLUMNS + 1), range(MIN_ROWS, MAX_ROWS + 1),
        )
    )
    return min(
        (grid for grid in choices if output_size(width, height, grid) == target),
        key=lambda grid: (grid.count, grid.columns),
    )


@dataclass(frozen=True)
class Layout:
    source_width: int
    source_height: int
    grid: Grid
    scale: int = 0
    input_scale: int = 1

    @property
    def size(self) -> tuple[int, int]:
        if self.scale:
            return (
                self.source_width * self.scale // self.input_scale,
                self.source_height * self.scale // self.input_scale,
            )
        return output_size(self.source_width, self.source_height, self.grid)

    @property
    def bounds(self) -> Rect:
        return Rect(0, 0, *self.size)

    @property
    def enlarges(self) -> bool:
        width, height = self.size
        return width > self.source_width and height > self.source_height

    @property
    def cells(self) -> tuple[Cell, ...]:
        return tuple(
            Cell(column, row)
            for row, column in product(range(self.grid.rows), range(self.grid.columns))
        )

    def core(self, cell: Cell) -> Rect:
        width, height = self.size
        columns, rows = self.grid.columns, self.grid.rows
        return Rect(
            cell.column * width // columns, cell.row * height // rows,
            (cell.column + 1) * width // columns, (cell.row + 1) * height // rows,
        )

    def frame(self, cell: Cell) -> Rect:
        box = self.core(cell)
        left = box.left - (FRAME - box.width) // 2
        top = box.top - (FRAME - box.height) // 2
        return Rect(left, top, left + FRAME, top + FRAME)

    def band(self, maximum: int = 32) -> tuple[int, int]:
        width, height = self.size
        return (
            min(maximum, max(1, width // self.grid.columns // 4)),
            min(maximum, max(1, height // self.grid.rows // 4)),
        )

    def neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        return tuple(
            Cell(column, row)
            for column, row in (
                (cell.column - 1, cell.row), (cell.column + 1, cell.row),
                (cell.column, cell.row - 1), (cell.column, cell.row + 1),
            )
            if 0 <= column < self.grid.columns and 0 <= row < self.grid.rows
        )

    def preview(self, anchor: Cell | None = None) -> tuple[Cell, ...]:
        if anchor is None:
            anchor = Cell((self.grid.columns - 2) // 2, (self.grid.rows - 2) // 2)
        left = min(anchor.column, self.grid.columns - 2)
        top = min(anchor.row, self.grid.rows - 2)
        return tuple(Cell(col, row) for row in (top, top + 1) for col in (left, left + 1))
