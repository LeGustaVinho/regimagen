"""Regimagen's main window."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from threading import Event

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from PySide6.QtCore import QPointF, QRectF, QSettings, QSignalBlocker, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFontDatabase, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractScrollArea, QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QStackedWidget,
    QStyle, QToolTip, QVBoxLayout, QWidget,
)

from .codex import AuthRequired, Cancelled, CodexError, QuotaPaused, find_codex
from .composite import export_parts, export_png
from .document import Document, box, export_document, render
from .document_fill import ActiveFillTarget, LayeredVariant, active_fill_target, fill_reference, layered_variant
from .editor import Editor, dimensions
from .files import atomic_output, load_source
from .fill import FillCandidate, FillPreview, generate_fill, generate_remove, render_fill, render_remove
from .fill_editor import FillEditor
from .geometry import (
    DEFAULT_SCALES, MAX_PNG_DIMENSION, MIN_COLUMNS, MIN_ROWS, Cell, Grid, Layout, plan_scales, suggested_grid,
)
from .jobs import load_job_input, pending_cells, refresh_thumbnails, run_scale_batch, run_tiles
from .project import PROJECT_SUFFIX, load_project, save_project
from .remove_detection import detect_removal, segment_removal_click, segment_removal_lasso
from .remove_editor import RemoveEditor
from .storage import FillEdit, Job, Snapshot, Source, Store, Tile
from .viewer import Comparison, qimage


def number(value: int) -> str:
    return f"{value:,}"


def logo(size=48) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 48, size / 48)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#283f3c"))
    painter.drawRoundedRect(QRectF(0, 0, 48, 48), 12, 12)
    for col, row, color in ((0, 0, "#8ce1c6"), (1, 0, "#559b8a"), (0, 1, "#559b8a"), (1, 1, "#8ce1c6")):
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(QRectF(10 + 16 * col, 10 + 16 * row, 12, 12), 3, 3)
    painter.end()
    return pixmap


class Task(QThread):
    updated = Signal(object)
    succeeded = Signal(object)
    failed = Signal(object)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.pause = Event()
        self.cancelled = Event()

    def run(self):
        try:
            self.succeeded.emit(self.operation(self.updated.emit, self.pause, self.cancelled))
        except Exception as exc:
            self.failed.emit(exc)


@dataclass
class FillSession:
    source: Source
    image: Image.Image
    sha256: str
    editor: FillEditor
    previous: FillEdit | None = None
    saved_mask: Image.Image | None = None
    saved_candidate: FillCandidate | None = None
    saved_preview: FillPreview | None = None
    variants: list[tuple[FillCandidate, FillPreview]] = field(default_factory=list)

    def reusable(self, mask: Image.Image, prompt: str, edit_alpha: bool) -> FillCandidate | None:
        if (self.previous is None or self.saved_mask is None or self.previous.base_sha256 != self.sha256
                or prompt != self.previous.prompt or edit_alpha != self.previous.edit_alpha):
            return None
        if mask.size == self.saved_mask.size and ImageChops.subtract(mask, self.saved_mask).getbbox() is None:
            return self.saved_candidate
        return None


@dataclass
class LayerFillSession:
    document: Document
    editor: FillEditor
    target: ActiveFillTarget | None
    variants: list[LayeredVariant] = field(default_factory=list)


def quota_reset_text(reset: int | None) -> str | None:
    if reset is None:
        return None
    try:
        return f"{datetime.fromtimestamp(reset):%Y-%m-%d %H:%M}"
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def quota_message(reset: int | None, action: str) -> str:
    timestamp = quota_reset_text(reset)
    if timestamp is None:
        return f"Codex usage limit reached. {action} when the limit resets."
    return f"Codex usage limit reached. {action} after {timestamp}."


def tile_state(tile: Tile) -> tuple[str, str]:
    if tile.error:
        return "Failed", "#9c5058"
    if tile.stale:
        return "Prompt changed", "#56617c"
    if tile.warning and tile.artifact:
        return "Needs review", "#9b783d"
    if tile.artifact:
        return "Ready", "#4d9f8c"
    if tile.attempt:
        return "Waiting to resume", "#56617c"
    return "Pending", "#313741"


class TileMap(QAbstractScrollArea):
    picked = Signal(object)
    row_height = 21
    min_column_width = 16
    max_visible_rows = 8

    def __init__(self):
        super().__init__()
        self.grid = Grid(2, 2)
        self.tiles: dict[Cell, Tile] = {}
        self.selection: Cell | None = None
        self.active: Cell | None = None
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Tile grid")
        self.setToolTip("Select a tile. Use the arrow keys to navigate the grid.")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._update_scrollbars()

    def set_tiles(self, grid: Grid, tiles: dict[Cell, Tile], selection: Cell | None, active: Cell | None):
        grid_changed = grid != self.grid
        selection_changed = selection != self.selection
        self.grid, self.tiles, self.selection, self.active = grid, tiles, selection, active
        self._update_scrollbars()
        if grid_changed:
            self.horizontalScrollBar().setValue(0)
            self.verticalScrollBar().setValue(0)
        if selection is not None and selection_changed:
            for index, bar in ((selection.column, self.horizontalScrollBar()), (selection.row, self.verticalScrollBar())):
                if index < bar.value():
                    bar.setValue(index)
                elif index >= bar.value() + bar.pageStep():
                    bar.setValue(index - bar.pageStep() + 1)
        self.viewport().update()

    def _update_scrollbars(self):
        rows = min(self.grid.rows, self.max_visible_rows)
        extent = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        width = self.width() - 6 - (extent if rows < self.grid.rows else 0)
        columns = min(self.grid.columns, max(1, width // self.min_column_width))
        horizontal, vertical = self.horizontalScrollBar(), self.verticalScrollBar()
        horizontal.setPageStep(columns)
        horizontal.setRange(0, self.grid.columns - columns)
        vertical.setPageStep(rows)
        vertical.setRange(0, self.grid.rows - rows)
        height = rows * self.row_height + 5 + (extent if columns < self.grid.columns else 0)
        if height != self.height():
            self.setFixedHeight(height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scrollbars()

    def scrollContentsBy(self, _dx, _dy):
        self.viewport().update()

    def _column_width(self):
        return (self.viewport().width() - 6) / self.horizontalScrollBar().pageStep()

    def _rect(self, cell: Cell) -> QRectF:
        width = self._column_width()
        column = cell.column - self.horizontalScrollBar().value()
        row = cell.row - self.verticalScrollBar().value()
        return QRectF(3 + column * width, 3 + row * self.row_height, width - 3, self.row_height - 3)

    def _at(self, position: QPointF) -> Cell | None:
        width = self._column_width()
        if width <= 0 or position.x() < 3 or position.y() < 3:
            return None
        horizontal, vertical = self.horizontalScrollBar(), self.verticalScrollBar()
        column = int((position.x() - 3) // width)
        row = int((position.y() - 3) // self.row_height)
        if column >= horizontal.pageStep() or row >= vertical.pageStep():
            return None
        cell = Cell(horizontal.value() + column, vertical.value() + row)
        return cell if self._rect(cell).contains(position) else None

    def paintEvent(self, event):
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        horizontal, vertical = self.horizontalScrollBar(), self.verticalScrollBar()
        for row in range(vertical.value(), vertical.value() + vertical.pageStep()):
            for col in range(horizontal.value(), horizontal.value() + horizontal.pageStep()):
                cell = Cell(col, row)
                _, color = tile_state(self.tiles.get(cell, Tile()))
                painter.setBrush(QColor("#768fe2" if cell == self.active else color))
                painter.setPen(Qt.PenStyle.NoPen)
                rect = self._rect(cell)
                painter.drawRoundedRect(rect, 3, 3)
                if cell == self.selection:
                    painter.setPen(QPen(QColor("#d4fff1"), 1.5))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(rect, 3, 3)

    def mousePressEvent(self, event):
        cell = self._at(event.position())
        if cell is not None and event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.picked.emit(cell)

    def mouseMoveEvent(self, event):
        cell = self._at(event.position())
        if cell is not None:
            state = "Editing" if cell == self.active else tile_state(self.tiles.get(cell, Tile()))[0]
            prompt = " · tile prompt set" if self.tiles.get(cell, Tile()).prompt else ""
            QToolTip.showText(event.globalPosition().toPoint(), f"{cell.label} · {state}{prompt}", self)
        else:
            QToolTip.hideText()

    def keyPressEvent(self, event):
        offsets = {
            Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1),
        }
        if event.key() not in offsets:
            return super().keyPressEvent(event)
        current = self.selection or Cell(0, 0)
        dx, dy = offsets[event.key()]
        self.picked.emit(Cell(
            max(0, min(self.grid.columns - 1, current.column + dx)),
            max(0, min(self.grid.rows - 1, current.row + dy)),
        ))
        event.accept()


class Window(QMainWindow):
    def __init__(self, store: Store, settings: QSettings):
        super().__init__()
        self.store, self.settings = store, settings
        self.source: Source | None = None
        self.job: Job | None = None
        self.original = QPixmap()
        self.sprite = None
        self.fill: FillSession | None = None
        self.layer_fill: LayerFillSession | None = None
        self.applied_fill: FillEdit | None = None
        self.project_path: Path | None = None
        self.selected: Cell | None = None
        self.active: Cell | None = None
        self.task: Task | None = None
        self.after_task: Callable[[], None] | None = None
        self.operation = ""
        self.closing = False
        self.export_message = ""
        self.output_folder: Path | None = None
        self.can_cancel_task = False
        self.setWindowTitle("Regimagen")
        self.setWindowIcon(QIcon(logo()))
        self.setAcceptDrops(True)
        self.setMinimumSize(920, 625)
        self.resize(1360, 850)
        self._build()
        self._history()
        self._refresh()

    @property
    def grid(self) -> Grid:
        return Grid(self.columns.value(), self.rows.value())

    @property
    def image_layout(self) -> Layout | None:
        if self.job is not None:
            return self.job.layout
        return Layout(self.source.width, self.source.height, self.grid) if self.source else None

    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        shell = QVBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        header = QFrame()
        header.setObjectName("header")
        toolbar = QHBoxLayout(header)
        toolbar.setContentsMargins(22, 14, 22, 14)
        toolbar.setSpacing(12)
        mark = QLabel()
        mark.setPixmap(logo(34))
        brand = QLabel("Regimagen")
        brand.setObjectName("brand")
        toolbar.addWidget(mark)
        toolbar.addWidget(brand)
        toolbar.addSpacing(14)
        self.open_button = QPushButton("File")
        file_menu = QMenu(self.open_button)
        self.new_action = file_menu.addAction("New canvas…", self._new_document)
        self.open_action = file_menu.addAction("Open…", self._choose_image)
        file_menu.addSeparator()
        self.save_action = file_menu.addAction("Save project", self._save_project)
        self.save_as_action = file_menu.addAction("Save project as…", lambda: self._save_project(save_as=True))
        self.export_action = file_menu.addAction("Export image…", self._export_document)
        for action, shortcut in (
            (self.new_action, "Ctrl+N"), (self.open_action, "Ctrl+O"),
            (self.save_action, "Ctrl+S"), (self.save_as_action, "Ctrl+Shift+S"),
        ):
            action.setShortcut(shortcut)
        self.open_button.setMenu(file_menu)
        self.addActions(file_menu.actions())
        toolbar.addWidget(self.open_button)
        self.history = QComboBox()
        self.history.setMinimumContentsLength(17)
        self.history.setMaximumWidth(255)
        self.history.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.history.currentIndexChanged.connect(self._load_history)
        toolbar.addWidget(self.history)
        self.back_button = QPushButton("Back to editor")
        self.back_button.clicked.connect(self._return_to_editor)
        toolbar.addWidget(self.back_button)
        self.version_button = QPushButton("Image version")
        version_menu = QMenu(self.version_button)
        version_menu.addAction("Reopen fill…", lambda: self._open_fill(reopen=True))
        version_menu.addAction("Export source PNG…", self._export_fill)
        version_menu.addAction("Open version in editor", self._edit_input)
        self.version_button.setMenu(version_menu)
        toolbar.addWidget(self.version_button)
        toolbar.addStretch()
        self.source_label = QLabel()
        self.source_label.setObjectName("muted")
        self.source_label.setMinimumWidth(116)
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        toolbar.addWidget(self.source_label)
        self.settings_button = QPushButton("Codex CLI")
        settings_menu = QMenu(self.settings_button)
        settings_menu.addAction("Select codex.exe…", self._choose_codex)
        settings_menu.addAction("Use automatic detection", self._reset_codex)
        self.settings_button.setMenu(settings_menu)
        toolbar.addWidget(self.settings_button)
        shell.addWidget(header)

        workspace = QHBoxLayout()
        workspace.setContentsMargins(18, 16, 18, 18)
        workspace.setSpacing(14)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._empty())
        self.comparison = Comparison()
        self.comparison.fill_button.clicked.connect(lambda: self._open_fill())
        self.stack.addWidget(self.comparison)
        workspace.addWidget(self.stack, 1)
        self.editor = Editor()
        self.editor.changed.connect(self._refresh)
        self.editor.import_requested.connect(self._import_layer)
        self.editor.reconstruct_requested.connect(self._reconstruct)
        self.editor.fill_requested.connect(self._open_layer_fill)
        self.editor.remove_requested.connect(lambda: self._open_layer_fill(remove=True))
        self.stack.addWidget(self.editor)
        self.sidebar = self._sidebar()
        workspace.addWidget(self.sidebar)
        shell.addLayout(workspace, 1)
        shell.addWidget(self._footer())
        self.stack.currentChanged.connect(self._clear_idle_feedback)
        self.editor.undo.indexChanged.connect(self._clear_idle_feedback)

    def _footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("taskFooter")
        footer.setMinimumHeight(42)
        row = QHBoxLayout(footer)
        row.setContentsMargins(18, 4, 18, 4)
        row.setSpacing(12)
        self.task_status = QLabel()
        self.task_status.setTextFormat(Qt.TextFormat.PlainText)
        self.task_status.setWordWrap(True)
        self.task_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.task_status, 1)
        self.task_progress = QProgressBar()
        self.task_progress.setAccessibleName("Task progress")
        self.task_progress.setTextVisible(False)
        self.task_progress.setFixedSize(128, 6)
        self.task_progress.hide()
        row.addWidget(self.task_progress)
        self.cancel_task = QPushButton("Cancel")
        self.cancel_task.clicked.connect(self._request_cancel)
        self.cancel_task.hide()
        row.addWidget(self.cancel_task)
        self.open_folder = QPushButton("Open folder")
        self.open_folder.setObjectName("openOutputFolder")
        self.open_folder.clicked.connect(self._open_output_folder)
        self.open_folder.hide()
        row.addWidget(self.open_folder)
        return footer

    def _empty(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("emptyPanel")
        column = QVBoxLayout(panel)
        column.setContentsMargins(24, 24, 24, 24)
        column.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark = QLabel()
        mark.setPixmap(logo(58))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(mark)
        column.addSpacing(12)
        title = QLabel("Open an image or project")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(title)
        help_text = QLabel("Drop a PNG, JPEG, WebP, or .regimagen project here.")
        help_text.setObjectName("muted")
        help_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(help_text)
        column.addSpacing(14)
        button = QPushButton("Choose file")
        button.setProperty("primary", True)
        button.clicked.connect(self._choose_image)
        column.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.empty_button = button
        self.new_button = QPushButton("New canvas")
        self.new_button.clicked.connect(self._new_document)
        column.addWidget(self.new_button, alignment=Qt.AlignmentFlag.AlignCenter)
        return panel

    def _sidebar(self) -> QScrollArea:
        pane = QScrollArea()
        pane.setObjectName("sidebar")
        pane.setWidgetResizable(True)
        pane.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pane.setFixedWidth(328)
        panel = QWidget()
        panel.setObjectName("sidebarContent")
        pane.setWidget(panel)
        column = QVBoxLayout(panel)
        column.setContentsMargins(16, 16, 16, 16)
        column.setSpacing(10)
        title = QLabel("Grid")
        title.setObjectName("section")
        column.addWidget(title)
        fields = QHBoxLayout()
        self.columns = self._field(fields, "Columns", MIN_COLUMNS)
        self.rows = self._field(fields, "Rows", MIN_ROWS)
        column.addLayout(fields)
        self.dimensions = QLabel("—")
        self.dimensions.setObjectName("dimensions")
        column.addWidget(self.dimensions)
        self.edits = QLabel()
        self.edits.setObjectName("muted")
        column.addWidget(self.edits)
        self.general_prompt = QPushButton("Global prompt…")
        self.general_prompt.clicked.connect(lambda: self._edit_prompt())
        column.addWidget(self.general_prompt)
        self.grid_map = TileMap()
        self.grid_map.picked.connect(self._select)
        column.addWidget(self.grid_map)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        column.addWidget(self.progress)
        self.status = QLabel()
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.status)
        self.generate = QPushButton("Generate image")
        self.generate.setProperty("primary", True)
        self.generate.clicked.connect(self._generate_or_pause)
        column.addWidget(self.generate)
        self.preview = QPushButton("Generate 2×2 preview")
        self.preview.clicked.connect(self._start_preview)
        column.addWidget(self.preview)
        self.batch = QPushButton("Generate scale batch…")
        self.batch.clicked.connect(self._start_batch)
        column.addWidget(self.batch)
        self.subagents = QCheckBox("Use Codex subagents")
        self.subagents.setChecked(self.settings.value("subagents", False, bool))
        self.subagents.toggled.connect(lambda enabled: self.settings.setValue("subagents", enabled))
        self.subagents.setToolTip("Generate up to four tiles in parallel. Pausing lets the active group finish.")
        column.addWidget(self.subagents)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        column.addSpacing(3)
        column.addWidget(divider)
        self.tile_title = QLabel("Tile")
        self.tile_title.setObjectName("section")
        column.addWidget(self.tile_title)
        self.tile_note = QLabel()
        self.tile_note.setWordWrap(True)
        self.tile_note.setObjectName("muted")
        column.addWidget(self.tile_note)
        self.cell_prompt = QPushButton("Tile prompt…")
        self.cell_prompt.clicked.connect(self._edit_cell_prompt)
        column.addWidget(self.cell_prompt)
        self.retry = QPushButton("Regenerate tile")
        self.retry.clicked.connect(self._retry)
        column.addWidget(self.retry)
        column.addStretch()
        self.export = QPushButton("Export PNG")
        self.export.clicked.connect(self._export)
        self.export.setShortcut("Ctrl+Shift+S")
        column.addWidget(self.export)
        self.edit_result = QPushButton("Open result in editor")
        self.edit_result.clicked.connect(self._open_result)
        column.addWidget(self.edit_result)
        self.parts = QPushButton("Export tiles (ZIP)")
        self.parts.clicked.connect(lambda: self._export(parts=True))
        column.addWidget(self.parts)
        return pane

    def _field(self, row, title: str, minimum: int) -> QSpinBox:
        field = QVBoxLayout()
        name = QLabel(title)
        name.setObjectName("muted")
        field.addWidget(name)
        value = QSpinBox()
        value.setRange(minimum, MAX_PNG_DIMENSION)
        value.setKeyboardTracking(False)
        value.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        value.setAccessibleName(title)
        value.valueChanged.connect(self._grid_changed)
        field.addWidget(value)
        row.addLayout(field)
        return value

    def _history(self):
        with QSignalBlocker(self.history):
            self.history.clear()
            self.history.addItem("Recent", None)
            for path in self._recent_projects():
                self.history.addItem(path.name, path)
                if self.stack.currentWidget() is self.editor and self.project_path == path:
                    self.history.setCurrentIndex(self.history.count() - 1)
            for item in self.store.recent_jobs():
                scale = f" · {item.scale}×" if item.scale else ""
                if item.input_scale > 1:
                    scale += f" from {item.input_scale}×"
                self.history.addItem(f"{item.source.name} · {item.grid.columns}×{item.grid.rows}{scale}", item)
                if self.stack.currentWidget() is self.comparison and self.job and self.job.id == item.id:
                    self.history.setCurrentIndex(self.history.count() - 1)
            for source in self.store.fill_sources():
                self.history.addItem(f"{source.name} · Image", source)
                if (self.stack.currentWidget() is self.comparison and self.job is None
                        and self.source and self.source.id == source.id):
                    self.history.setCurrentIndex(self.history.count() - 1)
            self.history.setToolTip(self.history.currentText())

    def _load_history(self, index: int):
        item: Path | Job | Source | None = self.history.itemData(index)
        if item is None or self.task is not None or self.fill is not None or self.layer_fill is not None:
            return
        if isinstance(item, Path):
            with QSignalBlocker(self.history):
                self.history.setCurrentIndex(0)
            self._import(item)
            return

        def load(_update, _pause, cancel):
            if isinstance(item, Source):
                image = load_source(item.path)
                sprite = image if image.mode == "RGBA" else None
                return item, qimage(np.asarray(image)), suggested_grid(*image.size), sprite, None
            image = load_job_input(item)
            sprite = image if image.mode == "RGBA" else None
            refresh_thumbnails(self.store, item, alpha=sprite, cancelled=cancel.is_set)
            return item.source, qimage(np.asarray(image)), item.grid, sprite, item

        self._launch("import", load, self._show_source, message="Opening image…")

    def _open_fill(self, *, reopen=False):
        if self.source is None or self.stack.currentWidget() is not self.comparison or self.task is not None:
            return
        previous = self.applied_fill if reopen else None
        if reopen and previous is None:
            return
        source = previous.base if previous is not None else self.source

        def load(_update, _pause, _cancel):
            with source.path.open("rb") as stream:
                sha256 = file_digest(stream, "sha256").hexdigest()
                stream.seek(0)
                image = load_source(stream, formats=("PNG",))
            mask = candidate = preview = None
            if previous is not None:
                with Image.open(previous.mask_path, formats=("PNG",)) as saved:
                    mask = saved.convert("L")
                bounds = mask.getbbox()
                if mask.size != image.size or bounds is None:
                    raise ValueError("The saved selection does not match the base image.")
                if previous.base_sha256 == sha256:
                    with Image.open(previous.candidate_path, formats=("PNG",)) as saved:
                        candidate = FillCandidate(saved.copy(), previous.metadata)
                    warning = previous.metadata.get("preview_warning", "")
                    with Image.open(previous.source.path, formats=("PNG",)) as saved:
                        if saved.size != image.size or saved.mode != image.mode:
                            raise ValueError("The saved fill does not match the base image.")
                        preview = FillPreview(saved.crop(bounds), bounds[:2], warning)
            return image, sha256, mask, candidate, preview

        def show(loaded):
            if loaded is None or self.closing:
                return
            image, sha256, mask, candidate, preview = loaded
            options = {} if previous is None else {"mask": mask, "prompt": previous.prompt,
                                                   "edit_alpha": previous.edit_alpha}
            editor = FillEditor(image, **options)
            editor.set_busy(True)
            self.fill = FillSession(source, image, sha256, editor, previous, mask, candidate, preview)
            editor.generate_requested.connect(self._generate_fill)
            editor.apply_requested.connect(self._apply_fill)
            editor.discard_requested.connect(self._discard_fill)
            editor.cancel_requested.connect(self._cancel_fill)
            editor.changed.connect(self._fill_changed)
            self.stack.addWidget(editor)
            self.stack.setCurrentWidget(editor)
            self.sidebar.hide()
            if candidate is not None and preview is not None:
                self._append_fill_variant((candidate, preview), already_applied=True)
            elif previous is not None:
                editor.set_message("The base image changed. Generate a new variation to apply this selection.")

        self._launch("open_fill", load, show, message="Opening Generative Fill…")

    def _open_layer_fill(self, *, remove=False):
        if self.task is not None or self.stack.currentWidget() is not self.editor:
            return
        document, layer, selection = self.editor.document, self.editor.active_layer, self.editor.selection
        if document is None:
            return

        def load(_update, _pause, cancel):
            image = fill_reference(render(document, cancelled=cancel.is_set))
            return image, None if remove else active_fill_target(document, layer)

        def show(loaded):
            if loaded is None or self.closing:
                return
            image, target = loaded
            active = (target.image, box(target.area)) if target is not None else None
            if remove:
                editor = RemoveEditor(image, mask=selection)
                editor.detect_requested.connect(self._detect_remove)
                editor.semantic_detection_requested.connect(lambda: self._detect_remove(semantic=True))
                editor.generate_requested.connect(self._generate_remove)
            else:
                editor = FillEditor(image, mask=selection, layered=True, active_target=active)
                editor.generate_requested.connect(self._generate_fill)
            editor.set_busy(True)
            session = self.layer_fill = LayerFillSession(document, editor, target)
            editor.cancel_requested.connect(self._cancel_fill)
            editor.apply_requested.connect(self._apply_layer_fill)
            editor.discard_requested.connect(self._close_layer_fill)
            editor.changed.connect(session.variants.clear)
            self.stack.addWidget(editor)
            self.stack.setCurrentWidget(editor)

        name = "Generative Remove" if remove else "Generative Fill"
        self._launch("open_fill", load, show, message=f"Opening {name}…")

    def _remove_session(self):
        session = self.layer_fill
        if self.task is not None or session is None or not isinstance(session.editor, RemoveEditor):
            return None
        if session.document is not self.editor.document:
            self._close_layer_fill()
            self.editor.set_message("The document changed. Open Generative Remove again to use the current image.")
            return None
        return session

    def _detect_remove(self, *, semantic=False):
        session = self._remove_session()
        if session is None:
            return
        editor = session.editor
        image, _ = editor.generation_input()
        selection, points = editor.detection_mask, editor.detection_points
        if not points and selection.getbbox() is None:
            return
        executable = self._codex_resolver()

        def detect(_update, _pause, cancel):
            if not semantic:
                if points:
                    targets = segment_removal_click(image, points[0], cancelled=cancel)
                else:
                    targets = segment_removal_lasso(image, selection, cancelled=cancel)
                if targets:
                    return targets, None
            if cancel.is_set():
                raise Cancelled("Object detection canceled.")
            guide = selection
            if points:
                x, y = points[0]
                radius = max(3, min(14, round(min(image.size) * 0.01)))
                guide = Image.new("L", image.size)
                ImageDraw.Draw(guide).ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            return detect_removal(image, guide, executable(), cancelled=cancel), True

        def show(result):
            if self.closing:
                return
            if result is None:
                editor.set_message("Object detection canceled.")
            else:
                targets, detailed = result
                editor.set_targets(targets, detailed=detailed)

        self._launch("detect_remove", detect, show, message="Detecting objects…")

    def _generate_remove(self):
        session = self._remove_session()
        if session is None:
            return
        inputs = session.editor.removal_inputs()
        if inputs is None:
            return
        core, footprint, labels = inputs
        image, _ = session.editor.generation_input()
        executable = self._codex_resolver()
        prepare = len(session.variants) % 2 == 1

        def generate(_update, _pause, cancel):
            candidate = generate_remove(image, core, footprint, labels, executable, cancelled=cancel, prepare=prepare)
            preview = render_remove(image, core, footprint, candidate)
            if cancel.is_set():
                raise Cancelled("Removal canceled.")
            return layered_variant(session.document, footprint, preview, None, name="Generative Remove")

        self._launch("generate_remove", generate, self._append_layer_variant, message="Removing selected area…")

    def _append_layer_variant(self, variant: LayeredVariant | None):
        session = self.layer_fill
        if session is None or self.closing:
            return
        if variant is None:
            session.editor.set_message("Generation canceled.")
        else:
            session.variants.append(variant)
            session.editor.add_variant(variant.preview)

    def _apply_layer_fill(self):
        session = self.layer_fill
        if session is None or self.task is not None or not session.editor.can_apply:
            return
        if isinstance(session.editor, RemoveEditor):
            if self._remove_session() is None:
                return
            inputs = session.editor.removal_inputs()
            if inputs is None:
                return
            selection = inputs[1]
        else:
            selection = session.editor.mask
        variant = session.variants[session.editor.selected_variant]
        self.editor.apply_fill(variant.document, variant.layer_id, selection)
        self._close_layer_fill()

    def _close_layer_fill(self):
        if self.layer_fill is None or self.task is not None:
            return
        editor = self.layer_fill.editor
        self.layer_fill = None
        self.stack.setCurrentWidget(self.editor)
        self.stack.removeWidget(editor)
        editor.deleteLater()
        self._refresh()

    def _append_fill_variant(self, result: tuple[FillCandidate, FillPreview] | None, *, already_applied=False):
        session = self.fill
        if session is None or self.closing:
            return
        if result is None:
            session.editor.set_message("Generation canceled.")
            return
        session.variants.append(result)
        session.editor.add_variant(result[1], already_applied=already_applied)

    def _fill_changed(self):
        session = self.fill
        if session is None:
            return
        session.variants.clear()
        if self.task is not None:
            return
        editor = session.editor
        mask, prompt, edit_alpha = editor.mask, editor.prompt, editor.edit_alpha
        if not editor.can_generate:
            return
        candidate = session.reusable(mask, prompt, edit_alpha)
        if candidate is not None:
            if session.saved_preview is not None and ImageChops.difference(mask, session.saved_mask).getbbox() is None:
                self._append_fill_variant((candidate, session.saved_preview), already_applied=True)
                return

            def render(*_):
                return candidate, render_fill(session.image, mask, candidate, edit_alpha=edit_alpha)

            self._launch("preview_fill", render, self._append_fill_variant, message="Preparing saved variation…")

    def _generate_fill(self):
        layered = self.layer_fill
        session = layered or self.fill
        if session is None or self.task is not None:
            return
        editor = session.editor
        if not editor.can_generate:
            return
        image, mask = editor.generation_input()
        prompt, edit_alpha = editor.prompt, editor.edit_alpha
        target = layered.target if layered is not None and editor.to_active_layer else None
        executable = self._codex_resolver()

        def generate(_update, _pause, cancel):
            candidate = generate_fill(image, mask, prompt, edit_alpha, executable(), cancelled=cancel)
            preview = render_fill(image, mask, candidate, edit_alpha=edit_alpha)
            result = (candidate, preview) if layered is None else layered_variant(layered.document, mask, preview, target)
            if cancel.is_set():
                raise Cancelled("Generation canceled.")
            return result

        self._launch("generate_fill", generate, self._append_fill_variant if layered is None else self._append_layer_variant,
                     message="Generating variation…")

    def _cancel_fill(self):
        if self.task is not None and self.operation in ("generate_fill", "detect_remove", "generate_remove"):
            self._request_cancel()

    def _apply_fill(self):
        session = self.fill
        if session is None or self.task is not None:
            return
        editor = session.editor
        index = editor.selected_variant
        if not editor.can_apply or index is None or not 0 <= index < len(session.variants):
            return
        mask, prompt, edit_alpha = editor.mask, editor.prompt, editor.edit_alpha
        candidate, preview = session.variants[index]

        def save(_update, _pause, cancel):
            image = preview.compose(session.image)
            picture = qimage(np.asarray(image))
            grid = suggested_grid(*image.size)
            source = self.store.save_fill(
                session.source, image, mask, candidate.image, prompt=prompt, edit_alpha=edit_alpha,
                metadata={**candidate.metadata, "preview_warning": preview.warning}, base_sha256=session.sha256,
                derived_from=session.previous.source.id if session.previous else None, cancelled=cancel.is_set,
            )
            sprite = image if image.mode == "RGBA" else None
            return source, picture, grid, sprite, None

        def show(loaded):
            if loaded is not None and not self.closing:
                self._close_fill()
                self._show_source(loaded)

        self._launch("apply_fill", save, show, message="Saving image version…")

    def _close_fill(self):
        editor = self.fill.editor
        self.fill = None
        self.stack.setCurrentWidget(self.comparison)
        self.stack.removeWidget(editor)
        editor.deleteLater()
        self.sidebar.show()

    def _discard_fill(self):
        if self.fill is not None and self.task is None:
            self._close_fill()
            self._refresh()

    def _export_fill(self):
        edit = self.applied_fill
        if edit is None or self.task is not None or self.fill is not None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export source PNG", edit.source.name, "PNG (*.png)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        destination = Path(path)
        add_suffix = destination.suffix.lower() != ".png"
        if add_suffix:
            destination = destination.with_name(destination.name + ".png")
        try:
            self.store.check_export_destination(destination)
        except (OSError, ValueError) as exc:
            QMessageBox.information(self, "Export source PNG", str(exc))
            return
        if add_suffix and destination.exists() and QMessageBox.question(
            self, "Export source PNG", f"Replace {destination.name}?",
        ) != QMessageBox.StandardButton.Yes:
            return

        def save(_update, _pause, cancel):
            def before_publish(_path):
                if cancel.is_set():
                    raise Cancelled("Export canceled.")
                self.store.check_export_destination(destination)

            with edit.source.path.open("rb") as source, atomic_output(destination, before_publish=before_publish) as out:
                copyfileobj(source, out)
            return destination

        self._launch("export_fill", save, self._export_finished, message="Exporting source PNG…", cancellable=True)

    def _can_open(self):
        return self.task is None and self.fill is None and self.layer_fill is None

    def _choose_image(self):
        if not self._can_open():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image or project", "",
            "Images and projects (*.png *.jpeg *.jpg *.webp *.regimagen)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._import(Path(path))

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if self._can_open() and len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if self._can_open() and len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()
            self._import(Path(urls[0].toLocalFile()))

    def _confirm_replace(self, proceed):
        if not self.editor.dirty:
            return True
        result = QMessageBox.question(
            self, "Unsaved project", "Save your changes before leaving this document?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if result == QMessageBox.StandardButton.Save:
            self._save_project(after_save=proceed)
        return result == QMessageBox.StandardButton.Discard

    def _new_document(self):
        if not self._can_open():
            return
        size = dimensions(self, "New canvas", (1920, 1080))
        if size is None:
            return

        def create():
            self._show_document(Document("Untitled", *size))

        if self._confirm_replace(create):
            create()

    def _import(self, path: Path):
        if not self._can_open():
            return
        project = path.resolve() if path.suffix.lower() == PROJECT_SUFFIX else None

        def load(_update, _pause, cancel):
            return load_project(project, cancelled=cancel.is_set) if project else Document.from_image(load_source(path), path.name)

        def show(document):
            self._show_document(document, project)
            if not self.closing:
                self._set_feedback(f"Opened {path.name}")

        def open_document():
            message = "Opening project…" if project else "Opening image…"
            self._launch("open_document", load, show, message=message)

        if self._confirm_replace(open_document):
            open_document()

    def _show_document(self, document: Document | None, path: Path | None = None):
        if document is None or self.closing:
            return
        self._clear_idle_feedback()
        self.project_path = path
        self.stack.setCurrentWidget(self.editor)
        self.editor.set_document(document)
        if path is not None:
            self._remember_project(path)
        self._history()
        self._refresh()

    def _recent_projects(self):
        return [
            Path(path) for path in self.settings.value("recentDocuments", [], list)
            if Path(path).is_file()
        ]

    def _remember_project(self, path: Path):
        recent = [str(other) for other in self._recent_projects() if other != path]
        self.settings.setValue("recentDocuments", [str(path), *recent[:9]])

    def _import_layer(self):
        if not self._can_open() or self.stack.currentWidget() is not self.editor:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import image as a layer", "", "Images (*.png *.jpeg *.jpg *.webp)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            source = Path(path)

            def show(image):
                if image is not None and not self.closing:
                    self.editor.add_image(image, source.name)
                    self._set_feedback(f"Imported {source.name} as a layer.")

            self._launch("import_layer", lambda *_: load_source(source), show, message="Importing layer…")

    def _save_destination(self, title, filename, formats):
        dialog = QFileDialog(self, title)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(list(formats))
        dialog.setDefaultSuffix(next(iter(formats.values()))[0])
        dialog.filterSelected.connect(lambda label: dialog.setDefaultSuffix(formats[label][0]))
        dialog.selectFile(filename)
        if not dialog.exec():
            return None
        destination = Path(dialog.selectedFiles()[0])
        if destination.suffix.lower().lstrip(".") not in {suffix for group in formats.values() for suffix in group}:
            QMessageBox.information(self, title, "Choose a filename with a supported extension.")
            return None
        try:
            self.store.check_export_destination(destination)
        except (OSError, ValueError) as exc:
            QMessageBox.information(self, title, str(exc))
            return None
        return destination

    def _save_project(self, *, save_as=False, after_save=None):
        if self.task is not None:
            return
        document = self.editor.document
        if document is None:
            return
        destination = self.project_path
        if save_as or destination is None:
            destination = self._save_destination(
                "Save project", Path(document.name).stem + PROJECT_SUFFIX,
                {"Regimagen project (*.regimagen)": (PROJECT_SUFFIX[1:],)},
            )
        if destination is None:
            return

        def save(_update, _pause, cancel):
            save_project(
                document, destination, cancelled=cancel.is_set,
                before_publish=lambda _path: self.store.check_export_destination(destination),
            )
            return destination.resolve()

        def saved(path):
            self.project_path = path
            self._remember_project(path)
            self.editor.mark_saved(document)
            self._set_feedback(f"Saved {path.name}", folder=path.parent)
            self._history()
            self.after_task = after_save

        self._launch("save_project", save, saved, message="Saving project…", cancellable=True)

    def _export_document(self):
        if not self._can_open() or self.stack.currentWidget() is not self.editor:
            return
        document = self.editor.document
        if document is None:
            return
        name = (self.project_path or Path(document.name)).stem
        destination = self._save_destination("Export image", f"{name}.png", {
            "PNG (*.png)": ("png",), "JPEG (*.jpg *.jpeg)": ("jpg", "jpeg"), "WebP (*.webp)": ("webp",),
        })
        if destination is None:
            return
        suffix = destination.suffix.lower()
        image_format = "JPEG" if suffix in (".jpg", ".jpeg") else suffix[1:].upper()
        label = f"Exporting {image_format}"

        def save(update, _pause, cancel):
            export_document(
                document, destination, cancelled=cancel.is_set, progress=lambda done, total: update((done, total)),
                before_publish=lambda _path: self.store.check_export_destination(destination),
            )
            return destination

        self._launch(
            "export_document", save, self._export_finished, lambda counts: self._file_progress(label, counts),
            message=f"{label}…", cancellable=True,
        )

    def _reconstruct(self):
        if not self._can_open() or self.stack.currentWidget() is not self.editor:
            return
        document = self.editor.document
        if document is None:
            return
        name = (self.project_path.stem + ".png") if self.project_path else document.name

        def load(_update, _pause, cancel):
            image = render(document, cancelled=cancel.is_set)
            source = self.store.document_source(document.id, image, name, cancelled=cancel.is_set)
            sprite = image if image.getchannel("A").getextrema() != (255, 255) else None
            job = next((item for item in self.store.recent_jobs(source_id=source.id) if item.scale == 0), None)
            if job:
                refresh_thumbnails(self.store, job, alpha=sprite, cancelled=cancel.is_set)
            grid = job.grid if job else suggested_grid(*image.size)
            return source, qimage(np.asarray(image)), grid, sprite, job

        self._launch("prepare_reconstruction", load, self._show_source, message="Preparing reconstruction…")

    def _return_to_editor(self):
        if not self._can_open():
            return
        if self.editor.document is None:
            self._edit_input()
        else:
            self.stack.setCurrentWidget(self.editor)
            self._history()
            self._refresh()

    def _edit_input(self):
        if self.source is None or not self._can_open() or self.stack.currentWidget() is not self.comparison:
            return
        source, job = self.source, self.job

        def load(*_):
            image = load_job_input(job) if job else load_source(source.path)
            return Document.from_image(image, source.name)

        def open_document():
            self._launch("open_document", load, self._show_document, message="Opening image…")

        if self._confirm_replace(open_document):
            open_document()

    def _open_result(self):
        if self.job is None or not self._can_open() or self.stack.currentWidget() is not self.comparison:
            return
        job, artifacts, alpha = self.job, self._snapshot().current_artifacts, self.sprite
        if len(artifacts) != job.grid.count:
            return
        name = f"{Path(job.source.name).stem} — Reconstruction.png"

        def load(update, _pause, cancel):
            with TemporaryDirectory(prefix="regimagen-reconstruction-") as folder:
                path = Path(folder) / "reconstruction.png"
                export_png(
                    job.layout, artifacts, path, alpha=alpha, cancelled=cancel.is_set,
                    progress=lambda done, total: update((done, total)),
                )
                return Document.from_image(load_source(path), name)

        def open_document():
            self._launch(
                "open_result", load, self._show_document, lambda counts: self._file_progress("Rendering result", counts),
                message="Rendering result…", cancellable=True,
            )

        if self._confirm_replace(open_document):
            open_document()

    def _show_source(self, loaded):
        if loaded is None or self.closing:
            return
        self.source, image, grid, sprite, job = loaded
        self.applied_fill = self.store.get_fill(self.source)
        self._set_input(image, sprite)
        self.stack.setCurrentWidget(self.comparison)
        self._show_grid(grid, job)

    def _set_input(self, image, sprite):
        self.sprite = sprite
        self.original = QPixmap.fromImage(image)
        self.source_label.setText(f"{number(image.width())} × {number(image.height())} px")

    def _grid_changed(self):
        if self.job and self.job.input_revision:
            source, grid = self.source, self.grid

            def load(*_):
                image = load_source(source.path)
                sprite = image if image.mode == "RGBA" else None
                return source, qimage(np.asarray(image)), grid, sprite, None

            self._launch("import", load, self._show_source, message="Opening image…")
        else:
            self._show_grid(self.grid)

    def _show_grid(self, grid: Grid, job: Job | None = None):
        with QSignalBlocker(self.columns), QSignalBlocker(self.rows):
            self.columns.setValue(grid.columns)
            self.rows.setValue(grid.rows)
        if self.source is None:
            return
        self._clear_idle_feedback()
        self.job = job or self.store.find_job(self.source, grid)
        input_scale = self.job.input_scale if self.job else 1
        self.source_label.setToolTip(
            f"{self.source.name} · {input_scale}× result" if input_scale > 1 else self.source.name
        )
        layout = self.image_layout
        self.selected = None
        self.export_message = ""
        self.comparison.set_source(layout, self.original, self.sprite)
        self._history()
        if self.job is not None and self.task is None:
            job, sprite = self.job, self.sprite
            snapshot = self.store.snapshot(job)
            if any(tile.artifact and tile.thumbnail is None for tile in snapshot.tiles.values()):
                def refresh(_update, _pause, cancel):
                    refresh_thumbnails(self.store, job, alpha=sprite, cancelled=cancel.is_set)

                self._launch("refresh_preview", refresh, message="Refreshing preview…")
                return
        self._refresh()

    def _edit_cell_prompt(self):
        if self.selected is not None:
            self._edit_prompt(self.selected)

    def _edit_prompt(self, cell: Cell | None = None):
        layout = self.image_layout
        if layout is None or self.task is not None:
            return
        snapshot = self._snapshot()
        if cell is None:
            title = "Global prompt"
            label = "Additional instructions for all tiles in this job."
            existing = snapshot.prompt
        else:
            title = cell.label
            label = "Additional instructions for this tile. They take priority over the global prompt."
            existing = snapshot.tiles.get(cell, Tile()).prompt
        text, accepted = QInputDialog.getMultiLineText(self, title, label, existing)
        if accepted and text.strip() != existing:
            self.job = self.job or self.store.ensure_job(self.source, layout.grid)
            self.store.set_prompt(self.job, text.strip(), cell)
            self.export_message = ""
            self._history()
            self._refresh()

    def _snapshot(self) -> Snapshot:
        if self.job:
            return self.store.snapshot(self.job)
        return Snapshot({})

    def _select(self, cell: Cell):
        if self.image_layout is None:
            return
        self.selected = cell
        self.comparison.select(cell, focus=True)
        self._refresh()

    def _refresh(self):
        busy = self.task is not None
        page = self.stack.currentWidget()
        comparing = page is self.comparison
        editing = page is self.editor
        session = self.layer_fill or self.fill
        available = not busy and session is None
        for action in (self.open_button, self.open_action, self.new_action, self.empty_button, self.new_button):
            action.setEnabled(available)
        for action in (self.save_action, self.save_as_action, self.export_action):
            action.setEnabled(available and editing)
        self.history.setEnabled(available and self.history.count() > 1)
        self.settings_button.setEnabled(available)
        self.version_button.setVisible(comparing and self.applied_fill is not None)
        self.version_button.setEnabled(available)
        self.back_button.setVisible(comparing)
        self.back_button.setEnabled(available)
        self.back_button.setText("Back to editor" if self.editor.document is not None else "Edit input")
        self.sidebar.setVisible(comparing)
        self.editor.set_busy(busy or not editing)
        document = self.editor.document
        if editing and document is not None:
            title = self.project_path.name if self.project_path else document.name
            mark = " *" if self.editor.dirty else ""
            self.setWindowTitle(f"{title}{mark} — Regimagen")
            self.source_label.setText(f"{number(document.width)} × {number(document.height)} px")
            self.source_label.setToolTip(str(self.project_path or document.name))
        elif comparing:
            self.setWindowTitle("Reconstruction — Regimagen")
            self.source_label.setText(f"{number(self.original.width())} × {number(self.original.height())} px")
        if session is not None:
            if isinstance(session.editor, RemoveEditor):
                session.editor.set_busy(
                    busy, generating=self.operation == "generate_remove", detecting=self.operation == "detect_remove",
                )
            else:
                session.editor.set_busy(busy, generating=self.operation == "generate_fill")
            return
        if not comparing:
            return
        self.comparison.fill_button.setEnabled(self.source is not None and not busy)
        layout = self.image_layout
        snapshot = self._snapshot()
        ready = len(snapshot.current_artifacts)
        review = sum(tile.warning is not None and not tile.stale for tile in snapshot.tiles.values())
        count = layout.grid.count if layout else 0
        valid_grid = layout is not None and not layout.grid_error
        valid = valid_grid and layout.enlarges
        self.subagents.setEnabled(not busy)
        self.general_prompt.setEnabled(valid_grid and not busy)
        self.general_prompt.setText("Global prompt set…" if snapshot.prompt else "Global prompt…")
        self.batch.setEnabled(self.source is not None and not busy)
        for field in (self.columns, self.rows):
            field.setEnabled(layout is not None and not busy)
        self.grid_map.setEnabled(layout is not None)
        self.grid_map.set_tiles(self.grid, snapshot.tiles, self.selected, self.active)
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000 * ready // max(count, 1))
        if layout:
            width, height = layout.size
            self.dimensions.setText(f"{number(width)} × {number(height)} px")
        else:
            self.dimensions.setText("—")
        variant = f"{layout.scale}× scale" if layout and layout.scale else "PNG"
        if layout and layout.input_scale > 1:
            variant += f" from {layout.input_scale}×"
        self.edits.setText(f"{count} tiles · {variant}" if layout else "")
        self.export.setEnabled(bool(layout and ready == count and not busy))
        self.edit_result.setEnabled(self.export.isEnabled())
        self.parts.setEnabled(ready > 0 and not busy)
        if self.operation in ("generate", "scale_batch"):
            self.generate.setText("Pause")
            self.generate.setEnabled(not self.task.pause.is_set())
        else:
            remaining = ready < count or any(tile.attempt for tile in snapshot.tiles.values())
            if count and ready == count and not remaining:
                self.generate.setText("Job complete")
            else:
                self.generate.setText("Resume job" if ready or snapshot.notice else "Generate image")
            self.generate.setEnabled(bool(valid and not busy and remaining))
        preview_cells = layout.preview(self.selected) if layout else ()
        preview_pending = pending_cells(snapshot, preview_cells) if self.job else preview_cells
        self.preview.setEnabled(bool(valid and not busy and preview_pending))
        self.preview.setText("Preview ready" if preview_cells and not preview_pending else "Generate 2×2 preview")
        self.preview.setToolTip("Generate four neighboring tiles near the selection or the center of the grid.")
        tile = snapshot.tiles.get(self.selected, Tile()) if self.selected is not None else None
        self.cell_prompt.setEnabled(valid_grid and tile is not None and not busy)
        self.cell_prompt.setText("Tile prompt set…" if tile and tile.prompt else "Tile prompt…")
        self.retry.setText("Regenerate tile" if tile and tile.artifact else "Generate tile")
        self.retry.setEnabled(bool(valid and tile and not busy))
        self.tile_title.setText(self.selected.label if self.selected else "Tile")
        if tile:
            self.tile_note.setText(tile.error or (tile.warning if not tile.stale else None) or tile_state(tile)[0])
        else:
            self.tile_note.setText("Select a tile in the grid to inspect it.")
        self.status.setText(self._status_text(layout, snapshot, ready, review))
        if layout:
            self.comparison.update_tiles(snapshot)

    def _status_text(self, layout: Layout | None, snapshot: Snapshot, ready: int, review: int) -> str:
        if self.operation == "import":
            return "Opening image…"
        if self.operation == "open_fill":
            return "Opening Generative Fill…"
        if self.operation == "refresh_preview":
            return "Refreshing preview…"
        if layout is None:
            return "Open an image to configure the grid."
        if layout.grid_error:
            return layout.grid_error
        if not layout.enlarges:
            suggestion = suggested_grid(layout.source_width, layout.source_height)
            return f"This grid is too small to enlarge the image. Use {suggestion.columns}×{suggestion.rows}."
        if self.operation in ("generate", "scale_batch") and self.task.pause.is_set():
            return "Pausing after the active edits finish…"
        if self.active is not None:
            if self.subagents.isChecked():
                return f"Generating with subagents · {ready}/{layout.grid.count} ready"
            return f"Editing · {self.active.label.lower()} · {ready}/{layout.grid.count} ready"
        if snapshot.notice:
            notice = snapshot.notice
            if snapshot.quota_reset:
                timestamp = datetime.fromtimestamp(snapshot.quota_reset)
                notice += f" Resets at {timestamp:%Y-%m-%d %H:%M}."
            return f"{notice}\n{self.export_message}" if self.export_message else notice
        if self.export_message:
            return self.export_message
        return f"{ready}/{layout.grid.count} ready" + (f" · {review} to review" if review else "")

    def _choose_codex(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Codex CLI", "", "Codex CLI (codex.exe)", options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.settings.setValue("codex", path)
            self.settings_button.setToolTip(path)

    def _reset_codex(self):
        self.settings.remove("codex")
        self.settings_button.setToolTip("Detect the Codex CLI automatically")

    def _generate_or_pause(self):
        if self.operation in ("generate", "scale_batch") and self.task is not None:
            self.task.pause.set()
            self._refresh()
        elif self.image_layout:
            self._start(self.image_layout.cells)

    def _start_preview(self):
        if self.image_layout:
            self._start(self.image_layout.preview(self.selected))

    def _retry(self):
        if self.selected is not None:
            self._start((self.selected,), replace=True)

    def _start(self, selection, *, replace=False):
        layout = self.image_layout
        if self.task is not None or layout is None or layout.grid_error or not layout.enlarges:
            return
        job = self.job or self.store.ensure_job(self.source, layout.grid)
        self.job = job
        cells = list(selection) if replace else pending_cells(self.store.snapshot(job), selection)
        if not cells:
            return
        self.export_message = ""
        self._history()
        use_subagents = self.subagents.isChecked()
        executable = self._codex_resolver()

        def generate(update, pause, cancel):
            run_tiles(
                self.store, job, cells, executable, pause=pause, cancelled=cancel,
                on_change=update, subagents=use_subagents,
            )

        self._launch("generate", generate, updated=self._generation_update, message="Generating tiles…")

    def _codex_resolver(self) -> Callable[[], Path]:
        custom = self.settings.value("codex", "", str)

        def resolve():
            executable = Path(custom) if custom else find_codex()
            if executable is None or not executable.is_file():
                raise CodexError("Install the Codex CLI or select codex.exe from the Codex CLI menu.")
            return executable

        return resolve

    def _start_batch(self):
        source = self.source
        if source is None or self.task is not None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Scale batch")
        dialog.setMinimumWidth(400)
        form = QVBoxLayout(dialog)
        form.addWidget(QLabel("Enter whole-number scales of 2× or higher"))
        default = ", ".join(f"{scale}×" for scale in DEFAULT_SCALES)
        field = QLineEdit(self.settings.value("scales", default, str))
        field.setAccessibleName("Scales")
        field.setPlaceholderText("2×, 3×, 4×, 5×")
        field.setMaxLength(500)
        form.addWidget(field)
        input_label = QLabel("Input image")
        mode = QComboBox()
        mode.addItem("Original image", False)
        mode.addItem("Previous stage result", True)
        mode.setCurrentIndex(int(self.settings.value("chainedScales", False, bool)))
        mode.setAccessibleName(input_label.text())
        input_label.setBuddy(mode)
        mode.setToolTip("All scales are relative to the original image.")
        form.addWidget(input_label)
        form.addWidget(mode)
        note = QLabel("The global prompt applies to every scale in this batch.")
        note.setWordWrap(True)
        form.addWidget(note)
        plan = QLabel()
        plan.setWordWrap(True)
        form.addWidget(plan)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        start = actions.button(QDialogButtonBox.StandardButton.Ok)
        start.setText("Choose folder…")
        form.addWidget(actions)
        actions.accepted.connect(dialog.accept)
        actions.rejected.connect(dialog.reject)
        layouts = []

        def update_plan():
            layouts.clear()
            try:
                layouts.extend(plan_scales(source.width, source.height, field.text(), chained=mode.currentData()))
            except ValueError as exc:
                layouts.clear()
                plan.setText(str(exc))
            else:
                plan.setText(
                    f"{len(layouts)} PNG files · up to {sum(item.grid.count for item in layouts)} edits\n"
                    + "\n".join(
                        f"{item.scale}× · {number(item.size[0])} × {number(item.size[1])} px"
                        f" · {item.grid.columns}×{item.grid.rows} grid"
                        + (f" · {item.input_scale}× input" if item.input_scale > 1 else " · original input")
                        for item in layouts[:8]
                    )
                )
            start.setEnabled(bool(layouts))

        field.textChanged.connect(update_plan)
        mode.currentIndexChanged.connect(update_plan)
        update_plan()
        if not dialog.exec():
            return
        parent = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the batch PNGs",
            options=QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog,
        )
        if not parent:
            return
        self.settings.setValue("scales", field.text())
        self.settings.setValue("chainedScales", mode.currentData())
        use_subagents = self.subagents.isChecked()
        executable = self._codex_resolver()
        prompt = self._snapshot().prompt
        displayed_input = self.job.input_path if self.job else source.path
        self.export_message = ""

        def generate(update, pause, cancel):
            input_path = displayed_input

            def stage(job, image):
                nonlocal input_path
                if job.input_path == input_path:
                    update((job, None, None))
                else:
                    input_path = job.input_path
                    sprite = image if image.mode == "RGBA" else None
                    update((job, qimage(np.asarray(image)), sprite))

            return run_scale_batch(
                self.store, source, layouts, executable, Path(parent),
                pause=pause, cancelled=cancel, prompt=prompt, subagents=use_subagents,
                on_stage=stage, on_change=update, on_export=lambda scale: update(f"Exporting {scale}× PNG…"),
            )

        self._launch("scale_batch", generate, self._batch_finished, self._batch_update, message="Preparing scale batch…")

    def _batch_update(self, update):
        if isinstance(update, tuple):
            job, image, sprite = update
            if image is not None:
                self._set_input(image, sprite)
            self._show_grid(job.grid, job)
        elif isinstance(update, str):
            self.status.setText(update)
        else:
            self._generation_update(update)

    def _batch_finished(self, result):
        if result is not None:
            folder, completed, count = result
            self.export_message = f"Exported {completed}/{count} PNGs · {folder}"
            self.status.setToolTip(str(folder))
        self._history()

    def _generation_update(self, active: Cell | None):
        self.active = active
        self._refresh()

    def _export(self, *, parts=False):
        job = self.job
        if self.task is not None or job is None:
            return
        snapshot = self.store.snapshot(job)
        extension = "zip" if parts else "png"
        title = "Export tiles (ZIP)" if parts else "Export PNG"
        variant = f"{job.scale}x" if job.scale else f"{job.grid.columns}x{job.grid.rows}"
        suffix = "-tiles" if parts else ""
        filename = f"{Path(job.source.name).stem}-{variant}{suffix}.{extension}"
        destination = self._save_destination(title, filename, {f"{extension.upper()} (*.{extension})": (extension,)})
        if destination is None:
            return
        alpha = self.sprite
        label = "Exporting tiles" if parts else "Exporting PNG"

        def save(update, _pause, cancel):
            exporter = export_parts if parts else export_png
            exporter(
                job.layout, snapshot.current_artifacts, destination,
                progress=lambda done, total: update((done, total)),
                cancelled=cancel.is_set,
                alpha=alpha,
                before_publish=lambda _path: self.store.check_export_destination(destination),
            )
            return destination

        self._launch(
            "export_tiles" if parts else "export_image", save, self._export_finished,
            lambda counts: self._file_progress(label, counts), message=f"{label}…", cancellable=True,
        )

    def _file_progress(self, label: str, counts: tuple[int, int]):
        done, total = counts
        if 0 <= done < total:
            self._report_progress(f"{label} · {100 * done // total}%", counts)
        else:
            self._report_progress(f"{label} · Finalizing…")

    def _export_finished(self, destination: Path):
        self._set_feedback(f"Exported {destination.name}", folder=destination.parent)

    def _set_feedback(
        self, message: str = "", *, busy: bool = False, progress: tuple[int, int] | None = None,
        folder: Path | None = None,
    ):
        self.task_status.setText(message)
        if progress is not None:
            done, total = progress
            self.task_progress.setRange(0, total)
            self.task_progress.setValue(done)
        elif busy:
            self.task_progress.setRange(0, 0)
        self.task_progress.setVisible(busy)
        self.cancel_task.setVisible(busy and self.can_cancel_task)
        self.cancel_task.setEnabled(busy and self.task is not None and not self.task.cancelled.is_set())
        self.output_folder = folder
        self.open_folder.setToolTip(str(folder) if folder is not None else "")
        self.open_folder.setVisible(folder is not None)

    def _report_progress(self, message: str, progress: tuple[int, int] | None = None):
        if self.task is not None and not (self.task.pause.is_set() or self.task.cancelled.is_set()):
            self._set_feedback(message, busy=True, progress=progress)

    def _clear_idle_feedback(self):
        if self.task is None:
            self._set_feedback()

    def _open_output_folder(self):
        folder = self.output_folder
        if folder is not None:
            url = QUrl.fromLocalFile(str(folder.absolute()))
            if not folder.is_dir() or not QDesktopServices.openUrl(url):
                self._set_feedback(f"Could not open the folder · {folder}", folder=folder)

    def _launch(
        self, name: str, operation, succeeded=None, updated=None, *, message: str,
        progress: tuple[int, int] | None = None, cancellable: bool = False,
    ):
        def complete(result):
            self._set_feedback()
            if succeeded is not None:
                succeeded(result)

        self.operation = name
        self.can_cancel_task = cancellable
        self.task = Task(operation, self)
        self.task.succeeded.connect(complete)
        if updated is not None:
            self.task.updated.connect(updated)
        self.task.failed.connect(self._show_failure)
        self.task.finished.connect(self._task_finished)
        session = self.layer_fill or self.fill
        if session is not None:
            session.editor.set_message("")
        self._set_feedback(message, busy=True, progress=progress)
        self._refresh()
        self.task.start()

    def _show_failure(self, error: Exception):
        detail = str(error)
        if isinstance(error, Cancelled):
            message = detail or "Canceled."
        elif isinstance(error, AuthRequired):
            message = "Codex sign-in required. Run codex login, then try again."
        elif isinstance(error, QuotaPaused):
            message = quota_message(error.resets_at, "Try again")
        elif self.operation == "save_project" or self.operation.startswith("export_"):
            prefix = "Could not save the project." if self.operation == "save_project" else "Export failed."
            recovery = " Check the destination or choose another folder." if isinstance(error, OSError) else ""
            message = f"{prefix} {detail}{recovery}"
        elif self.operation in ("import", "open_document", "open_result"):
            message = f"Could not open the file. {detail}"
        else:
            session = self.layer_fill or self.fill
            if session is not None:
                session.editor.set_message(detail)
                message = "Object detection failed." if self.operation == "detect_remove" else "Could not complete the edit."
            else:
                prefix = {
                    "generate": "Generation failed.",
                    "scale_batch": "Scale batch failed.",
                    "prepare_reconstruction": "Could not prepare reconstruction.",
                    "refresh_preview": "Could not refresh the preview.",
                    "import_layer": "Could not import the layer.",
                    "open_fill": "Could not open the generative editor.",
                }.get(self.operation, "Could not complete the operation.")
                message = f"{prefix} {detail}"
        self._set_feedback(message)

    def _request_cancel(self):
        if self.task is not None and not self.task.cancelled.is_set():
            self.task.cancelled.set()
            self._set_feedback("Cancelling…", busy=True)
            self._refresh()

    def _task_finished(self):
        self.task.deleteLater()
        self.task = None
        self.operation = ""
        self.active = None
        after_task, self.after_task = self.after_task, None
        self._refresh()
        if self.closing:
            self.closing = False
            self.close()
        elif after_task is not None:
            after_task()

    def closeEvent(self, event):
        if self.task is not None:
            self.closing = True
            self._request_cancel()
            event.ignore()
            return
        if not self._confirm_replace(self.close):
            event.ignore()
            return
        super().closeEvent(event)


STYLESHEET = """
QWidget { color: #edf1f7; font-size: 13px; }
QWidget#root { background: #161921; }
QFrame#header { background: #1c2029; border-bottom: 1px solid #303644; }
QFrame#taskFooter { background: #1c2029; border-top: 1px solid #303644; }
QLabel#brand { font-size: 20px; font-weight: 700; }
QLabel#section { font-size: 14px; font-weight: 650; }
QLabel#emptyTitle { font-size: 26px; font-weight: 650; }
QLabel#dimensions { font-size: 22px; font-weight: 650; }
QLabel#muted { color: #aeb8ca; }
QFrame#emptyPanel, QFrame#imagePanel {
    background: #1b1f28; border: 1px solid #313747; border-radius: 10px;
}
QLabel#imageTitle { color: #d8e1ef; font-weight: 600; padding: 10px 12px; }
QFrame#divider { background: #343a49; }
QFrame#editorSidebar { background: #1e232d; border: 1px solid #313747; border-radius: 10px; }
QListWidget { background: #161921; border: 1px solid #313747; border-radius: 6px; }
QListWidget::item { padding: 7px 5px; }
QListWidget::item:selected { background: #344d4c; }
QScrollArea#sidebar { background: #1e232d; border: 1px solid #313747; border-radius: 10px; }
QWidget#sidebarContent { background: #1e232d; }
QPushButton, QComboBox, QSpinBox {
    background: #272d39; border: 1px solid #3c4659; border-radius: 6px;
    padding: 6px 10px; min-height: 22px;
}
QPushButton:hover, QComboBox:hover, QSpinBox:hover { background: #323a4a; }
QPushButton:focus, QComboBox:focus, QSpinBox:focus { border-color: #8ce1c6; }
QPushButton#openOutputFolder { padding: 3px 10px; min-height: 18px; }
QPushButton[primary="true"] { color: #102722; background: #8ce1c6; border-color: #8ce1c6; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #a5ecd6; }
QPushButton:disabled, QPushButton[primary="true"]:disabled, QSpinBox:disabled, QComboBox:disabled {
    color: #8b94a4; background: #252a35; border-color: #303646;
}
QProgressBar { background: #343a49; border: 0; border-radius: 3px; }
QProgressBar::chunk { background: #80d5ba; border-radius: 3px; }
QSplitter::handle { background: transparent; width: 8px; }
QMenu, QComboBox QAbstractItemView { background: #272d39; color: #edf1f7; selection-background-color: #3e4a61; }
QToolTip { color: #edf1f7; background: #272d39; border: 1px solid #48566d; }
QMessageBox { background: #252a35; }
"""


def apply_style(app: QApplication):
    app.setStyle("Fusion")
    font = app.font()
    QFontDatabase.addApplicationFont(str(Path(__file__).with_name("assets") / "Inter.ttf"))
    families = QFontDatabase.families()
    if "Inter" in families:
        font.setFamily("Inter")
    elif "Segoe UI" in families:
        font.setFamily("Segoe UI")
    app.setFont(font)
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#1c2029",
        QPalette.ColorRole.WindowText: "#edf1f7",
        QPalette.ColorRole.Base: "#272d39",
        QPalette.ColorRole.Text: "#edf1f7",
        QPalette.ColorRole.Button: "#272d39",
        QPalette.ColorRole.ButtonText: "#edf1f7",
        QPalette.ColorRole.Highlight: "#4e887c",
        QPalette.ColorRole.HighlightedText: "#ffffff",
    }.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
