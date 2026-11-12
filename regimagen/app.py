"""Regimagen's main window."""

from datetime import datetime
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, QSignalBlocker, QSize, QThread, Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QStackedWidget,
    QToolTip, QVBoxLayout, QWidget,
)

from .codex import Cancelled, find_codex
from .composite import export_parts, export_png
from .files import load_source
from .geometry import (
    MAX_COLUMNS, MAX_ROWS, MIN_COLUMNS, MIN_ROWS, Cell, Grid, Layout,
    plan_scales, scale_limit, suggested_grid,
)
from .jobs import pending_cells, refresh_thumbnails, run_scale_batch, run_tiles
from .storage import Job, Snapshot, Source, Store, Tile
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
    failed = Signal(str)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.pause = Event()
        self.cancelled = Event()

    def run(self):
        try:
            self.succeeded.emit(self.operation(self.updated.emit, self.pause, self.cancelled))
        except Cancelled:
            self.succeeded.emit(None)
        except Exception as exc:
            self.failed.emit(str(exc))


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


class TileMap(QWidget):
    picked = Signal(object)

    def __init__(self):
        super().__init__()
        self.grid = Grid(2, 2)
        self.tiles: dict[Cell, Tile] = {}
        self.selection: Cell | None = None
        self.active: Cell | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Tile grid")
        self.setToolTip("Select a tile. Use the arrow keys to navigate the grid.")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QSize(280, self.grid.rows * 21 + 5)

    def set_tiles(self, grid: Grid, tiles: dict[Cell, Tile], selection: Cell | None, active: Cell | None):
        self.grid, self.tiles, self.selection, self.active = grid, tiles, selection, active
        self.updateGeometry()
        self.update()

    def _rect(self, cell: Cell) -> QRectF:
        width = (self.width() - 6) / self.grid.columns
        return QRectF(3 + cell.column * width, 3 + cell.row * 21, width - 3, 18)

    def _at(self, position: QPointF) -> Cell | None:
        for row in range(self.grid.rows):
            for col in range(self.grid.columns):
                cell = Cell(col, row)
                if self._rect(cell).contains(position):
                    return cell
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for row in range(self.grid.rows):
            for col in range(self.grid.columns):
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
        self.selected: Cell | None = None
        self.active: Cell | None = None
        self.task: Task | None = None
        self.operation = ""
        self.closing = False
        self.export_message = ""
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
        self.open_button = QPushButton("Open image")
        self.open_button.setShortcut("Ctrl+O")
        self.open_button.clicked.connect(self._choose_image)
        toolbar.addWidget(self.open_button)
        self.history = QComboBox()
        self.history.setMinimumContentsLength(17)
        self.history.setMaximumWidth(255)
        self.history.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.history.currentIndexChanged.connect(self._load_history)
        toolbar.addWidget(self.history)
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
        self.stack.addWidget(self.comparison)
        workspace.addWidget(self.stack, 1)
        workspace.addWidget(self._sidebar())
        shell.addLayout(workspace, 1)

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
        title = QLabel("Open an image")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(title)
        help_text = QLabel("Drag a PNG, JPEG, or WebP here.")
        help_text.setObjectName("muted")
        help_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(help_text)
        column.addSpacing(14)
        button = QPushButton("Choose file")
        button.setProperty("primary", True)
        button.clicked.connect(self._choose_image)
        column.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.empty_button = button
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
        self.columns = self._field(fields, "Columns", MIN_COLUMNS, MAX_COLUMNS)
        self.rows = self._field(fields, "Rows", MIN_ROWS, MAX_ROWS)
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
        self.parts = QPushButton("Export tiles (ZIP)")
        self.parts.clicked.connect(lambda: self._export(parts=True))
        column.addWidget(self.parts)
        return pane

    def _field(self, row, title: str, minimum: int, maximum: int) -> QSpinBox:
        field = QVBoxLayout()
        name = QLabel(title)
        name.setObjectName("muted")
        field.addWidget(name)
        value = QSpinBox()
        value.setRange(minimum, maximum)
        value.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        value.setAccessibleName(title)
        value.valueChanged.connect(self._grid_changed)
        field.addWidget(value)
        row.addLayout(field)
        return value

    def _history(self):
        with QSignalBlocker(self.history):
            self.history.clear()
            self.history.addItem("Saved projects", None)
            for item in self.store.recent_jobs():
                scale = f" · {item.scale}×" if item.scale else ""
                if item.input_scale > 1:
                    scale += f" from {item.input_scale}×"
                self.history.addItem(f"{item.source.name} · {item.grid.columns}×{item.grid.rows}{scale}", item)
                if self.job and self.job.id == item.id:
                    self.history.setCurrentIndex(self.history.count() - 1)
            self.history.setToolTip(self.history.currentText())

    def _load_history(self, index: int):
        item: Job | None = self.history.itemData(index)
        if item is None or self.task is not None:
            return
        def load(_update, _pause, cancel):
            image = load_source(item.input_path)
            sprite = image if image.mode == "RGBA" else None
            refresh_thumbnails(self.store, item, alpha=sprite, cancelled=cancel.is_set)
            return item.source, qimage(np.asarray(image)), item.grid, sprite, item

        self._launch("import", load, self._show_source)

    def _choose_image(self):
        if self.task is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpeg *.jpg *.webp)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._import(Path(path))

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if self.task is None and len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if self.task is None and len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()
            self._import(Path(urls[0].toLocalFile()))

    def _import(self, path: Path):
        def load(*_):
            image = load_source(path)
            source = self.store.import_source(image, path.name)
            sprite = image if image.mode == "RGBA" else None
            return source, qimage(np.asarray(image)), suggested_grid(*image.size), sprite, None

        self._launch("import", load, self._show_source)

    def _show_source(self, loaded):
        if loaded is None:
            return
        self.source, image, grid, sprite, job = loaded
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

            self._launch("import", load, self._show_source)
        else:
            self._show_grid(self.grid)

    def _show_grid(self, grid: Grid, job: Job | None = None):
        with QSignalBlocker(self.columns), QSignalBlocker(self.rows):
            self.columns.setValue(grid.columns)
            self.rows.setValue(grid.rows)
        if self.source is None:
            return
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

                self._launch("refresh_preview", refresh)
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
            existing = snapshot.tiles[cell].prompt
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
        layout = self.image_layout
        return Snapshot({cell: Tile() for cell in layout.cells} if layout else {})

    def _select(self, cell: Cell):
        if self.image_layout is None:
            return
        self.selected = cell
        self.comparison.select(cell, focus=True)
        self._refresh()

    def _refresh(self):
        layout = self.image_layout
        snapshot = self._snapshot()
        busy = self.task is not None
        ready = len(snapshot.current_artifacts)
        review = sum(tile.warning is not None and not tile.stale for tile in snapshot.tiles.values())
        count = layout.grid.count if layout else 0
        valid = layout is not None and layout.enlarges
        self.open_button.setEnabled(not busy)
        self.empty_button.setEnabled(not busy)
        self.history.setEnabled(not busy and self.history.count() > 1)
        self.settings_button.setEnabled(not busy)
        self.subagents.setEnabled(not busy)
        self.general_prompt.setEnabled(layout is not None and not busy)
        self.general_prompt.setText("Global prompt set…" if snapshot.prompt else "Global prompt…")
        self.batch.setEnabled(bool(
            self.source and not busy and scale_limit(self.source.width, self.source.height) >= 2
        ))
        for field in (self.columns, self.rows):
            field.setEnabled(layout is not None and not busy)
        self.grid_map.setEnabled(layout is not None)
        self.grid_map.set_tiles(self.grid, snapshot.tiles, self.selected, self.active)
        self.progress.setRange(0, max(count, 1))
        if self.operation not in ("export_image", "export_tiles"):
            self.progress.setValue(ready)
        if layout:
            width, height = layout.size
            self.dimensions.setText(f"{number(width)} × {number(height)} px")
        else:
            self.dimensions.setText("—")
        variant = f"{layout.scale}× scale" if layout and layout.scale else "PNG"
        if layout and layout.input_scale > 1:
            variant += f" from {layout.input_scale}×"
        self.edits.setText(f"{count} edits · {variant}" if layout else "")
        self.export.setEnabled(bool(layout and ready == count and not busy))
        self.parts.setEnabled(ready > 0 and not busy)
        if self.operation in ("generate", "scale_batch"):
            self.generate.setText("Pause")
            self.generate.setEnabled(not self.task.pause.is_set())
        else:
            remaining = pending_cells(snapshot, layout.cells) if layout else ()
            if count and ready == count and not remaining:
                self.generate.setText("Job complete")
            else:
                self.generate.setText("Resume job" if ready or snapshot.notice else "Generate image")
            self.generate.setEnabled(bool(valid and not busy and remaining))
        preview_cells = layout.preview(self.selected) if layout else ()
        preview_pending = pending_cells(snapshot, preview_cells)
        self.preview.setEnabled(bool(valid and not busy and preview_pending))
        self.preview.setText("Preview ready" if preview_cells and not preview_pending else "Generate 2×2 preview")
        self.preview.setToolTip("Generate four neighboring tiles near the selection or the center of the grid.")
        tile = snapshot.tiles.get(self.selected)
        self.cell_prompt.setEnabled(tile is not None and not busy)
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
        if self.operation == "refresh_preview":
            return "Refreshing preview…"
        if self.operation in ("export_image", "export_tiles"):
            return self.export_message or ("Preparing tile archive…" if self.operation == "export_tiles" else "Assembling PNG…")
        if layout is None:
            return "Open an image to configure the grid."
        if not layout.enlarges:
            suggestion = suggested_grid(layout.source_width, layout.source_height)
            if Layout(layout.source_width, layout.source_height, suggestion).enlarges:
                return f"This grid is too small to enlarge the image. Use {suggestion.columns}×{suggestion.rows}."
            return "The image exceeds the available resolution for this aspect ratio."
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
        if self.task is not None or layout is None or not layout.enlarges:
            return
        executable = self._codex_executable()
        if executable is None:
            return
        job = self.job or self.store.ensure_job(self.source, layout.grid)
        self.job = job
        cells = list(selection) if replace else pending_cells(self.store.snapshot(job), selection)
        if not cells:
            return
        self.export_message = ""
        self._history()
        use_subagents = self.subagents.isChecked()

        def generate(update, pause, cancel):
            run_tiles(
                self.store, job, cells, executable, pause=pause, cancelled=cancel,
                on_change=update, subagents=use_subagents,
            )

        self._launch("generate", generate, updated=self._generation_update)

    def _codex_executable(self) -> Path | None:
        custom = self.settings.value("codex", "", str)
        executable = Path(custom) if custom else find_codex()
        if executable is None or not executable.is_file():
            QMessageBox.information(
                self, "Codex CLI", "Install the Codex CLI or select codex.exe from the Codex CLI menu.",
            )
            return None
        return executable

    def _start_batch(self):
        source = self.source
        if source is None or self.task is not None:
            return
        limit = scale_limit(source.width, source.height)
        dialog = QDialog(self)
        dialog.setWindowTitle("Scale batch")
        dialog.setMinimumWidth(400)
        form = QVBoxLayout(dialog)
        form.addWidget(QLabel(f"Scales from 2× to {limit}× for this image"))
        default = ", ".join(f"{scale}×" for scale in range(2, min(limit, 5) + 1))
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
        executable = self._codex_executable()
        if executable is None:
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

        self._launch("scale_batch", generate, self._batch_finished, self._batch_update)

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
        dialog = QFileDialog(self, title)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("ZIP (*.zip)" if parts else "PNG (*.png)")
        dialog.setDefaultSuffix(extension)
        variant = f"{job.scale}x" if job.scale else f"{job.grid.columns}x{job.grid.rows}"
        suffix = "-tiles" if parts else ""
        dialog.selectFile(f"{Path(job.source.name).stem}-{variant}{suffix}.{extension}")
        if not dialog.exec():
            return
        destination = Path(dialog.selectedFiles()[0])
        if destination.suffix.lower() != f".{extension}":
            QMessageBox.information(self, title, f"Choose a filename ending in .{extension}.")
            return
        self.export_message = ""
        alpha = self.sprite

        def save(update, _pause, cancel):
            exporter = export_parts if parts else export_png
            exporter(
                job.layout, snapshot.current_artifacts, destination,
                progress=lambda done, total: update((done, total)),
                cancelled=cancel.is_set,
                alpha=alpha,
            )
            return destination

        self._launch("export_tiles" if parts else "export_image", save, self._export_finished, self._export_update)

    def _export_update(self, state: tuple[int, int]):
        done, total = state
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        kind = "tiles" if self.operation == "export_tiles" else "PNG"
        self.export_message = f"Exporting {kind} · {100 * done // total}%"
        self.status.setText(self.export_message)

    def _export_finished(self, destination: Path | None):
        self.export_message = f"Exported {destination.name}" if destination else ""
        if destination:
            self.status.setToolTip(str(destination))

    def _launch(self, name: str, operation, succeeded=None, updated=None):
        self.operation = name
        self.task = Task(operation, self)
        if succeeded is not None:
            self.task.succeeded.connect(succeeded)
        if updated is not None:
            self.task.updated.connect(updated)
        self.task.failed.connect(self._show_failure)
        self.task.finished.connect(self._task_finished)
        self._refresh()
        self.task.start()

    def _show_failure(self, message: str):
        if not self.closing:
            QMessageBox.warning(self, "Regimagen", message)

    def _task_finished(self):
        self.task.deleteLater()
        self.task = None
        self.operation = ""
        self.active = None
        if self.closing:
            self.close()
        else:
            self._refresh()

    def closeEvent(self, event):
        if self.task is not None:
            self.closing = True
            self.task.cancelled.set()
            event.ignore()
            return
        super().closeEvent(event)


STYLESHEET = """
QWidget { color: #edf1f7; font-size: 13px; }
QWidget#root { background: #161921; }
QFrame#header { background: #1c2029; border-bottom: 1px solid #303644; }
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
QScrollArea#sidebar { background: #1e232d; border: 1px solid #313747; border-radius: 10px; }
QWidget#sidebarContent { background: #1e232d; }
QPushButton, QComboBox, QSpinBox {
    background: #272d39; border: 1px solid #3c4659; border-radius: 6px;
    padding: 6px 10px; min-height: 22px;
}
QPushButton:hover, QComboBox:hover, QSpinBox:hover { background: #323a4a; }
QPushButton:focus, QComboBox:focus, QSpinBox:focus { border-color: #8ce1c6; }
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
