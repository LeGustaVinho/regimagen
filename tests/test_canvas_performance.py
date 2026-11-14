from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest

from regimagen.canvas import CanvasView


def test_brush_preview_reuses_overlay_without_darkening_overlaps(qt_app):
    base = QPixmap(160, 120)
    base.fill(Qt.GlobalColor.black)
    canvas = CanvasView(base)
    canvas.resize(400, 300)
    canvas.set_tool("brush")
    canvas.set_brush_preview(QColor(255, 0, 0, 128), 12)
    canvas.show()
    try:
        qt_app.processEvents()
        viewport = canvas.viewport()

        def color_at(point):
            image = viewport.grab().toImage()
            return image.pixelColor(point * image.devicePixelRatio())

        first, last = (canvas.mapFromScene(QPointF(x, 60)) for x in (20, 110))
        middle = canvas.mapFromScene(QPointF(70, 60))
        left = Qt.MouseButton.LeftButton
        QTest.mousePress(viewport, left, pos=first)
        QTest.mouseMove(viewport, last)
        canvas._show_stroke()
        preview = canvas._brush_stroke
        image = preview.image
        first_pass = color_at(middle)
        assert abs(first_pass.red() - 128) <= 1
        QTest.mouseMove(viewport, first)
        canvas._show_stroke()
        assert preview.image is image
        assert color_at(middle) == first_pass

        canvas.zoom(1.25, viewport.rect().center())
        assert preview.image is not image
        middle = canvas.mapFromScene(QPointF(70, 60))
        assert color_at(middle) == first_pass
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
        assert canvas._brush_stroke is None
        assert color_at(middle) == QColor(Qt.GlobalColor.black)
    finally:
        canvas.close()
