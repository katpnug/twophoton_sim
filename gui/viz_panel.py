"""
viz_panel.py
============
Right-side visualization panel:
  - Live movie frame viewer (with colormap, brightness, contrast)
  - Cell mask overlay
  - ΔF/F trace plot
  - Neuropil map
  - Motion XY trace
  - Mean fluorescence histogram
"""

from __future__ import annotations
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QSlider, QComboBox, QCheckBox, QSizePolicy,
    QFrame, QGroupBox, QScrollArea, QSpinBox, QPushButton, QApplication,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

import pyqtgraph as pg
import pyqtgraph.exporters

from core.stack_simulator import SimResult


# Configure pyqtgraph
pg.setConfigOptions(antialias=True, background="#0d1220", foreground="#8da4c8")


def _linear_colormap(name: str, colors: list[tuple[int, int, int]]) -> pg.ColorMap:
    """Create a tiny built-in colormap for environments missing pg map files."""
    pos = np.linspace(0.0, 1.0, len(colors))
    return pg.ColorMap(pos, np.array(colors, dtype=np.ubyte), name=name)


_CMAP_SPECS = {
    "Greys": ("gist_gray", [(0, 0, 0), (255, 255, 255)]),
    "Hot": (
        "inferno",
        [(0, 0, 4), (87, 15, 109), (187, 55, 84), (249, 142, 8), (252, 255, 164)],
    ),
    "Viridis": (
        "viridis",
        [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    ),
    "Plasma": (
        "plasma",
        [(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)],
    ),
    "Cyan": ("CET-C1", [(0, 0, 0), (0, 72, 96), (0, 170, 190), (210, 255, 255)]),
}
CMAP_NAMES = tuple(_CMAP_SPECS.keys())
_CMAP_CACHE: dict[str, pg.ColorMap] = {}


def _get_cmap(display_name: str) -> pg.ColorMap:
    pg_name, fallback = _CMAP_SPECS.get(display_name, _CMAP_SPECS["Greys"])
    if display_name not in _CMAP_CACHE:
        try:
            _CMAP_CACHE[display_name] = pg.colormap.get(pg_name)
        except (FileNotFoundError, OSError, KeyError, ValueError):
            _CMAP_CACHE[display_name] = _linear_colormap(display_name, fallback)
    return _CMAP_CACHE[display_name]


def _gray_rgba(norm: np.ndarray) -> np.ndarray:
    return np.stack([norm, norm, norm, np.ones_like(norm)], axis=-1).astype(np.float32)


def _colorize(norm: np.ndarray, cmap_name: str) -> np.ndarray:
    try:
        return _get_cmap(cmap_name).map(norm, mode="float")
    except Exception:
        return _gray_rgba(norm)

TRACE_COLORS = [
    "#4a9eff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8",
    "#ff922b", "#74c0fc", "#f783ac", "#94d82d", "#ffec99",
]


def _reset_button(slot) -> QPushButton:
    btn = QPushButton("Reset")
    btn.clicked.connect(slot)
    btn.setFixedHeight(24)
    btn.setStyleSheet("""
        QPushButton {
            background: #1e2433;
            color: #8da4c8;
            border: 1px solid #2a4070;
            border-radius: 3px;
            padding: 0 8px;
            font-size: 11px;
        }
        QPushButton:hover { background: #252d42; color: #c8daff; }
    """)
    return btn


def _rgba_from_hex(color: str, alpha: float) -> tuple[float, float, float, float]:
    qcolor = QColor(color)
    return (
        qcolor.redF(),
        qcolor.greenF(),
        qcolor.blueF(),
        float(np.clip(alpha, 0.0, 1.0)),
    )


class FrameViewer(QWidget):
    """Widget that displays a single 2D frame with optional overlays."""

    frame_changed = Signal(int)
    selection_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Controls bar ──────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self._cmap_cb = QComboBox()
        self._cmap_cb.addItems(list(CMAP_NAMES))
        self._cmap_cb.currentTextChanged.connect(self._refresh)
        self._cmap_cb.setStyleSheet("QComboBox { background: #1e2433; color: #a3c4f3; }")
        ctrl.addWidget(QLabel("Colormap:"))
        ctrl.addWidget(self._cmap_cb)

        self._auto_contrast_cb = QCheckBox("Auto contrast")
        self._auto_contrast_cb.setChecked(False)
        self._auto_contrast_cb.stateChanged.connect(self._refresh)
        ctrl.addWidget(self._auto_contrast_cb)

        self._overlay_cb = QCheckBox("Cell outlines")
        self._overlay_cb.setChecked(True)
        self._overlay_cb.stateChanged.connect(self._refresh)
        ctrl.addWidget(self._overlay_cb)

        ctrl.addWidget(QLabel("Outline:"))
        self._outline_source_cb = QComboBox()
        self._outline_source_cb.addItems(["Soma + axons", "Soma only", "Axons only"])
        self._outline_source_cb.currentIndexChanged.connect(self._build_outlines)
        self._outline_source_cb.setStyleSheet("QComboBox { background: #1e2433; color: #a3c4f3; }")
        ctrl.addWidget(self._outline_source_cb)

        ctrl.addWidget(QLabel("Width:"))
        self._outline_width = QSpinBox()
        self._outline_width.setRange(1, 8)
        self._outline_width.setValue(3)
        self._outline_width.valueChanged.connect(self._build_outlines)
        ctrl.addWidget(self._outline_width)

        self._fill_cb = QCheckBox("Fill")
        self._fill_cb.setChecked(False)
        self._fill_cb.stateChanged.connect(self._refresh)
        ctrl.addWidget(self._fill_cb)

        clear_btn = _reset_button(self._clear_selection)
        clear_btn.setText("Clear")
        ctrl.addWidget(clear_btn)

        reset_btn = _reset_button(self._reset_view)
        ctrl.addWidget(reset_btn)

        ctrl.addStretch()
        self._frame_lbl = QLabel("Frame: —")
        self._frame_lbl.setStyleSheet("color: #6888b0; font-size: 11px;")
        ctrl.addWidget(self._frame_lbl)

        layout.addLayout(ctrl)

        # ── pyqtgraph ImageItem ───────────────────────────────────────────
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setBackground("#0d1220")

        self._img_item = pg.ImageItem()
        self._plot.addItem(self._img_item)

        # Overlays for cell masks and outlines
        self._mask_overlay = pg.ImageItem()
        self._plot.addItem(self._mask_overlay)
        self._outline_items: list[pg.PlotCurveItem] = []
        self._selected_items: list[pg.ScatterPlotItem] = []
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        layout.addWidget(self._plot, stretch=1)

        # ── Frame slider ──────────────────────────────────────────────────
        slider_row = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_playback)
        self._play_btn.setFixedWidth(58)
        self._play_btn.setStyleSheet("""
            QPushButton {
                background: #1e2433;
                color: #8da4c8;
                border: 1px solid #2a4070;
                border-radius: 3px;
                padding: 0 8px;
                font-size: 11px;
            }
            QPushButton:checked { background: #4a7ecf; color: white; }
            QPushButton:disabled { background: #141a28; color: #445; }
        """)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #1e2433; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #4a7ecf; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #2a5090; border-radius: 2px; }
        """)
        slider_row.addWidget(QLabel("Frame:"))
        slider_row.addWidget(self._play_btn)
        slider_row.addWidget(self._slider, stretch=1)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        layout.addLayout(slider_row)

        self._frame: np.ndarray | None = None
        self._movie: np.ndarray | None = None   # (T, H, W) float
        self._masks: list[np.ndarray] | None = None
        self._soma_masks: list[np.ndarray] | None = None
        self._process_masks: list[np.ndarray] | None = None
        self._cell_xyr: np.ndarray | None = None
        self._result: SimResult | None = None
        self._current_frame_idx = 0
        self._mask_overlay_ready = False
        self._fixed_levels: tuple[float, float] | None = None
        self._selected_cells: set[int] = set()

    def _reset_view(self):
        self._plot.enableAutoRange()
        self._plot.autoRange()

    def _toggle_playback(self, checked: bool):
        if checked:
            if self._movie is None:
                self._play_btn.setChecked(False)
                return
            self._play_btn.setText("Stop")
            self._play_timer.start(self._play_interval_ms())
        else:
            self._play_timer.stop()
            self._play_btn.setText("Play")

    def _play_interval_ms(self) -> int:
        if self._result is not None and len(self._result.t) > 1:
            dt = float(np.median(np.diff(self._result.t)))
            return max(1, int(round(1000 * dt)))
        return 33

    def _advance_frame(self):
        if self._movie is None:
            self._play_btn.setChecked(False)
            self._toggle_playback(False)
            return
        next_idx = self._slider.value() + 1
        if next_idx > self._slider.maximum():
            next_idx = 0
        self._slider.setValue(next_idx)

    # ── Public API ────────────────────────────────────────────────────────

    def set_frame(self, frame: np.ndarray, frame_idx: int = 0):
        """Display a single frame (H, W) float array."""
        self._play_btn.setEnabled(False)
        self._frame = frame
        self._current_frame_idx = frame_idx
        self._frame_lbl.setText(f"Frame: {frame_idx}")
        self._refresh()

    def set_result(self, result: SimResult):
        """Load full result for scrubbing."""
        self._result = result
        self._movie = result.movie  # (H, W, T) float64
        self._masks = result.masks
        self._soma_masks = result.soma_masks
        self._process_masks = result.process_masks
        self._cell_xyr = result.cell_xyr
        self._selected_cells.clear()
        self.selection_changed.emit([])
        self._fixed_levels = None
        if self._movie is not None:
            lo, hi = np.percentile(self._movie, [1, 99.5])
            if hi <= lo:
                hi = lo + 1
            self._fixed_levels = (float(lo), float(hi))
        T = self._movie.shape[2] if self._movie is not None else 0
        self._slider.setRange(0, max(0, T - 1))
        self._play_btn.setEnabled(T > 1)
        self._play_timer.setInterval(self._play_interval_ms())
        self._slider.setValue(0)
        self._on_slider(0)
        self._build_outlines()

    def _on_slider(self, idx: int):
        if self._movie is not None:
            self._frame = self._movie[:, :, idx].astype(np.float32)
            self._current_frame_idx = idx
            self._frame_lbl.setText(f"Frame: {idx}")
            self._refresh()

    def _build_outlines(self):
        # Remove old outlines
        for item in self._outline_items:
            self._plot.removeItem(item)
        self._outline_items.clear()
        self._mask_overlay.clear()
        self._mask_overlay_ready = False

        outline_masks = self._outline_masks()
        if outline_masks is None or self._cell_xyr is None:
            return

        if outline_masks:
            H, W = outline_masks[0].shape
            overlay = np.zeros((W, H, 4), dtype=np.float32)
        else:
            overlay = None

        outline_width = self._outline_width.value()

        for i, mask in enumerate(outline_masks):
            # Extract boundary using simple gradient
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(mask)
            outline = mask & ~eroded
            ys, xs = np.where(outline)
            color = TRACE_COLORS[i % len(TRACE_COLORS)]
            if overlay is not None:
                rgba = _rgba_from_hex(color, 0.22)
                overlay[mask.T] = rgba
            if len(xs) == 0:
                continue
            item = pg.ScatterPlotItem(
                x=xs, y=ys,
                size=outline_width,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(color),
            )
            self._plot.addItem(item)
            self._outline_items.append(item)

        if overlay is not None:
            self._mask_overlay.setImage(overlay)
            self._mask_overlay_ready = True
            self._mask_overlay.setVisible(self._fill_cb.isChecked())
        self._refresh_selection_overlay()
        self._refresh()

    def _outline_masks(self) -> list[np.ndarray] | None:
        mode = self._outline_source_cb.currentText()
        if mode == "Soma only":
            return self._soma_masks if self._soma_masks is not None else self._masks
        if mode == "Axons only":
            return self._process_masks if self._process_masks is not None else self._masks
        return self._masks

    def _clear_selection(self):
        if not self._selected_cells:
            return
        self._selected_cells.clear()
        self._refresh_selection_overlay()
        self.selection_changed.emit([])

    def _on_plot_clicked(self, event):
        if event.button() != Qt.LeftButton or self._masks is None or self._cell_xyr is None:
            return
        pos = self._plot.plotItem.vb.mapSceneToView(event.scenePos())
        cell_idx = self._cell_at_pos(pos.x(), pos.y())
        if cell_idx is None:
            return

        multi_select = bool(QApplication.keyboardModifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
        if not multi_select:
            self._selected_cells = {cell_idx}
        elif cell_idx in self._selected_cells:
            self._selected_cells.remove(cell_idx)
        else:
            self._selected_cells.add(cell_idx)

        self._refresh_selection_overlay()
        self.selection_changed.emit(sorted(self._selected_cells))

    def _cell_at_pos(self, x: float, y: float) -> int | None:
        if self._masks is None or self._cell_xyr is None or not self._masks:
            return None
        H, W = self._masks[0].shape
        xi = int(round(x))
        yi = int(round(y))
        candidates = []
        if 0 <= xi < W and 0 <= yi < H:
            candidates = [i for i, mask in enumerate(self._masks) if mask[yi, xi]]
        if candidates:
            xy = self._cell_xyr[candidates, :2]
            d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
            return int(candidates[int(np.argmin(d))])

        xy = self._cell_xyr[:, :2]
        radii = self._cell_xyr[:, 2]
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        nearest = int(np.argmin(d))
        if d[nearest] <= radii[nearest] + 3:
            return nearest
        return None

    def _refresh_selection_overlay(self):
        for item in self._selected_items:
            self._plot.removeItem(item)
        self._selected_items.clear()
        outline_masks = self._outline_masks()
        if outline_masks is None:
            return
        size = max(5, self._outline_width.value() + 3)
        for idx in sorted(self._selected_cells):
            if idx < 0 or idx >= len(outline_masks):
                continue
            from scipy.ndimage import binary_erosion
            mask = outline_masks[idx]
            outline = mask & ~binary_erosion(mask)
            ys, xs = np.where(outline)
            if len(xs) == 0:
                continue
            item = pg.ScatterPlotItem(
                x=xs, y=ys,
                size=size,
                pen=pg.mkPen(None),
                brush=pg.mkBrush("#ffffff"),
            )
            self._plot.addItem(item)
            self._selected_items.append(item)

    def _refresh(self):
        if self._frame is None:
            return

        F = self._frame.T  # pyqtgraph: (W, H) for row-major display

        # Normalize. Fixed levels keep background/noise visually stable over time.
        if self._fixed_levels is not None and not self._auto_contrast_cb.isChecked():
            lo, hi = self._fixed_levels
        else:
            lo, hi = np.percentile(F, [1, 99.5])
            if hi <= lo:
                hi = lo + 1
        F_norm = np.clip((F - lo) / (hi - lo), 0, 1)

        # Apply colormap
        cmap_name = self._cmap_cb.currentText()
        colored = _colorize(F_norm, cmap_name)  # (W, H, 4) RGBA

        self._img_item.setImage(colored)

        # Toggle outlines
        show = self._overlay_cb.isChecked() and bool(self._outline_items)
        for item in self._outline_items:
            item.setVisible(show)
        self._mask_overlay.setVisible(self._fill_cb.isChecked() and self._mask_overlay_ready)


class TracePlot(QWidget):
    """ΔF/F trace viewer for a selected subset of cells."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Show cells:"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 500)
        self._n_spin.setValue(10)
        self._n_spin.setFixedWidth(58)
        self._n_spin.setAccelerated(True)
        self._n_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._n_spin.valueChanged.connect(self._refresh_and_fit)
        ctrl.addWidget(self._n_spin)
        step_col = QVBoxLayout()
        step_col.setContentsMargins(0, 0, 0, 0)
        step_col.setSpacing(1)
        self._n_up_btn = QPushButton("+")
        self._n_down_btn = QPushButton("-")
        for btn in (self._n_up_btn, self._n_down_btn):
            btn.setFixedSize(22, 14)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet("""
                QPushButton {
                    background: #223556;
                    color: #eaf2ff;
                    border: 1px solid #2a4070;
                    border-radius: 2px;
                    padding: 0;
                    font-size: 11px;
                    font-weight: 700;
                }
                QPushButton:hover { background: #253554; color: #ffffff; }
                QPushButton:pressed { background: #4a7ecf; color: white; }
                QPushButton:disabled { background: #141a28; color: #5d6c84; }
            """)
            step_col.addWidget(btn)
        self._n_up_btn.setToolTip("Show one more cell")
        self._n_down_btn.setToolTip("Show one fewer cell")
        self._n_up_btn.clicked.connect(lambda: self._step_show_cells(1))
        self._n_down_btn.clicked.connect(lambda: self._step_show_cells(-1))
        ctrl.addLayout(step_col)

        self._source_cb = QComboBox()
        self._source_cb.addItems([
            "dFF (extracted)",
            "F raw (L2 ROI)",
            "clean C (ground truth)",
            "spike raster",
        ])
        self._source_cb.currentIndexChanged.connect(self._refresh_and_fit)
        ctrl.addWidget(self._source_cb)

        self._sort_cb = QComboBox()
        self._sort_cb.addItems([
            "cell index", "selected order", "mod onset time", "first spike time",
            "activity gain", "baseline F0", "peak response",
        ])
        self._sort_cb.currentIndexChanged.connect(self._refresh_and_fit)
        ctrl.addWidget(self._sort_cb)

        self._selected_cb = QCheckBox("Selected only")
        self._selected_cb.setChecked(False)
        self._selected_cb.stateChanged.connect(self._refresh_and_fit)
        ctrl.addWidget(self._selected_cb)

        self._spike_overlay_cb = QCheckBox("Overlay spikes")
        self._spike_overlay_cb.setChecked(False)
        self._spike_overlay_cb.stateChanged.connect(self._refresh)
        ctrl.addWidget(self._spike_overlay_cb)

        reset_btn = _reset_button(self._reset_view)
        ctrl.addWidget(reset_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#0d1220")
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "ΔF/F  (offset)")
        layout.addWidget(self._plot)

        self._result: SimResult | None = None
        self._selected_cells: list[int] = []
        self._fit_next_refresh = True

    def set_result(self, result: SimResult):
        self._result = result
        self._selected_cells = []
        self._fit_next_refresh = True
        self._selected_cb.setChecked(False)
        self._refresh()

    def set_selected_cells(self, cells: list[int]):
        self._selected_cells = list(cells)
        self._fit_next_refresh = True
        if cells and not self._selected_cb.isChecked():
            self._selected_cb.setChecked(True)
        else:
            self._refresh()

    def _reset_view(self):
        self._plot.enableAutoRange()
        self._plot.autoRange()

    def _refresh_and_fit(self, *_args):
        self._fit_next_refresh = True
        self._refresh()

    def _step_show_cells(self, delta: int):
        self._sync_show_cells_limit(self._available_cell_count())
        next_value = int(np.clip(
            self._n_spin.value() + delta,
            self._n_spin.minimum(),
            self._n_spin.maximum(),
        ))
        self._n_spin.setValue(next_value)

    def _available_cell_count(self) -> int:
        if self._result is None:
            return self._n_spin.maximum()
        r = self._result
        if self._source_cb.currentText() == "spike raster":
            return len(r.spike_times or [])
        data, _label = self._trace_data_and_label(r)
        return int(data.shape[0]) if data is not None else self._n_spin.maximum()

    @staticmethod
    def _spike_tick_data(times: np.ndarray, y0: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
        if times is None or len(times) == 0:
            return np.array([]), np.array([])
        x = np.empty(3 * len(times), dtype=float)
        y = np.empty(3 * len(times), dtype=float)
        x[0::3] = times
        x[1::3] = times
        x[2::3] = np.nan
        y[0::3] = y0
        y[1::3] = y1
        y[2::3] = np.nan
        return x, y

    def _refresh(self):
        self._plot.clear()
        if self._result is None:
            return

        r = self._result
        t = r.t
        source_name = self._source_cb.currentText()
        spike_times = r.spike_times or []

        if source_name == "spike raster":
            if not spike_times:
                return
            self._sync_show_cells_limit(len(spike_times))
            order = self._cell_order(len(spike_times), None)
            n_show = min(self._n_spin.value(), len(order))
            for row, cell_idx in enumerate(order[:n_show]):
                color = TRACE_COLORS[cell_idx % len(TRACE_COLORS)]
                x, y = self._spike_tick_data(spike_times[cell_idx], row - 0.35, row + 0.35)
                if len(x):
                    self._plot.plot(x, y, pen=pg.mkPen(color, width=1.0))
            self._plot.setLabel("left", "Cell")
            self._plot.setYRange(-0.75, max(0.75, n_show - 0.25), padding=0.02)
            self._plot.setXRange(float(t[0]), float(t[-1]), padding=0.02)
            return

        data, y_label = self._trace_data_and_label(r)

        if data is None:
            return

        self._sync_show_cells_limit(data.shape[0])
        self._plot.setLabel("left", y_label)
        order = self._cell_order(data.shape[0], data)
        n_show = min(self._n_spin.value(), len(order))
        if n_show == 0:
            return
        offset_step = np.nanmax(np.abs(data[order[:n_show]])) * 1.5 + 0.1
        y_min = np.inf
        y_max = -np.inf

        for row, cell_idx in enumerate(order[:n_show]):
            trace = data[cell_idx]
            offset = row * offset_step
            trace_y = trace + offset
            finite_trace = trace_y[np.isfinite(trace_y)]
            if finite_trace.size:
                y_min = min(y_min, float(np.nanmin(finite_trace)))
                y_max = max(y_max, float(np.nanmax(finite_trace)))
            color = TRACE_COLORS[cell_idx % len(TRACE_COLORS)]
            self._plot.plot(t, trace_y,
                            pen=pg.mkPen(color, width=1.2),
                            name=f"Cell {cell_idx}")
            if self._spike_overlay_cb.isChecked() and cell_idx < len(spike_times):
                trace_floor = offset + float(np.nanmin(trace))
                lane_top = min(offset - 0.08 * offset_step, trace_floor - 0.05 * offset_step)
                lane_bottom = lane_top - 0.08 * offset_step
                x, y = self._spike_tick_data(
                    spike_times[cell_idx],
                    lane_bottom,
                    lane_top,
                )
                if len(x):
                    y_min = min(y_min, float(lane_bottom))
                    y_max = max(y_max, float(lane_top))
                    self._plot.plot(x, y, pen=pg.mkPen(color, width=0.7))

        if self._fit_next_refresh and np.isfinite(y_min) and np.isfinite(y_max):
            yr = max(y_max - y_min, 1e-6)
            self._plot.setYRange(y_min - 0.04 * yr, y_max + 0.04 * yr, padding=0)
            if len(t) > 1:
                self._plot.setXRange(float(t[0]), float(t[-1]), padding=0.02)
            self._fit_next_refresh = False

    def _trace_data_and_label(self, result: SimResult):
        source_name = self._source_cb.currentText()
        if source_name == "dFF (extracted)":
            return result.dFF if result.dFF is not None else result.C, "dF/F (offset)"
        if source_name == "F raw (L2 ROI)":
            return result.Fraw, "F raw, L2 ROI (offset)"
        return result.C, "Clean C (offset)"

    def _sync_show_cells_limit(self, n_cells: int):
        max_cells = max(1, int(n_cells))
        if self._n_spin.maximum() == max_cells:
            self._update_show_cell_buttons()
            return
        old_state = self._n_spin.blockSignals(True)
        self._n_spin.setMaximum(max_cells)
        if self._n_spin.value() > max_cells:
            self._n_spin.setValue(max_cells)
        self._update_show_cell_buttons()
        self._n_spin.blockSignals(old_state)

    def _update_show_cell_buttons(self):
        self._n_up_btn.setEnabled(self._n_spin.value() < self._n_spin.maximum())
        self._n_down_btn.setEnabled(self._n_spin.value() > self._n_spin.minimum())

    def _cell_order(self, n_cells: int, data: np.ndarray | None) -> list[int]:
        if self._selected_cb.isChecked():
            cells = [i for i in self._selected_cells if 0 <= i < n_cells]
        else:
            cells = list(range(n_cells))

        mode = self._sort_cb.currentText()
        r = self._result
        if mode == "selected order" and self._selected_cells:
            selected_rank = {cell: rank for rank, cell in enumerate(self._selected_cells)}
            return sorted(cells, key=lambda i: selected_rank.get(i, n_cells + i))
        if mode == "mod onset time" and r is not None and r.mod_windows:
            def mod_onset(i):
                vals = []
                for window in r.mod_windows:
                    if isinstance(window, dict) and "onsets" in window and i < len(window["onsets"]):
                        val = float(window["onsets"][i])
                        if np.isfinite(val):
                            vals.append(val)
                return min(vals) if vals else np.inf
            return sorted(cells, key=mod_onset)
        if mode == "first spike time" and r is not None and r.spike_times:
            def first_spike(i):
                if i >= len(r.spike_times) or len(r.spike_times[i]) == 0:
                    return np.inf
                return float(r.spike_times[i][0])
            return sorted(cells, key=first_spike)
        if mode == "activity gain" and r is not None and r.cell_gain_per_cell is not None:
            return sorted(cells, key=lambda i: -float(r.cell_gain_per_cell[i]) if i < len(r.cell_gain_per_cell) else np.inf)
        if mode == "baseline F0" and r is not None and r.cell_F0_per_cell is not None:
            return sorted(cells, key=lambda i: -float(r.cell_F0_per_cell[i]) if i < len(r.cell_F0_per_cell) else np.inf)
        if mode == "peak response" and data is not None:
            return sorted(cells, key=lambda i: -float(np.nanmax(data[i])))
        return cells


class NeuropilMapView(QWidget):
    """Shows the neuropil spatial map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Neuropil background field"))
        ctrl.addStretch()
        reset_btn = _reset_button(self._reset_view)
        ctrl.addWidget(reset_btn)
        layout.addLayout(ctrl)

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setBackground("#0d1220")
        self._img = pg.ImageItem()
        self._plot.addItem(self._img)
        layout.addWidget(self._plot)

    def _reset_view(self):
        self._plot.enableAutoRange()
        self._plot.autoRange()

    def set_result(self, result: SimResult):
        np_map = result.neuropil_base
        lo, hi = np_map.min(), np_map.max()
        if hi > lo:
            n = (np_map - lo) / (hi - lo)
        else:
            n = np_map
        colored = _colorize(n.T, "Hot")
        self._img.setImage(colored)


class MotionPlot(QWidget):
    """X/Y motion trace over time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Frame motion (px)"))
        ctrl.addStretch()
        reset_btn = _reset_button(self._reset_view)
        ctrl.addWidget(reset_btn)
        layout.addLayout(ctrl)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#0d1220")
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Displacement (px)")
        self._plot.addLegend()
        layout.addWidget(self._plot)

    def _reset_view(self):
        self._plot.enableAutoRange()
        self._plot.autoRange()

    def set_result(self, result: SimResult):
        self._plot.clear()
        t = result.t
        dx = result.motion_xy[:, 0]
        dy = result.motion_xy[:, 1]
        self._plot.plot(t, dx, pen=pg.mkPen("#4a9eff", width=1.5), name="dx")
        self._plot.plot(t, dy, pen=pg.mkPen("#ff6b6b", width=1.5), name="dy")


class MeanFramePlot(QWidget):
    """Mean projection and histogram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()

        # Mean image
        self._img_plot = pg.PlotWidget()
        self._img_plot.setAspectLocked(True)
        self._img_plot.setMouseEnabled(x=True, y=True)
        self._img_plot.hideAxis("left")
        self._img_plot.hideAxis("bottom")
        self._img_plot.setBackground("#0d1220")
        self._img_item = pg.ImageItem()
        self._img_plot.addItem(self._img_item)
        row.addWidget(self._img_plot, stretch=2)

        # Histogram
        self._hist_plot = pg.PlotWidget()
        self._hist_plot.setBackground("#0d1220")
        self._hist_plot.setMouseEnabled(x=True, y=True)
        self._hist_plot.showGrid(x=True, y=False, alpha=0.15)
        self._hist_plot.setLabel("bottom", "Intensity")
        self._hist_plot.setLabel("left", "Count")
        row.addWidget(self._hist_plot, stretch=1)

        layout.addLayout(row)

    def set_result(self, result: SimResult):
        if result.movie is None:
            return
        mean_frame = result.movie.mean(axis=2)
        lo, hi = np.percentile(mean_frame, [1, 99.5])
        n = np.clip((mean_frame - lo) / max(hi - lo, 1e-6), 0, 1)
        colored = _colorize(n.T, "Greys")
        self._img_item.setImage(colored)

        # Histogram
        self._hist_plot.clear()
        vals = mean_frame.ravel()
        counts, edges = np.histogram(vals, bins=64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self._hist_plot.plot(
            centers, counts,
            stepMode="left",
            fillLevel=0,
            fillBrush=pg.mkBrush("#4a7ecf40"),
            pen=pg.mkPen("#4a7ecf", width=1.5),
        )


class VizPanel(QWidget):
    """Tabbed visualization panel (right side)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #0d1220; }
            QTabBar::tab {
                background: #131929; color: #5a7499;
                padding: 6px 14px; font-size: 11px;
                border: none; border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected { color: #a3c4f3; border-bottom: 2px solid #4a7ecf; }
            QTabBar::tab:hover { color: #c8daff; }
        """)

        self.frame_viewer = FrameViewer()
        self.trace_plot = TracePlot()
        self.frame_viewer.selection_changed.connect(self.trace_plot.set_selected_cells)
        self.neuropil_view = NeuropilMapView()
        self.motion_plot = MotionPlot()
        self.mean_plot = MeanFramePlot()

        self._tabs.addTab(self.frame_viewer, "🎞  Frame Viewer")
        self._tabs.addTab(self.trace_plot,   "📈  ΔF/F Traces")
        self._tabs.addTab(self.mean_plot,    "📊  Mean + Histogram")
        self._tabs.addTab(self.neuropil_view,"🌊  Neuropil Map")
        self._tabs.addTab(self.motion_plot,  "📐  Motion")

        layout.addWidget(self._tabs)

    def set_frame(self, frame: np.ndarray, idx: int):
        self.frame_viewer.set_frame(frame, idx)

    def set_result(self, result: SimResult):
        self.frame_viewer.set_result(result)
        self.trace_plot.set_result(result)
        self.neuropil_view.set_result(result)
        self.motion_plot.set_result(result)
        self.mean_plot.set_result(result)
