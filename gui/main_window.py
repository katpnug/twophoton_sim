"""
main_window.py
==============
Main application window for the 2P TIFF Simulator.

Layout:
  ┌─────────────────────────────────────────────────────────────┐
  │  Header bar (title + Run / Save buttons)                    │
  ├────────────────┬────────────────────────────────────────────┤
  │                │                                            │
  │  Param Panel   │         Visualization Panel                │
  │  (scrollable)  │    (tabbed: frames / traces / neuropil…)   │
  │                │                                            │
  ├────────────────┴────────────────────────────────────────────┤
  │  Status bar + progress                                      │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog,
    QSplitter, QMessageBox, QFrame,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QIcon

from gui.param_panel import ParamPanel
from gui.viz_panel import VizPanel
from gui.sim_worker import SimWorker
from core.stack_simulator import SimResult, save_tiff
from core.export_data import export_sim_data

APP_DARK = "#0a0f1c"
PANEL_BG = "#131929"
ACCENT   = "#4a7ecf"
ACCENT2  = "#2a9d8f"
TEXT     = "#c8daff"
SUBTEXT  = "#6888b0"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2P TIFF Simulator")
        self.resize(1400, 880)
        self._worker: SimWorker | None = None
        self._result: SimResult | None = None
        self._setup_ui()
        self._apply_stylesheet()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_body(), stretch=1)
        root.addWidget(self._build_statusbar())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setStyleSheet(f"background: {PANEL_BG}; border-bottom: 1px solid #1e2d4a;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        # Logo / title
        title = QLabel("⬡  2P TIFF Simulator")
        title.setFont(QFont("Courier New", 15, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT}; letter-spacing: 1px;")
        layout.addWidget(title)

        subtitle = QLabel("")
        subtitle.setStyleSheet(f"color: {SUBTEXT}; font-size: 11px;")
        layout.addWidget(subtitle)

        layout.addStretch()

        # Buttons
        self._load_params_btn = self._make_btn("Load Params", "#34495e", self._on_load_params)
        self._save_params_btn = self._make_btn("Save Params", "#34495e", self._on_save_params)
        self._run_btn = self._make_btn("▶  Run Simulation", ACCENT, self._on_run)
        self._stop_btn = self._make_btn("■  Stop", "#c0392b", self._on_stop)
        self._save_btn = self._make_btn("💾  Save TIFF", ACCENT2, self._on_save)
        self._data_btn = self._make_btn("Save Data", "#6c5ce7", self._on_export_data)
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._data_btn.setEnabled(False)

        layout.addWidget(self._load_params_btn)
        layout.addWidget(self._save_params_btn)
        layout.addWidget(self._run_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._save_btn)
        layout.addWidget(self._data_btn)

        return header

    def _make_btn(self, text, color, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        btn.setFixedHeight(34)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0 18px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: {color}cc; }}
            QPushButton:disabled {{ background: #2a3040; color: #445; }}
        """)
        return btn

    def _build_body(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #1e2d4a; }")

        self._param_panel = ParamPanel()
        self._viz_panel = VizPanel()

        splitter.addWidget(self._param_panel)
        splitter.addWidget(self._viz_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        return splitter

    def _build_statusbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet(f"background: {PANEL_BG}; border-top: 1px solid #1e2d4a;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)

        self._status_lbl = QLabel("Ready. Configure parameters and click Run.")
        self._status_lbl.setStyleSheet(f"color: {SUBTEXT}; font-size: 11px;")
        layout.addWidget(self._status_lbl)

        layout.addStretch()

        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet(f"color: {SUBTEXT}; font-size: 11px;")
        layout.addWidget(self._info_lbl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(200)
        self._progress.setFixedHeight(16)
        self._progress.setVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar {
                background: #1a2035;
                border: 1px solid #2a4070;
                border-radius: 8px;
                text-align: center;
                color: #6888b0;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2a6abf, stop:1 #4a9eff);
                border-radius: 7px;
            }
        """)
        layout.addWidget(self._progress)

        return bar

    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {APP_DARK}; }}
            QWidget {{ color: {TEXT}; font-family: 'Segoe UI', 'SF Pro Display', sans-serif; }}
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
                background: #1a2035;
                border: 1px solid #2a4070;
                border-radius: 3px;
                color: {TEXT};
                padding: 2px 4px;
                min-width: 70px;
            }}
            QDoubleSpinBox, QSpinBox {{
                padding-right: 24px;
                min-height: 22px;
            }}
            QDoubleSpinBox::up-button, QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px;
                border-left: 1px solid #2a4070;
                border-bottom: 1px solid #18243c;
                background: #223556;
            }}
            QDoubleSpinBox::down-button, QSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px;
                border-left: 1px solid #2a4070;
                background: #223556;
            }}
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
                background: #253554;
            }}
            QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed,
            QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{
                background: {ACCENT};
            }}
            QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
                border: 1px solid {ACCENT};
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: #1a2035;
                selection-background-color: #2a5090;
                color: {TEXT};
            }}
            QCheckBox {{ color: #8da4c8; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid #2a4070;
                border-radius: 3px;
                background: #1a2035;
            }}
            QCheckBox::indicator:checked {{ background: {ACCENT}; }}
            QLabel {{ color: #8da4c8; }}
            QTabWidget::pane {{ background: {APP_DARK}; }}
            QScrollBar:vertical {{
                background: {APP_DARK}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #2a4070; border-radius: 4px;
            }}
        """)

    # ── Simulation control ────────────────────────────────────────────────────

    def _on_load_params(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameter Preset", "",
            "JSON parameter presets (*.json);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                preset = json.load(fh)
            self._param_panel.apply_preset_dict(preset)
            self._status_lbl.setText(f"Loaded parameters: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Load Parameter Error", str(e))
            self._status_lbl.setText("Parameter load failed.")

    def _on_save_params(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameter Preset", "twophoton_params.json",
            "JSON parameter presets (*.json);;All files (*)"
        )
        if not path:
            return
        if Path(path).suffix.lower() != ".json":
            path = f"{path}.json"

        try:
            preset = self._param_panel.to_preset_dict()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(preset, fh, indent=2)
                fh.write("\n")
            self._status_lbl.setText(f"Saved parameters: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Parameter Error", str(e))
            self._status_lbl.setText("Parameter save failed.")

    def _on_run(self):
        if self._worker and self._worker.isRunning():
            return

        try:
            spike_opts = self._param_panel.get_spike_opts()
            calcium_params = self._param_panel.get_calcium_params()
            sim_params = self._param_panel.get_sim_params()
        except Exception as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return

        self._result = None
        self._save_btn.setEnabled(False)
        self._data_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_lbl.setText("Initialising simulation…")

        self._worker = SimWorker(spike_opts, calcium_params, sim_params)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.abort()
        self._set_idle("Simulation stopped.")

    def _on_progress(self, current: int, total: int, stage: str):
        self._progress.setValue(current)
        self._status_lbl.setText(stage)

    def _on_frame_ready(self, idx: int, frame: np.ndarray):
        """Display live preview frame during rendering."""
        self._viz_panel.set_frame(frame, idx)

    def _on_finished(self, result: SimResult):
        self._result = result
        self._viz_panel.set_result(result)

        T = result.t[-1] if len(result.t) > 0 else 0
        n_active = len(result.active_idx)
        n_total = len(result.masks)
        info = (
            f"  {n_active} active / {n_total} total cells  •  "
            f"{result.movie_uint16.shape if result.movie_uint16 is not None else '—'} uint16  •  "
            f"max={result.uint16_max:.3f}  •  scale={result.uint16_scale:.1f}"
        )
        self._info_lbl.setText(info)
        self._save_btn.setEnabled(True)
        self._data_btn.setEnabled(True)
        self._set_idle(f"Simulation complete ({T:.1f}s recording, {n_total} cells).")

    def _on_error(self, msg: str):
        self._set_idle("Simulation failed.")
        QMessageBox.critical(self, "Simulation Error", msg)

    def _set_idle(self, status: str = "Ready."):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._status_lbl.setText(status)

    # ── Save ─────────────────────────────────────────────────────────────────

    def _on_save(self):
        if self._result is None or self._result.movie_uint16 is None:
            QMessageBox.information(self, "Nothing to Save",
                                    "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save TIFF Stack", "sim_2p.tif",
            "TIFF files (*.tif *.tiff);;All files (*)"
        )
        if not path:
            return

        self._status_lbl.setText(f"Saving {path}…")
        try:
            save_tiff(self._result.movie_uint16, path)
            self._status_lbl.setText(f"Saved: {Path(path).name}")
        except ImportError:
            QMessageBox.warning(
                self, "Missing Dependency",
                "tifffile is required to save TIFFs.\n\n"
                "Install it with:  pip install tifffile"
            )
            self._status_lbl.setText("Save failed (tifffile not found).")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            self._status_lbl.setText("Save failed.")

    def _on_export_data(self):
        if self._result is None:
            QMessageBox.information(self, "Nothing to Export",
                                    "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Simulation Data", "sim_2p_data.npz",
            "NumPy compressed (*.npz);;NumPy pickle (*.npy);;MATLAB (*.mat);;HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        if not path:
            return

        self._status_lbl.setText(f"Exporting {path}...")
        try:
            export_sim_data(self._result, path)
            self._status_lbl.setText(f"Exported: {Path(path).name}")
        except ImportError as e:
            QMessageBox.warning(self, "Missing Dependency", str(e))
            self._status_lbl.setText("Export failed (missing dependency).")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            self._status_lbl.setText("Export failed.")
