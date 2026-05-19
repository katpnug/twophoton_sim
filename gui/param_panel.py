"""
param_panel.py
==============
Left-side collapsible parameter panel.
Returns SpikeGenOpts, CalciumParams, SimParams on demand.
"""

from __future__ import annotations
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGroupBox, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QHBoxLayout, QFrame, QPushButton, QSizePolicy, QLineEdit,
    QStyle, QStyleOptionSpinBox,
)
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QFont, QColor, QPalette, QPainter

from core.spike_generator import SpikeGenOpts, ModEpoch
from core.calcium_dynamics import CalciumParams
from core.stack_simulator import SimParams


class _ReliableSpinButtonMixin:
    """Make the styled right-side spin button column easy to hit."""

    _button_column_width = 24

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QColor("#eaf2ff"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        col_x = max(0, self.width() - self._button_column_width)
        half_h = self.height() // 2
        painter.drawText(QRect(col_x, 0, self._button_column_width, half_h),
                         Qt.AlignCenter, "+")
        painter.drawText(QRect(col_x, half_h, self._button_column_width, self.height() - half_h),
                         Qt.AlignCenter, "-")
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            opt = QStyleOptionSpinBox()
            self.initStyleOption(opt)
            hit = self.style().hitTestComplexControl(QStyle.CC_SpinBox, opt, pos, self)
            if hit in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
                super().mousePressEvent(event)
                return
            if pos.x() >= self.width() - self._button_column_width:
                self.stepBy(1 if pos.y() < self.height() / 2 else -1)
                event.accept()
                return
        super().mousePressEvent(event)


class ReliableDoubleSpinBox(_ReliableSpinButtonMixin, QDoubleSpinBox):
    pass


class ReliableSpinBox(_ReliableSpinButtonMixin, QSpinBox):
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dbl(min_val, max_val, step, default, decimals=3, suffix="") -> QDoubleSpinBox:
    w = ReliableDoubleSpinBox()
    w.setRange(min_val, max_val)
    w.setSingleStep(step)
    w.setValue(default)
    w.setDecimals(decimals)
    w.setMinimumHeight(26)
    if suffix:
        w.setSuffix(f" {suffix}")
    return w


def _int(min_val, max_val, default) -> QSpinBox:
    w = ReliableSpinBox()
    w.setRange(min_val, max_val)
    w.setValue(default)
    w.setMinimumHeight(26)
    return w


class CollapsibleGroup(QWidget):
    """A QGroupBox-like widget with a clickable title to collapse/expand."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._expanded = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self._btn = QPushButton(f"▾  {title}")
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.clicked.connect(self._toggle)
        self._btn.setStyleSheet("""
            QPushButton {
                background: #1e2433;
                color: #a3c4f3;
                border: none;
                border-left: 3px solid #4a7ecf;
                padding: 6px 10px;
                text-align: left;
                font-weight: 600;
                font-size: 12px;
                margin-top: 6px;
            }
            QPushButton:hover { background: #252d42; }
        """)
        layout.addWidget(self._btn)

        # Content container
        self._content = QWidget()
        self._content_layout = QFormLayout(self._content)
        self._content_layout.setContentsMargins(12, 6, 6, 6)
        self._content_layout.setSpacing(5)
        self._content_layout.setLabelAlignment(Qt.AlignRight)
        self._content_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addWidget(self._content)

    def _toggle(self, checked):
        self._expanded = checked
        self._content.setVisible(checked)
        arrow = "▾" if checked else "▸"
        old = self._btn.text()
        self._btn.setText(arrow + old[1:])

    def add_row(self, label: str, widget: QWidget):
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #8da4c8; font-size: 11px;")
        self._content_layout.addRow(lbl, widget)

    def content_layout(self):
        return self._content_layout


# ──────────────────────────────────────────────────────────────────────────────
# Main parameter panel
# ──────────────────────────────────────────────────────────────────────────────

class ParamPanel(QScrollArea):
    """Scrollable left panel with all simulation parameters."""

    params_changed = Signal()
    PARAM_FILE_VERSION = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(360)
        self.setMaximumWidth(460)
        self.setStyleSheet("""
            QScrollArea { background: #131929; border: none; }
            QScrollBar:vertical {
                background: #0d1220; width: 6px; border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #2a4070; border-radius: 3px;
            }
        """)

        container = QWidget()
        container.setStyleSheet("background: #131929;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 16)
        layout.setSpacing(2)

        self._build_recording_group(layout)
        self._build_spike_group(layout)
        self._build_calcium_group(layout)
        self._build_cell_group(layout)
        self._build_neuropil_group(layout)
        self._build_motion_group(layout)
        self._build_noise_group(layout)

        layout.addStretch()
        self.setWidget(container)

    # ── Group builders ────────────────────────────────────────────────────────

    def _build_recording_group(self, layout):
        g = CollapsibleGroup("📹  Recording")
        layout.addWidget(g)

        self.n_units = _int(1, 500, 100)
        self.n_inactive = _int(0, 500, 100)
        self.duration_s = _dbl(1, 600, 1, 30, decimals=0, suffix="s")
        self.img_h = _int(32, 1024, 128)
        self.img_w = _int(32, 1024, 256)
        self.fps = _dbl(1, 200, 1, 115, decimals=0, suffix="Hz")
        self.seed = _int(0, 99999, 42)

        g.add_row("N active neurons", self.n_units)
        g.add_row("N inactive neurons", self.n_inactive)
        g.add_row("Duration", self.duration_s)
        g.add_row("Image H (px)", self.img_h)
        g.add_row("Image W (px)", self.img_w)
        g.add_row("Frame rate", self.fps)
        g.add_row("Seed", self.seed)

    def _build_spike_group(self, layout):
        g = CollapsibleGroup("⚡  Spiking Activity")
        layout.addWidget(g)

        self.baseline_rate_min = _dbl(0, 100, 0.5, 0.5, suffix="Hz")
        self.baseline_rate_max = _dbl(0, 100, 0.5, 3.0, suffix="Hz")
        self.cv = _dbl(0.1, 5.0, 0.1, 1.0)
        self.burst_alpha = _dbl(0, 1, 0.05, 0.0)
        self.burst_tau = _dbl(0.001, 1.0, 0.005, 0.015, suffix="s")

        self.use_mod = QCheckBox("Enable epoch")
        self.use_mod.setChecked(True)
        self.mod_onsets = QLineEdit("1:5:30")
        self.mod_onsets.setMinimumWidth(150)
        self.mod_onsets.setPlaceholderText("e.g. 0:.5:10 or 0, 2.5, 5")
        self.mod_active_duration = _dbl(0.001, 600, 0.1, 0.25, suffix="s")
        self.mod_onset_jitter = _dbl(0, 60, 0.05, 0.1, suffix="s")
        self.mod_offset_jitter = _dbl(0, 60, 0.05, 0.0, suffix="s")
        self.mod_lock_onset_jitter = QCheckBox("Lock onset jitter per cell")
        self.mod_lock_onset_jitter.setChecked(False)
        self.mod_lock_duration = QCheckBox("Fixed duration after onset jitter")
        self.mod_lock_duration.setChecked(False)
        self.mod_peak_min = _dbl(0, 500, 5, 20.0, suffix="Hz")
        self.mod_peak_max = _dbl(0, 500, 5, 80.0, suffix="Hz")
        self.mod_profile = QComboBox()
        self.mod_profile.addItems(["box", "cosine"])

        g.add_row("Baseline rate min", self.baseline_rate_min)
        g.add_row("Baseline rate max", self.baseline_rate_max)
        g.add_row("ISI CV", self.cv)
        g.add_row("Burst α (Hawkes)", self.burst_alpha)
        g.add_row("Burst τ", self.burst_tau)
        g.add_row("", self.use_mod)
        g.add_row("Activity onsets", self.mod_onsets)
        g.add_row("Active Duration", self.mod_active_duration)
        g.add_row("Onset jitter", self.mod_onset_jitter)
        g.add_row("", self.mod_lock_onset_jitter)
        g.add_row("Offset jitter", self.mod_offset_jitter)
        g.add_row("", self.mod_lock_duration)
        g.add_row("Peak rate min", self.mod_peak_min)
        g.add_row("Peak rate max", self.mod_peak_max)
        g.add_row("Epoch profile", self.mod_profile)

        self.use_mod.toggled.connect(self._update_mod_visibility)
        self.mod_lock_duration.toggled.connect(self._update_mod_visibility)
        self._update_mod_visibility(self.use_mod.isChecked())

    def _update_mod_visibility(self, _checked=None):
        checked = self.use_mod.isChecked()
        for w in [self.mod_onsets, self.mod_active_duration, self.mod_onset_jitter,
                  self.mod_lock_onset_jitter, self.mod_lock_duration,
                  self.mod_peak_min, self.mod_peak_max, self.mod_profile]:
            w.setEnabled(checked)
        self.mod_offset_jitter.setEnabled(checked and not self.mod_lock_duration.isChecked())

    def _build_calcium_group(self, layout):
        g = CollapsibleGroup("🧬  Calcium Indicator")
        layout.addWidget(g)

        self.indicator_preset = QComboBox()
        self.indicator_preset.addItems(["GCaMP6f", "GCaMP6s", "GCaMP7f",
                                         "GCaMP8f", "GCaMP8m", "GCaMP8s",
                                         "Voltage (fast)", "Custom"])
        self.tau_rise = _dbl(0.001, 2.0, 0.001, 0.026, suffix="s")
        self.tau_decay = _dbl(0.001, 10.0, 0.005, 0.202, suffix="s")
        self.single_ap_amp = _dbl(0.01, 5.0, 0.05, 0.25)
        self.burst_tail_cb = QCheckBox("Prolong burst tail")
        self.burst_tail_cb.setChecked(False)
        self.burst_tail_window = _dbl(0.01, 5.0, 0.01, 0.20, suffix="s")
        self.burst_tail_threshold = _int(1, 50, 3)
        self.burst_tail_tau = _dbl(0.01, 10.0, 0.05, 0.70, suffix="s")
        self.burst_tail_scale = _dbl(0, 5.0, 0.05, 0.25)

        g.add_row("Preset", self.indicator_preset)
        g.add_row("τ rise", self.tau_rise)
        g.add_row("τ decay", self.tau_decay)
        g.add_row("Single-AP ΔF/F", self.single_ap_amp)
        g.add_row("", self.burst_tail_cb)
        g.add_row("Burst window", self.burst_tail_window)
        g.add_row("Burst threshold", self.burst_tail_threshold)
        g.add_row("Tail τ", self.burst_tail_tau)
        g.add_row("Tail scale", self.burst_tail_scale)

        self.indicator_preset.setCurrentText("GCaMP8f")
        self.indicator_preset.currentTextChanged.connect(self._apply_indicator_preset)
        self._apply_indicator_preset(self.indicator_preset.currentText())
        self.burst_tail_cb.toggled.connect(self._update_burst_tail_controls)
        self._update_burst_tail_controls()

    def _apply_indicator_preset(self, name):
        presets = {
            "GCaMP6f":       (0.026, 0.202, 0.22),
            "GCaMP6s":       (0.058, 0.656, 0.35),
            "GCaMP7f":       (0.025, 0.262, 0.21),
            "GCaMP8f":       (0.007, 0.097, 0.41),
            "GCaMP8m":       (0.007, 0.171, 0.76),
            "GCaMP8s":       (0.010, 0.442, 1.11),
            "jGCaMP8f":      (0.007, 0.097, 0.41),
            "jGCaMP8m":      (0.007, 0.171, 0.76),
            "jGCaMP8s":      (0.010, 0.442, 1.11),
            "Voltage (fast)":(0.003, 0.015, 0.60),
        }
        if name in presets:
            r, d, a = presets[name]
            self.tau_rise.setValue(r)
            self.tau_decay.setValue(d)
            self.single_ap_amp.setValue(a)
            for w in [self.tau_rise, self.tau_decay, self.single_ap_amp]:
                w.setEnabled(name == "Custom")
        else:
            for w in [self.tau_rise, self.tau_decay, self.single_ap_amp]:
                w.setEnabled(True)

    def _update_burst_tail_controls(self, *_args):
        enabled = self.burst_tail_cb.isChecked()
        for w in [self.burst_tail_window, self.burst_tail_threshold,
                  self.burst_tail_tau, self.burst_tail_scale]:
            w.setEnabled(enabled)

    def _build_cell_group(self, layout):
        g = CollapsibleGroup("🔵  Cell Morphology")
        layout.addWidget(g)

        self.cell_gain = _dbl(0.1, 20, 0.5, 1.0)
        self.F0 = _dbl(0.01, 50, 0.5, 1.0)
        self.gain_cv = _dbl(0, 3, 0.05, 0.50)
        self.F0_cv = _dbl(0, 3, 0.05, 0.35)
        self.morphology_mode = QComboBox()
        self.morphology_mode.addItems(["Soma only", "Soma + processes", "Dendritic branches"])
        self.cell_rad_min = _dbl(1, 50, 0.5, 2.0, suffix="px")
        self.cell_rad_max = _dbl(1, 50, 0.5, 3.5, suffix="px")
        self.cell_min_sep = _dbl(0.5, 5, 0.1, 1.5)
        self.shape_irreg = _dbl(0, 1, 0.05, 0.10)
        self.psf_sigma = _dbl(0, 5, 0.1, 0.9, suffix="px")
        self.donut_sigma = _dbl(0, 10, 0.5, 0.0, suffix="px")
        self.donut_contrast = _dbl(0, 1, 0.05, 0.0)
        self.process_prob = _dbl(0, 1, 0.05, 1.0)
        self.process_count = _int(1, 12, 1)
        self.process_diam = _dbl(0.2, 20, 0.1, 1.0, suffix="px")
        self.process_len_min = _dbl(1, 1000, 5, 25, suffix="px")
        self.process_len_max = _dbl(1, 1000, 5, 120, suffix="px")
        self.process_angle = _dbl(-180, 180, 5, 0, suffix="deg")
        self.process_angle_jitter = _dbl(0, 180, 5, 60, suffix="deg")
        self.process_continuity = _dbl(0, 1, 0.05, 0.75)
        self.process_F0_scale = _dbl(0, 5, 0.05, 0.60)
        self.process_gain_scale = _dbl(0, 5, 0.05, 0.80)
        self.process_flow_speed = _dbl(0, 1000, 10, 80, suffix="px/s")
        self.process_flow_bins = _int(1, 20, 7)
        self.varicosity_density = _dbl(0, 0.5, 0.005, 0.0, suffix="/px")
        self.varicosity_sigma = _dbl(0.2, 10, 0.1, 1.0, suffix="px")
        self.varicosity_strength = _dbl(0, 20, 0.5, 3.0)

        g.add_row("Cell gain", self.cell_gain)
        g.add_row("F₀ baseline", self.F0)
        g.add_row("Gain CV", self.gain_cv)
        g.add_row("F0 CV", self.F0_cv)
        g.add_row("Morphology", self.morphology_mode)
        g.add_row("Radius min", self.cell_rad_min)
        g.add_row("Radius max", self.cell_rad_max)
        g.add_row("Min separation", self.cell_min_sep)
        g.add_row("Shape irregularity", self.shape_irreg)
        g.add_row("PSF σ", self.psf_sigma)
        g.add_row("Donut σ", self.donut_sigma)
        g.add_row("Donut contrast", self.donut_contrast)
        g.add_row("Process probability", self.process_prob)
        g.add_row("Processes / ROI", self.process_count)
        g.add_row("Process diameter", self.process_diam)
        g.add_row("Length min", self.process_len_min)
        g.add_row("Length max", self.process_len_max)
        g.add_row("Orientation", self.process_angle)
        g.add_row("Angle jitter", self.process_angle_jitter)
        g.add_row("Continuity", self.process_continuity)
        g.add_row("Process F0 x", self.process_F0_scale)
        g.add_row("Process gain x", self.process_gain_scale)
        g.add_row("Flow speed", self.process_flow_speed)
        g.add_row("Flow bins", self.process_flow_bins)
        g.add_row("Varicosity density", self.varicosity_density)
        g.add_row("Varicosity size", self.varicosity_sigma)
        g.add_row("Varicosity strength", self.varicosity_strength)

        self.morphology_mode.currentTextChanged.connect(self._on_morphology_mode_changed)
        self._update_morphology_visibility()

    def _on_morphology_mode_changed(self, mode):
        self._apply_morphology_preset(mode)
        self._update_morphology_visibility()

    def _apply_morphology_preset(self, mode):
        if mode == "Soma + processes":
            self.n_units.setValue(45)
            self.n_inactive.setValue(45)
            self.F0.setValue(2.0)
            self.cell_rad_min.setValue(2.0)
            self.cell_rad_max.setValue(4.0)
            self.shape_irreg.setValue(0.18)
            self.psf_sigma.setValue(0.75)
            self.process_prob.setValue(0.50)
            self.process_count.setValue(2)
            self.process_diam.setValue(0.65)
            self.process_len_min.setValue(45)
            self.process_len_max.setValue(180)
            self.process_angle.setValue(90)
            self.process_angle_jitter.setValue(55)
            self.process_continuity.setValue(0.75)
            self.process_F0_scale.setValue(0.10)
            self.process_gain_scale.setValue(0.30)
            self.process_flow_speed.setValue(60)
            self.process_flow_bins.setValue(8)
            self.varicosity_density.setValue(0.018)
            self.varicosity_sigma.setValue(0.85)
            self.varicosity_strength.setValue(3.0)
            if hasattr(self, "img_h"):
                self.img_h.setValue(256)
                self.img_w.setValue(256)
            if hasattr(self, "np_blobs"):
                self.np_blobs.setValue(120)
                self.np_sigma_min.setValue(25)
                self.np_sigma_max.setValue(80)
                self.np_noise.setValue(0.004)
                self.np_level.setValue(0.045)
            if hasattr(self, "read_noise"):
                self.read_noise.setValue(0.012)
                self.shot_coeff.setValue(0.05)
                self.counts_per_unit.setValue(300)
        elif mode == "Soma only":
            self.n_units.setValue(100)
            self.n_inactive.setValue(100)
            self.F0.setValue(1.0)
            if hasattr(self, "img_h"):
                self.img_h.setValue(128)
                self.img_w.setValue(256)
            self.cell_rad_min.setValue(2.0)
            self.cell_rad_max.setValue(3.5)
            self.shape_irreg.setValue(0.10)
            self.psf_sigma.setValue(0.9)
            self.process_prob.setValue(1.0)
            self.process_count.setValue(1)
            self.process_diam.setValue(1.0)
            self.process_len_min.setValue(25)
            self.process_len_max.setValue(120)
            self.process_angle.setValue(0)
            self.process_angle_jitter.setValue(60)
            self.process_continuity.setValue(0.75)
            self.process_F0_scale.setValue(0.60)
            self.process_gain_scale.setValue(0.80)
            self.process_flow_speed.setValue(80)
            self.process_flow_bins.setValue(7)
            self.varicosity_density.setValue(0.0)
            self.varicosity_sigma.setValue(1.0)
            self.varicosity_strength.setValue(3.0)
            if hasattr(self, "np_blobs"):
                self.np_blobs.setValue(10)
                self.np_sigma_min.setValue(15)
                self.np_sigma_max.setValue(35)
                self.np_noise.setValue(0.03)
                self.np_level.setValue(0.1)
            if hasattr(self, "read_noise"):
                self.read_noise.setValue(0.10)
                self.shot_coeff.setValue(0.35)
                self.counts_per_unit.setValue(660)
        elif mode == "Dendritic branches":
            self.process_prob.setValue(1.0)
            self.process_count.setValue(4)
            self.process_diam.setValue(1.2)
            self.process_len_min.setValue(100)
            self.process_len_max.setValue(260)
            self.process_angle.setValue(0)
            self.process_angle_jitter.setValue(35)
            self.process_continuity.setValue(0.85)
            self.process_F0_scale.setValue(0.18)
            self.process_gain_scale.setValue(0.40)
            self.process_flow_speed.setValue(45)
            self.process_flow_bins.setValue(8)
            self.varicosity_density.setValue(0.010)
            self.varicosity_sigma.setValue(1.3)
            self.varicosity_strength.setValue(2.0)

    def _update_morphology_visibility(self, *_args):
        mode = self.morphology_mode.currentText()
        has_soma = mode != "Dendritic branches"
        has_process = mode != "Soma only"
        for w in [self.cell_rad_min, self.cell_rad_max, self.cell_min_sep,
                  self.shape_irreg, self.donut_sigma, self.donut_contrast]:
            w.setEnabled(has_soma)
        for w in [self.process_prob, self.process_count, self.process_diam, self.process_len_min,
                  self.process_len_max, self.process_angle, self.process_angle_jitter,
                  self.process_continuity, self.process_F0_scale, self.process_gain_scale,
                  self.process_flow_speed, self.process_flow_bins, self.varicosity_density,
                  self.varicosity_sigma, self.varicosity_strength]:
            w.setEnabled(has_process)

    def _build_neuropil_group(self, layout):
        g = CollapsibleGroup("🌊  Neuropil Background")
        layout.addWidget(g)

        self.np_blobs = _int(0, 50, 10)
        self.np_sigma_min = _dbl(1, 200, 1, 15, suffix="px")
        self.np_sigma_max = _dbl(1, 200, 1, 35, suffix="px")
        self.np_amp = _dbl(0, 1, 0.01, 0.00)
        self.np_freq = _dbl(0, 10, 0.1, 0.0, suffix="Hz")
        self.np_noise = _dbl(0, 2, 0.01, 0.03)
        self.np_bleed = _dbl(0, 1, 0.05, 0.20)
        self.np_level = _dbl(0, 5, 0.05, 0.1)

        g.add_row("Blobs", self.np_blobs)
        g.add_row("Blob σ min", self.np_sigma_min)
        g.add_row("Blob σ max", self.np_sigma_max)
        g.add_row("Oscillation amp", self.np_amp)
        g.add_row("Oscillation freq", self.np_freq)
        g.add_row("Neuropil noise", self.np_noise)
        g.add_row("Bleed fraction", self.np_bleed)
        g.add_row("DC level", self.np_level)

    def _build_motion_group(self, layout):
        g = CollapsibleGroup("📐  Motion Artifacts")
        layout.addWidget(g)

        self.jitter_std = _dbl(0, 10, 0.1, 0.5, suffix="px")
        self.jitter_tau = _dbl(0.001, 5, 0.01, 0.05, suffix="s")
        self.jump_px = _dbl(0, 20, 0.5, 2.0, suffix="px")
        self.jump_rate = _dbl(0, 10, 0.01, 0.05, suffix="Hz")
        self.jump_hold = _dbl(0, 1, 0.01, 0.02, suffix="s")

        g.add_row("Jitter σ", self.jitter_std)
        g.add_row("Jitter τ", self.jitter_tau)
        g.add_row("Saccade size", self.jump_px)
        g.add_row("Saccade rate", self.jump_rate)
        g.add_row("Saccade hold", self.jump_hold)

    def _build_noise_group(self, layout):
        g = CollapsibleGroup("📡  Noise")
        layout.addWidget(g)

        self.read_noise = _dbl(0, 2, 0.005, 0.10)
        self.shot_coeff = _dbl(0, 2, 0.01, 0.35)
        self.adc_bit_depth = _int(1, 16, 13)
        self.counts_per_unit = _dbl(1, 10000, 10, 660, decimals=0, suffix="ct/a.u.")

        g.add_row("Read noise σ", self.read_noise)
        g.add_row("Shot noise coeff", self.shot_coeff)
        g.add_row("ADC bit depth", self.adc_bit_depth)
        g.add_row("Counts scale", self.counts_per_unit)

    # ── Param extraction ──────────────────────────────────────────────────────

    def _parse_mod_onsets(self) -> list[float]:
        text = self.mod_onsets.text().strip()
        if not text:
            raise ValueError("Add at least one activity onset, e.g. 0:.5:10 or 0, 2.5, 5.")

        onsets = []
        for raw_part in text.replace(";", ",").split(","):
            part = raw_part.strip()
            if not part:
                continue

            fields = [p.strip() for p in part.split(":")]
            if len(fields) == 1:
                try:
                    onsets.append(float(fields[0]))
                except ValueError as exc:
                    raise ValueError(f"Activity onset '{part}' is not numeric.") from exc
            elif len(fields) == 3:
                try:
                    start, step, stop = (float(value) for value in fields)
                except ValueError as exc:
                    raise ValueError(f"Activity onset range '{part}' has non-numeric values.") from exc
                if step <= 0:
                    raise ValueError(f"Activity onset range '{part}' needs a positive step.")
                if stop < start:
                    raise ValueError(f"Activity onset range '{part}' needs stop >= start.")
                n_steps = int(np.floor((stop - start) / step + 1e-9))
                onsets.extend(start + step * i for i in range(n_steps + 1))
            else:
                raise ValueError(f"Activity onset range '{part}' must be start:step:stop.")

        duration = self.mod_active_duration.value()
        if not onsets:
            raise ValueError("Add at least one activity onset, e.g. 0:.5:10 or 0, 2.5, 5.")

        checked = []
        for onset in onsets:
            if onset < 0:
                raise ValueError(f"Activity onset {onset:g}s must be >= 0.")
            if onset + duration > self.duration_s.value():
                raise ValueError(
                    f"Activity onset {onset:g}s plus Active Duration {duration:g}s exceeds "
                    f"the recording duration "
                    f"({self.duration_s.value():.1f}s)."
                )
            checked.append(onset)

        return sorted(set(round(onset, 9) for onset in checked))

    def get_spike_opts(self) -> SpikeGenOpts:
        if self.baseline_rate_max.value() < self.baseline_rate_min.value():
            raise ValueError("Baseline rate max must be greater than or equal to min.")

        mods = []
        if self.use_mod.isChecked():
            if self.mod_peak_max.value() < self.mod_peak_min.value():
                raise ValueError("Peak rate max must be greater than or equal to min.")

            active_duration = self.mod_active_duration.value()
            for t_on in self._parse_mod_onsets():
                mods.append(ModEpoch(
                    t_on=t_on,
                    duration=active_duration,
                    peak_rate_range=(self.mod_peak_min.value(), self.mod_peak_max.value()),
                    onset_jitter=self.mod_onset_jitter.value(),
                    offset_jitter=self.mod_offset_jitter.value(),
                    clamp_duration=self.mod_lock_duration.isChecked(),
                    profile=self.mod_profile.currentText(),
                ))
        return SpikeGenOpts(
            n_units=self.n_units.value(),
            T=self.duration_s.value(),
            baseline_rate=(self.baseline_rate_min.value(), self.baseline_rate_max.value()),
            dt=1.0 / self.fps.value(),
            CV=self.cv.value(),
            burst_alpha=self.burst_alpha.value(),
            burst_tau=self.burst_tau.value(),
            seed=self.seed.value(),
            mod=mods,
            lock_onset_jitter_per_unit=self.mod_lock_onset_jitter.isChecked(),
        )

    def get_calcium_params(self) -> CalciumParams:
        return CalciumParams(
            tau_rise=self.tau_rise.value(),
            tau_decay=self.tau_decay.value(),
            single_ap_amp=self.single_ap_amp.value(),
            dt=1.0 / self.fps.value(),
            burst_tail_enabled=self.burst_tail_cb.isChecked(),
            burst_tail_window=self.burst_tail_window.value(),
            burst_tail_threshold=self.burst_tail_threshold.value(),
            burst_tail_tau=self.burst_tail_tau.value(),
            burst_tail_scale=self.burst_tail_scale.value(),
        )

    def get_sim_params(self) -> SimParams:
        morphology_mode = {
            "Soma only": "soma",
            "Soma + processes": "soma_process",
            "Dendritic branches": "dendrite",
        }[self.morphology_mode.currentText()]
        return SimParams(
            img_size=(self.img_h.value(), self.img_w.value()),
            F0=self.F0.value(),
            cell_gain=self.cell_gain.value(),
            F0_cv=self.F0_cv.value(),
            cell_gain_cv=self.gain_cv.value(),
            inactive_count=self.n_inactive.value(),
            cell_rad_px=(self.cell_rad_min.value(), self.cell_rad_max.value()),
            cell_min_sep=self.cell_min_sep.value(),
            shape_irreg_amp=self.shape_irreg.value(),
            psf_sigma=self.psf_sigma.value(),
            donut_sigma=self.donut_sigma.value(),
            donut_contrast=self.donut_contrast.value(),
            morphology_mode=morphology_mode,
            process_prob=self.process_prob.value(),
            process_count=self.process_count.value(),
            process_diameter_px=self.process_diam.value(),
            process_length_px=(self.process_len_min.value(), self.process_len_max.value()),
            process_orientation_deg=self.process_angle.value(),
            process_orientation_jitter_deg=self.process_angle_jitter.value(),
            process_continuity=self.process_continuity.value(),
            process_F0_scale=self.process_F0_scale.value(),
            process_gain_scale=self.process_gain_scale.value(),
            process_flow_speed_px_s=self.process_flow_speed.value(),
            process_flow_bins=self.process_flow_bins.value(),
            varicosity_density_per_px=self.varicosity_density.value(),
            varicosity_sigma_px=self.varicosity_sigma.value(),
            varicosity_strength=self.varicosity_strength.value(),
            neuropil_blobs=self.np_blobs.value(),
            neuropil_sigma_px=(self.np_sigma_min.value(), self.np_sigma_max.value()),
            neuropil_amp=self.np_amp.value(),
            neuropil_freq=self.np_freq.value(),
            neuropil_noise=self.np_noise.value(),
            neuropil_bleed_frac=self.np_bleed.value(),
            neuropil_level=self.np_level.value(),
            jitter_std=self.jitter_std.value(),
            jitter_tau=self.jitter_tau.value(),
            jump_px=self.jump_px.value(),
            jump_rate=self.jump_rate.value(),
            jump_hold_sec=self.jump_hold.value(),
            read_noise=self.read_noise.value(),
            shot_coeff=self.shot_coeff.value(),
            uint16_counts_per_unit=self.counts_per_unit.value(),
            uint16_bit_depth=self.adc_bit_depth.value(),
            seed=self.seed.value(),
        )

    def to_preset_dict(self) -> dict:
        """Return all GUI-adjustable simulation parameters as a JSON-friendly dict."""
        return {
            "version": self.PARAM_FILE_VERSION,
            "recording": {
                "n_active_neurons": self.n_units.value(),
                "n_inactive_neurons": self.n_inactive.value(),
                "duration_s": self.duration_s.value(),
                "image_h_px": self.img_h.value(),
                "image_w_px": self.img_w.value(),
                "frame_rate_hz": self.fps.value(),
                "seed": self.seed.value(),
            },
            "spiking_activity": {
                "baseline_rate_min_hz": self.baseline_rate_min.value(),
                "baseline_rate_max_hz": self.baseline_rate_max.value(),
                "isi_cv": self.cv.value(),
                "burst_alpha": self.burst_alpha.value(),
                "burst_tau_s": self.burst_tau.value(),
                "enable_epoch": self.use_mod.isChecked(),
                "activity_onsets": self.mod_onsets.text(),
                "active_duration_s": self.mod_active_duration.value(),
                "onset_jitter_s": self.mod_onset_jitter.value(),
                "lock_onset_jitter_per_cell": self.mod_lock_onset_jitter.isChecked(),
                "offset_jitter_s": self.mod_offset_jitter.value(),
                "fixed_duration_after_onset_jitter": self.mod_lock_duration.isChecked(),
                "peak_rate_min_hz": self.mod_peak_min.value(),
                "peak_rate_max_hz": self.mod_peak_max.value(),
                "epoch_profile": self.mod_profile.currentText(),
            },
            "calcium_indicator": {
                "preset": self.indicator_preset.currentText(),
                "tau_rise_s": self.tau_rise.value(),
                "tau_decay_s": self.tau_decay.value(),
                "single_ap_dff": self.single_ap_amp.value(),
                "prolong_burst_tail": self.burst_tail_cb.isChecked(),
                "burst_window_s": self.burst_tail_window.value(),
                "burst_threshold_spikes": self.burst_tail_threshold.value(),
                "tail_tau_s": self.burst_tail_tau.value(),
                "tail_scale": self.burst_tail_scale.value(),
            },
            "cell_morphology": {
                "cell_gain": self.cell_gain.value(),
                "f0_baseline": self.F0.value(),
                "gain_cv": self.gain_cv.value(),
                "f0_cv": self.F0_cv.value(),
                "morphology": self.morphology_mode.currentText(),
                "radius_min_px": self.cell_rad_min.value(),
                "radius_max_px": self.cell_rad_max.value(),
                "min_separation": self.cell_min_sep.value(),
                "shape_irregularity": self.shape_irreg.value(),
                "psf_sigma_px": self.psf_sigma.value(),
                "donut_sigma_px": self.donut_sigma.value(),
                "donut_contrast": self.donut_contrast.value(),
                "process_probability": self.process_prob.value(),
                "processes_per_roi": self.process_count.value(),
                "process_diameter_px": self.process_diam.value(),
                "length_min_px": self.process_len_min.value(),
                "length_max_px": self.process_len_max.value(),
                "orientation_deg": self.process_angle.value(),
                "angle_jitter_deg": self.process_angle_jitter.value(),
                "continuity": self.process_continuity.value(),
                "process_f0_x": self.process_F0_scale.value(),
                "process_gain_x": self.process_gain_scale.value(),
                "flow_speed_px_s": self.process_flow_speed.value(),
                "flow_bins": self.process_flow_bins.value(),
                "varicosity_density_per_px": self.varicosity_density.value(),
                "varicosity_size_px": self.varicosity_sigma.value(),
                "varicosity_strength": self.varicosity_strength.value(),
            },
            "neuropil": {
                "blobs": self.np_blobs.value(),
                "blob_sigma_min_px": self.np_sigma_min.value(),
                "blob_sigma_max_px": self.np_sigma_max.value(),
                "oscillation_amp": self.np_amp.value(),
                "oscillation_freq_hz": self.np_freq.value(),
                "neuropil_noise": self.np_noise.value(),
                "bleed_fraction": self.np_bleed.value(),
                "dc_level": self.np_level.value(),
            },
            "motion": {
                "jitter_sigma_px": self.jitter_std.value(),
                "jitter_tau_s": self.jitter_tau.value(),
                "saccade_size_px": self.jump_px.value(),
                "saccade_rate_hz": self.jump_rate.value(),
                "saccade_hold_s": self.jump_hold.value(),
            },
            "noise": {
                "read_noise_sigma": self.read_noise.value(),
                "shot_noise_coeff": self.shot_coeff.value(),
                "adc_bit_depth": self.adc_bit_depth.value(),
                "counts_scale_ct_per_au": self.counts_per_unit.value(),
            },
        }

    def apply_preset_dict(self, preset: dict) -> None:
        """Apply a parameter preset previously produced by to_preset_dict."""
        if not isinstance(preset, dict):
            raise ValueError("Parameter preset must be a JSON object.")

        recording = preset.get("recording", {})
        spiking = preset.get("spiking_activity", {})
        calcium = preset.get("calcium_indicator", {})
        morph = preset.get("cell_morphology", {})
        neuropil = preset.get("neuropil", {})
        motion = preset.get("motion", {})
        noise = preset.get("noise", {})

        self._set_combo_text(self.morphology_mode, morph.get("morphology"))
        self._set_combo_text(self.indicator_preset, calcium.get("preset"))

        self._set_spin(self.n_units, recording.get("n_active_neurons"))
        self._set_spin(self.n_inactive, recording.get("n_inactive_neurons"))
        self._set_spin(self.duration_s, recording.get("duration_s"))
        self._set_spin(self.img_h, recording.get("image_h_px"))
        self._set_spin(self.img_w, recording.get("image_w_px"))
        self._set_spin(self.fps, recording.get("frame_rate_hz"))
        self._set_spin(self.seed, recording.get("seed"))

        self._set_spin(self.baseline_rate_min, spiking.get("baseline_rate_min_hz"))
        self._set_spin(self.baseline_rate_max, spiking.get("baseline_rate_max_hz"))
        self._set_spin(self.cv, spiking.get("isi_cv"))
        self._set_spin(self.burst_alpha, spiking.get("burst_alpha"))
        self._set_spin(self.burst_tau, spiking.get("burst_tau_s"))
        self._set_checked(self.use_mod, spiking.get("enable_epoch"))
        if "activity_onsets" in spiking:
            self.mod_onsets.setText(str(spiking["activity_onsets"]))
        self._set_spin(self.mod_active_duration, spiking.get("active_duration_s"))
        self._set_spin(self.mod_onset_jitter, spiking.get("onset_jitter_s"))
        self._set_checked(self.mod_lock_onset_jitter, spiking.get("lock_onset_jitter_per_cell"))
        self._set_spin(self.mod_offset_jitter, spiking.get("offset_jitter_s"))
        self._set_checked(self.mod_lock_duration, spiking.get("fixed_duration_after_onset_jitter"))
        self._set_spin(self.mod_peak_min, spiking.get("peak_rate_min_hz"))
        self._set_spin(self.mod_peak_max, spiking.get("peak_rate_max_hz"))
        self._set_combo_text(self.mod_profile, spiking.get("epoch_profile"))

        self._set_spin(self.tau_rise, calcium.get("tau_rise_s"))
        self._set_spin(self.tau_decay, calcium.get("tau_decay_s"))
        self._set_spin(self.single_ap_amp, calcium.get("single_ap_dff"))
        self._set_checked(self.burst_tail_cb, calcium.get("prolong_burst_tail"))
        self._set_spin(self.burst_tail_window, calcium.get("burst_window_s"))
        self._set_spin(self.burst_tail_threshold, calcium.get("burst_threshold_spikes"))
        self._set_spin(self.burst_tail_tau, calcium.get("tail_tau_s"))
        self._set_spin(self.burst_tail_scale, calcium.get("tail_scale"))

        self._set_spin(self.cell_gain, morph.get("cell_gain"))
        self._set_spin(self.F0, morph.get("f0_baseline"))
        self._set_spin(self.gain_cv, morph.get("gain_cv"))
        self._set_spin(self.F0_cv, morph.get("f0_cv"))
        self._set_spin(self.cell_rad_min, morph.get("radius_min_px"))
        self._set_spin(self.cell_rad_max, morph.get("radius_max_px"))
        self._set_spin(self.cell_min_sep, morph.get("min_separation"))
        self._set_spin(self.shape_irreg, morph.get("shape_irregularity"))
        self._set_spin(self.psf_sigma, morph.get("psf_sigma_px"))
        self._set_spin(self.donut_sigma, morph.get("donut_sigma_px"))
        self._set_spin(self.donut_contrast, morph.get("donut_contrast"))
        self._set_spin(self.process_prob, morph.get("process_probability"))
        self._set_spin(self.process_count, morph.get("processes_per_roi"))
        self._set_spin(self.process_diam, morph.get("process_diameter_px"))
        self._set_spin(self.process_len_min, morph.get("length_min_px"))
        self._set_spin(self.process_len_max, morph.get("length_max_px"))
        self._set_spin(self.process_angle, morph.get("orientation_deg"))
        self._set_spin(self.process_angle_jitter, morph.get("angle_jitter_deg"))
        self._set_spin(self.process_continuity, morph.get("continuity"))
        self._set_spin(self.process_F0_scale, morph.get("process_f0_x"))
        self._set_spin(self.process_gain_scale, morph.get("process_gain_x"))
        self._set_spin(self.process_flow_speed, morph.get("flow_speed_px_s"))
        self._set_spin(self.process_flow_bins, morph.get("flow_bins"))
        self._set_spin(self.varicosity_density, morph.get("varicosity_density_per_px"))
        self._set_spin(self.varicosity_sigma, morph.get("varicosity_size_px"))
        self._set_spin(self.varicosity_strength, morph.get("varicosity_strength"))

        self._set_spin(self.np_blobs, neuropil.get("blobs"))
        self._set_spin(self.np_sigma_min, neuropil.get("blob_sigma_min_px"))
        self._set_spin(self.np_sigma_max, neuropil.get("blob_sigma_max_px"))
        self._set_spin(self.np_amp, neuropil.get("oscillation_amp"))
        self._set_spin(self.np_freq, neuropil.get("oscillation_freq_hz"))
        self._set_spin(self.np_noise, neuropil.get("neuropil_noise"))
        self._set_spin(self.np_bleed, neuropil.get("bleed_fraction"))
        self._set_spin(self.np_level, neuropil.get("dc_level"))

        self._set_spin(self.jitter_std, motion.get("jitter_sigma_px"))
        self._set_spin(self.jitter_tau, motion.get("jitter_tau_s"))
        self._set_spin(self.jump_px, motion.get("saccade_size_px"))
        self._set_spin(self.jump_rate, motion.get("saccade_rate_hz"))
        self._set_spin(self.jump_hold, motion.get("saccade_hold_s"))

        self._set_spin(self.read_noise, noise.get("read_noise_sigma"))
        self._set_spin(self.shot_coeff, noise.get("shot_noise_coeff"))
        self._set_spin(self.adc_bit_depth, noise.get("adc_bit_depth"))
        self._set_spin(self.counts_per_unit, noise.get("counts_scale_ct_per_au"))

        self._update_mod_visibility()
        self._update_burst_tail_controls()
        self._update_morphology_visibility()

    @staticmethod
    def _set_spin(widget, value):
        if value is None:
            return
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        else:
            widget.setValue(float(value))

    @staticmethod
    def _set_checked(widget, value):
        if value is None:
            return
        widget.setChecked(bool(value))

    @staticmethod
    def _set_combo_text(widget, value):
        if value is None:
            return
        idx = widget.findText(str(value))
        if idx < 0:
            raise ValueError(f"Unknown option '{value}' for parameter preset.")
        widget.setCurrentIndex(idx)
