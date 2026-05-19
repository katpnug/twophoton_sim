"""
sim_worker.py
=============
QThread worker that runs the full simulation pipeline:
  Spike generation → Calcium dynamics → Stack rendering

Emits progress and results back to the GUI thread via Qt signals.
"""

from __future__ import annotations
import traceback
import numpy as np

from PySide6.QtCore import QThread, Signal, QObject

from core.spike_generator import SpikeGenOpts, ModEpoch, generate_spikes
from core.calcium_dynamics import CalciumParams, spikes_to_calcium
from core.stack_simulator import SimParams, SimResult, simulate_stack


class SimWorker(QThread):
    # Signals
    progress = Signal(int, int, str)       # current, total, stage
    frame_ready = Signal(int, object)      # frame_idx, 2D ndarray (float32)
    finished = Signal(object)              # SimResult
    error = Signal(str)                    # error message

    def __init__(
        self,
        spike_opts: SpikeGenOpts,
        calcium_params: CalciumParams,
        sim_params: SimParams,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.spike_opts = spike_opts
        self.calcium_params = calcium_params
        self.sim_params = sim_params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            self._run()
        except Exception:
            self.error.emit(traceback.format_exc())

    def _run(self):
        opts = self.spike_opts
        cal = self.calcium_params
        sim = self.sim_params

        # ── Stage 1: Spike generation ─────────────────────────────────────
        total_stages = opts.n_units + 1   # approx
        self.progress.emit(0, 100, "Generating spikes…")
        spike_result = generate_spikes(opts)
        if self._abort:
            return

        # ── Stage 2: Calcium dynamics ─────────────────────────────────────
        self.progress.emit(10, 100, "Computing calcium dynamics…")
        C = spikes_to_calcium(spike_result.spike_times, opts.T, cal)
        if self._abort:
            return

        # ── Stage 3: Stack rendering ─────────────────────────────────────
        T_frames = C.shape[1]

        # collect every ~N-th frame for live preview (max 60 preview frames)
        preview_every = max(1, T_frames // 60)
        frame_buffer: list[tuple[int, np.ndarray]] = []

        def on_frame(k: int, F: np.ndarray):
            if self._abort:
                return
            pct = 30 + int(70 * k / T_frames)
            self.progress.emit(pct, 100, f"Rendering frame {k+1}/{T_frames}…")
            if k % preview_every == 0:
                self.frame_ready.emit(k, F.astype(np.float32))

        self.progress.emit(30, 100, "Rendering stack…")
        result = simulate_stack(
            C=C,
            dt=1.0 / (T_frames / opts.T),   # dt from T and n_frames
            params=sim,
            on_frame=on_frame,
            extract_traces=True,
            return_movie=True,
            baseline_prct=8.0,
        )

        if self._abort:
            return

        # attach spike result for downstream use
        result.spike_times = spike_result.spike_times
        result.rate_per_unit = spike_result.rate_per_unit
        result.rate_t = spike_result.rate_t
        result.baseline_rate_per_unit = spike_result.baseline_rate_per_unit
        result.mod_windows = spike_result.mod_windows

        self.progress.emit(100, 100, "Done.")
        self.finished.emit(result)
