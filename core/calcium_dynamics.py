"""
calcium_dynamics.py
===================
Convert spike trains to calcium fluorescence traces via an AR(2) model,
mirroring the MATLAB toolchain's calcium generation logic.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class CalciumParams:
    """AR(2) calcium parameters for a given cell type / indicator."""
    # Rise and decay time constants (seconds)
    tau_rise: float = 0.026     # s  (GCaMP6f-like)
    tau_decay: float = 0.202    # s
    # Amplitude of a single AP (in ΔF/F units for the *clean* signal)
    single_ap_amp: float = 0.22
    dt: float = 1e-3
    burst_tail_enabled: bool = False
    burst_tail_window: float = 0.20
    burst_tail_threshold: int = 3
    burst_tail_tau: float = 0.70
    burst_tail_scale: float = 0.25

    # Presets
    @classmethod
    def gcamp6f(cls, dt=1e-3):
        return cls(tau_rise=0.026, tau_decay=0.202, single_ap_amp=0.22, dt=dt)

    @classmethod
    def gcamp6s(cls, dt=1e-3):
        return cls(tau_rise=0.058, tau_decay=0.656, single_ap_amp=0.35, dt=dt)

    @classmethod
    def gcamp7f(cls, dt=1e-3):
        return cls(tau_rise=0.025, tau_decay=0.262, single_ap_amp=0.21, dt=dt)

    @classmethod
    def jgcamp8f(cls, dt=1e-3):
        return cls(tau_rise=0.007, tau_decay=0.097, single_ap_amp=0.41, dt=dt)

    @classmethod
    def gcamp8f(cls, dt=1e-3):
        return cls.jgcamp8f(dt=dt)

    @classmethod
    def jgcamp8m(cls, dt=1e-3):
        return cls(tau_rise=0.007, tau_decay=0.171, single_ap_amp=0.76, dt=dt)

    @classmethod
    def gcamp8m(cls, dt=1e-3):
        return cls.jgcamp8m(dt=dt)

    @classmethod
    def jgcamp8s(cls, dt=1e-3):
        return cls(tau_rise=0.010, tau_decay=0.442, single_ap_amp=1.11, dt=dt)

    @classmethod
    def gcamp8s(cls, dt=1e-3):
        return cls.jgcamp8s(dt=dt)

    @classmethod
    def voltage_fast(cls, dt=1e-3):
        """Voltage indicator – very fast transients."""
        return cls(tau_rise=0.003, tau_decay=0.015, single_ap_amp=0.60, dt=dt)


def spikes_to_calcium(
    spike_times_list: list[np.ndarray],
    T: float,
    params: CalciumParams,
    seed: int | None = None,
) -> np.ndarray:
    """
    Convert a list of spike-time arrays to calcium traces.

    Parameters
    ----------
    spike_times_list : list of 1-D arrays, length N_units
    T                : total duration (s)
    params           : CalciumParams
    seed             : optional RNG seed (unused, placeholder)

    Returns
    -------
    C : (N_units, n_frames) array of calcium fluorescence (ΔF/F units)
    """
    dt = params.dt
    t = np.arange(0, T + dt * 0.5, dt)
    n_frames = len(t)
    n_units = len(spike_times_list)

    # Build AR(2) kernel from rise/decay time constants
    kernel = _build_ar2_kernel(params.tau_rise, params.tau_decay, dt)
    kernel_len = len(kernel)

    C = np.zeros((n_units, n_frames), dtype=np.float64)

    for i, st in enumerate(spike_times_list):
        if len(st) == 0:
            continue
        # Convert spike times to a binary spike train on the dt grid
        spike_train = np.zeros(n_frames)
        spike_idx = np.round(st / dt).astype(int)
        spike_idx = spike_idx[(spike_idx >= 0) & (spike_idx < n_frames)]
        np.add.at(spike_train, spike_idx, 1)

        # Convolve with kernel (full, then trim)
        ca = np.convolve(spike_train, kernel, mode="full")[:n_frames]
        trace = ca * params.single_ap_amp
        if params.burst_tail_enabled:
            trace = _apply_burst_tail(trace, spike_train, params)
        C[i] = trace

    return C


def _build_ar2_kernel(tau_rise: float, tau_decay: float, dt: float) -> np.ndarray:
    """
    Double-exponential kernel that approximates AR(2) calcium dynamics.
    Peak is normalised to 1.
    """
    t_max = 5.0 * tau_decay  # kernel duration
    t_k = np.arange(0, t_max + dt * 0.5, dt)

    if tau_rise < dt * 0.5:  # single-exponential (fast rise)
        k = np.exp(-t_k / tau_decay)
    else:
        # Double-exponential (rise and decay)
        k = np.exp(-t_k / tau_decay) - np.exp(-t_k / tau_rise)
        # handle numerical issues at t=0
        k = np.where(np.isfinite(k), k, 0.0)

    pk = k.max()
    if pk > 0:
        k /= pk
    return k


def _apply_burst_tail(
    trace: np.ndarray,
    spike_train: np.ndarray,
    params: CalciumParams,
) -> np.ndarray:
    """
    Peak-preserving slow tail recruited by short high-frequency spike bursts.

    The AR(2) trace sets the burst peak. This only raises the decay after a
    complete burst group, so ongoing bursts do not get clipped into plateaus.
    """
    dt = params.dt
    n_frames = len(trace)
    window = max(dt, float(params.burst_tail_window))
    threshold = max(1, int(params.burst_tail_threshold))
    spike_idx = np.flatnonzero(spike_train > 0)
    if len(spike_idx) < threshold:
        return trace

    out = trace.copy()
    tau = max(dt, float(params.burst_tail_tau))
    scale = float(np.clip(params.burst_tail_scale, 0.0, 1.0))
    if scale <= 0:
        return trace

    for group in _burst_groups(spike_idx, window, threshold, dt):
        if not _qualifies_as_burst(group, window, threshold, dt):
            continue
        burst_end = int(group[-1])
        search_stop = min(n_frames, burst_end + max(1, int(round(window / dt))))
        peak_idx = int(burst_end + np.argmax(trace[burst_end:search_stop]))
        peak_val = float(trace[peak_idx])
        if peak_val <= 0:
            continue

        tail_stop = min(n_frames, peak_idx + max(1, int(round(5.0 * tau / dt))))
        rel_t = np.arange(tail_stop - peak_idx) * dt
        slow_floor = peak_val * np.exp(-rel_t / tau)
        target = trace[peak_idx:tail_stop] * (1.0 - scale) + slow_floor * scale
        out[peak_idx:tail_stop] = np.maximum(out[peak_idx:tail_stop], target)
        out[peak_idx] = trace[peak_idx]

    return out


def _burst_groups(spike_idx: np.ndarray, window: float, threshold: int, dt: float) -> list[np.ndarray]:
    if len(spike_idx) == 0:
        return []
    groups = []
    start = 0
    max_gap = max(1, int(round(window / dt)))
    for j in range(1, len(spike_idx)):
        if spike_idx[j] - spike_idx[j - 1] > max_gap:
            groups.append(spike_idx[start:j])
            start = j
    groups.append(spike_idx[start:])
    return [g for g in groups if len(g) >= threshold]


def _qualifies_as_burst(group: np.ndarray, window: float, threshold: int, dt: float) -> bool:
    max_span = max(1, int(round(window / dt)))
    for j in range(0, len(group) - threshold + 1):
        if group[j + threshold - 1] - group[j] <= max_span:
            return True
    return False
