"""
spike_generator.py
==================
Python port of generate_poisson_like_spikes (simgrc_2026_kpn.m).

Generates inhomogeneous spike trains with:
  - Exact Poisson (CV=1, burst_alpha=0)
  - Gamma-renewal (CV≠1, burst_alpha=0)
  - Hawkes self-excitation (burst_alpha > 0)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModEpoch:
    """A single modulation epoch."""
    t_on: float
    peak_rate_range: tuple[float, float]
    t_off: Optional[float] = None
    duration: Optional[float] = None
    onset_jitter: float = 0.0
    offset_jitter: float = 0.0
    profile: str = "box"           # 'box' | 'cosine'
    clamp_duration: bool = False
    lock_onset_jitter_per_unit: Optional[bool] = None
    lock_offset_jitter_per_unit: Optional[bool] = None

    def __post_init__(self):
        if self.t_off is None and self.duration is None:
            raise ValueError("ModEpoch needs t_off or duration.")
        if self.t_off is None:
            self.t_off = self.t_on + self.duration


@dataclass
class SpikeGenOpts:
    n_units: int
    T: float                          # seconds
    baseline_rate: float | tuple      # Hz or (min, max)
    dt: float = 1e-3
    CV: float = 1.0
    burst_alpha: float = 0.0
    burst_tau: float = 0.015
    seed: Optional[int] = None
    mod: list[ModEpoch] = field(default_factory=list)
    lock_onset_jitter_per_unit: bool = False
    lock_offset_jitter_per_unit: bool = False


@dataclass
class SpikeGenResult:
    spike_times: list[np.ndarray]
    rate_t: np.ndarray
    rate_per_unit: np.ndarray
    t: np.ndarray
    baseline_rate_per_unit: np.ndarray
    mod_windows: list[dict] = field(default_factory=list)


def generate_spikes(opts: SpikeGenOpts) -> SpikeGenResult:
    rng = np.random.default_rng(opts.seed)

    n = opts.n_units
    T = opts.T
    dt = opts.dt
    t = np.arange(0, T + dt * 0.5, dt)
    nt = len(t)

    # Baseline rates per unit
    br = opts.baseline_rate
    if np.isscalar(br):
        base_rates = np.full(n, float(br))
    else:
        lo, hi = float(br[0]), float(br[1])
        base_rates = rng.uniform(lo, hi, n)

    rate_per_unit = np.tile(base_rates[:, None], (1, nt))  # (n, nt)

    # Apply modulation epochs
    mod_windows = []
    if opts.mod:
        base_onset_lock = rng.standard_normal(n)
        base_offset_lock = rng.standard_normal(n)
        rc_prop = 0.10

        for M in opts.mod:
            epoch_onsets = np.full(n, np.nan)
            epoch_offsets = np.full(n, np.nan)
            lock_on = M.lock_onset_jitter_per_unit if M.lock_onset_jitter_per_unit is not None else opts.lock_onset_jitter_per_unit
            lock_off = M.lock_offset_jitter_per_unit if M.lock_offset_jitter_per_unit is not None else opts.lock_offset_jitter_per_unit

            peak_rates = rng.uniform(M.peak_rate_range[0], M.peak_rate_range[1], n)

            if lock_on:
                jitter_on_vec = M.onset_jitter * base_onset_lock
            else:
                jitter_on_vec = M.onset_jitter * rng.standard_normal(n)

            if M.clamp_duration:
                jitter_off_vec = np.zeros(n)
            else:
                if lock_off:
                    jitter_off_vec = M.offset_jitter * base_offset_lock
                else:
                    jitter_off_vec = M.offset_jitter * rng.standard_normal(n)

            dur_nominal = M.t_off - M.t_on

            for i in range(n):
                if M.clamp_duration:
                    ton = M.t_on + jitter_on_vec[i]
                    toff = ton + dur_nominal
                else:
                    ton = M.t_on + jitter_on_vec[i]
                    toff = M.t_off + jitter_off_vec[i]

                if toff <= ton:
                    continue

                epoch_onsets[i] = ton
                epoch_offsets[i] = toff

                idx = np.where((t >= ton) & (t <= toff))[0]
                if len(idx) == 0:
                    continue

                L = len(idx)
                if M.profile == "cosine":
                    rcl = max(1, round(rc_prop * L))
                    w = np.ones(L)
                    ramp = (1 - np.cos(np.linspace(0, np.pi, rcl))) / 2
                    w[:rcl] = ramp
                    w[L - rcl:] = np.minimum(w[L - rcl:], ramp[::-1])
                else:
                    w = np.ones(L)

                add_rate = (peak_rates[i] - base_rates[i]) * w
                rate_per_unit[i, idx] = np.maximum(0.0, base_rates[i] + add_rate)

            mod_windows.append({
                "nominal_on": M.t_on,
                "nominal_off": M.t_off,
                "onsets": epoch_onsets,
                "offsets": epoch_offsets,
                "peak_rates": peak_rates,
                "profile": M.profile,
            })

    rate_per_unit = np.where(np.isfinite(rate_per_unit), rate_per_unit, 0.0)
    rate_per_unit = np.maximum(0.0, rate_per_unit)

    # Spike generation
    use_hawkes = opts.burst_alpha > 0
    use_poisson = (not use_hawkes) and (abs(opts.CV - 1.0) < 1e-12)
    use_gamma = (not use_hawkes) and (not use_poisson)

    spike_times = []
    for i in range(n):
        lam = rate_per_unit[i]  # Hz at each dt step
        if use_poisson:
            st = _poisson_thinning(t, lam, rng)
        elif use_gamma:
            st = _gamma_renewal(t, lam, opts.CV, rng)
        else:
            st = _hawkes_thinning(t, lam, opts.burst_alpha, opts.burst_tau, rng)
        spike_times.append(st)

    return SpikeGenResult(
        spike_times=spike_times,
        rate_t=t,
        rate_per_unit=rate_per_unit,
        t=t,
        baseline_rate_per_unit=base_rates,
        mod_windows=mod_windows,
    )


# -----------------------------------------------------------------------
# Private spike-generation helpers
# -----------------------------------------------------------------------

def _poisson_thinning(t, lam, rng):
    """Ogata thinning for inhomogeneous Poisson."""
    lam_max = lam.max()
    if lam_max == 0:
        return np.array([])
    T = t[-1]
    dt = t[1] - t[0]
    spikes = []
    s = 0.0
    while s < T:
        u = rng.exponential(1.0 / lam_max)
        s += u
        if s >= T:
            break
        idx = min(int(s / dt), len(lam) - 1)
        if rng.random() < lam[idx] / lam_max:
            spikes.append(s)
    return np.array(spikes)


def _gamma_renewal(t, lam, CV, rng):
    """Time-rescaling for gamma renewal process."""
    if CV <= 0:
        CV = 1e-3
    shape = 1.0 / (CV ** 2)
    rate_param = shape            # scale = 1/shape so mean=1
    dt = t[1] - t[0]
    T = t[-1]
    # Integrated rate (compensator)
    Lambda = np.cumsum(lam) * dt  # total expected spikes up to t[i]
    spikes = []
    n_cum = rng.gamma(shape, 1.0 / rate_param)
    while n_cum < Lambda[-1]:
        # invert: find time where Lambda(t) = n_cum
        idx = np.searchsorted(Lambda, n_cum)
        if idx >= len(t):
            break
        spikes.append(t[min(idx, len(t) - 1)])
        n_cum += rng.gamma(shape, 1.0 / rate_param)
    return np.array(spikes)


def _hawkes_thinning(t, lam0, alpha, tau, rng):
    """Ogata thinning for Hawkes self-excitation."""
    T = t[-1]
    dt = t[1] - t[0]
    spikes = []
    history = []
    s = 0.0
    while s < T:
        # Conditional intensity upper bound
        hawkes_contrib = sum(alpha / tau * np.exp(-(s - sp) / tau) for sp in history if sp < s)
        idx = min(int(s / dt), len(lam0) - 1)
        lam_bar = lam0[idx] + hawkes_contrib + 1e-6
        u = rng.exponential(1.0 / lam_bar)
        s += u
        if s >= T:
            break
        hawkes_contrib_new = sum(alpha / tau * np.exp(-(s - sp) / tau) for sp in history if sp < s)
        idx2 = min(int(s / dt), len(lam0) - 1)
        lam_s = lam0[idx2] + hawkes_contrib_new
        if rng.random() < lam_s / lam_bar:
            spikes.append(s)
            history.append(s)
    return np.array(spikes)
