#!/usr/bin/env python3
"""
run_headless.py
===============
CLI / scripting interface – generate a TIFF stack without opening the GUI.

Example:
    python run_headless.py --n_units 100 --duration 30 --output my_sim.tif
    python run_headless.py --help
"""

import argparse
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from core.spike_generator import SpikeGenOpts, ModEpoch, generate_spikes
from core.calcium_dynamics import CalciumParams, spikes_to_calcium
from core.stack_simulator import SimParams, simulate_stack, save_tiff


INDICATOR_PRESETS = {
    "gcamp6f":   dict(tau_rise=0.026, tau_decay=0.202, single_ap_amp=0.22),
    "gcamp6s":   dict(tau_rise=0.058, tau_decay=0.656, single_ap_amp=0.35),
    "gcamp7f":   dict(tau_rise=0.025, tau_decay=0.262, single_ap_amp=0.21),
    "gcamp8f":   dict(tau_rise=0.007, tau_decay=0.097, single_ap_amp=0.41),
    "jgcamp8f":  dict(tau_rise=0.007, tau_decay=0.097, single_ap_amp=0.41),
    "gcamp8m":   dict(tau_rise=0.007, tau_decay=0.171, single_ap_amp=0.76),
    "jgcamp8m":  dict(tau_rise=0.007, tau_decay=0.171, single_ap_amp=0.76),
    "gcamp8s":   dict(tau_rise=0.010, tau_decay=0.442, single_ap_amp=1.11),
    "jgcamp8s":  dict(tau_rise=0.010, tau_decay=0.442, single_ap_amp=1.11),
    "voltage":   dict(tau_rise=0.003, tau_decay=0.015, single_ap_amp=0.60),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="2P TIFF Simulator – headless mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Recording
    p.add_argument("--n_units",    type=int,   default=100,   help="Number of active neurons")
    p.add_argument("--duration",   type=float, default=30.0,  help="Recording duration (s)")
    p.add_argument("--fps",        type=float, default=115.0, help="Frame rate (Hz)")
    p.add_argument("--img_h",      type=int,   default=128,   help="Image height (px)")
    p.add_argument("--img_w",      type=int,   default=256,   help="Image width (px)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output",     type=str,   default="sim_2p.tif")

    # Spiking
    p.add_argument("--baseline_rate", type=float, nargs=2, default=[0.5, 3.0],
                   metavar=("MIN", "MAX"), help="Baseline firing rate range (Hz)")
    p.add_argument("--cv",            type=float, default=1.0, help="ISI coefficient of variation")
    p.add_argument("--burst_alpha",   type=float, default=0.0)
    p.add_argument("--mod_t_on",      type=float, default=1.0,  help="Epoch onset (s)")
    p.add_argument("--mod_duration",  type=float, default=0.25, help="Epoch duration (s)")
    p.add_argument("--mod_onsets", type=str, default="1:5:30",
                   help="Activity onset list, e.g. '0:.5:10' or '0,2.5,5'")
    p.add_argument("--mod_active_duration", type=float, default=0.25,
                   help="Duration of each activity epoch (s)")
    p.add_argument("--mod_epoch", type=float, nargs=2, action="append",
                   metavar=("ON", "OFF"),
                   help="Activity epoch onset/offset pair. May be repeated.")
    p.add_argument("--mod_onset_jitter", type=float, default=0.1,
                   help="Per-unit Gaussian onset jitter SD (s)")
    p.add_argument("--mod_lock_onset_jitter", action="store_true",
                   help="Use one onset-jitter offset per unit across all activity epochs")
    p.add_argument("--mod_offset_jitter", type=float, default=0.0,
                   help="Per-unit Gaussian offset jitter SD (s)")
    p.add_argument("--mod_lock_duration", action="store_true",
                   help="Apply onset jitter while preserving epoch duration")
    p.add_argument("--mod_peak",      type=float, nargs=2, default=[20.0, 80.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--no_mod", action="store_true", help="Disable activity epoch")

    # Indicator
    p.add_argument("--indicator", default="gcamp8f",
                   choices=list(INDICATOR_PRESETS.keys()))
    p.add_argument("--burst_tail", action="store_true",
                   help="Add a nonlinear prolonged calcium tail after spike bursts")
    p.add_argument("--burst_tail_window", type=float, default=0.20,
                   help="Window for counting burst spikes (s)")
    p.add_argument("--burst_tail_threshold", type=int, default=3,
                   help="Spike count threshold for recruiting the slow tail")
    p.add_argument("--burst_tail_tau", type=float, default=0.70,
                   help="Slow burst-tail decay time constant (s)")
    p.add_argument("--burst_tail_scale", type=float, default=0.25,
                   help="Slow tail amplitude as a fraction of single-AP amplitude per excess burst spike")

    # Cell morphology
    p.add_argument("--F0",             type=float, default=1.0)
    p.add_argument("--F0_cv",          type=float, default=0.35)
    p.add_argument("--cell_gain",      type=float, default=1.0)
    p.add_argument("--cell_gain_cv",   type=float, default=0.50)
    p.add_argument("--inactive_frac",  type=float, default=0.15)
    p.add_argument("--inactive_count", type=int, default=100,
                   help="Explicit number of inactive cells to render")
    p.add_argument("--cell_rad",       type=float, nargs=2, default=[2.0, 3.5])
    p.add_argument("--psf_sigma",      type=float, default=0.9)

    # Neuropil
    p.add_argument("--np_bleed",       type=float, default=0.20)
    p.add_argument("--np_level",       type=float, default=0.1)

    # Motion
    p.add_argument("--jitter_std",     type=float, default=0.5)
    p.add_argument("--jump_px",        type=float, default=2.0)
    p.add_argument("--jump_rate",      type=float, default=0.05)

    # Noise
    p.add_argument("--read_noise",     type=float, default=0.10)
    p.add_argument("--shot_coeff",     type=float, default=0.35)
    p.add_argument("--adc_bit_depth",  type=int,   default=13)
    p.add_argument("--counts_per_unit", type=float, default=660.0)

    return p.parse_args()


def parse_onset_list(text: str) -> list[float]:
    onsets = []
    for raw_part in text.replace(";", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = [p.strip() for p in part.split(":")]
        if len(fields) == 1:
            onsets.append(float(fields[0]))
        elif len(fields) == 3:
            start, step, stop = (float(value) for value in fields)
            if step <= 0 or stop < start:
                raise ValueError(f"Invalid onset range: {part}")
            n_steps = int(np.floor((stop - start) / step + 1e-9))
            onsets.extend(start + step * i for i in range(n_steps + 1))
        else:
            raise ValueError(f"Invalid onset range: {part}")
    if not onsets:
        raise ValueError("At least one modulation onset is required.")
    return sorted(set(round(onset, 9) for onset in onsets))


def main():
    args = parse_args()
    dt = 1.0 / args.fps

    print(f"[2P Sim] {args.n_units} units, {args.duration}s @ {args.fps}Hz → {args.output}")

    # ── Spike generation ─────────────────────────────────────────────────────
    mods = []
    if not args.no_mod:
        active_duration = args.mod_active_duration or args.mod_duration
        if args.mod_onsets:
            epochs = [(t_on, t_on + active_duration) for t_on in parse_onset_list(args.mod_onsets)]
        else:
            epochs = args.mod_epoch or [(args.mod_t_on, args.mod_t_on + active_duration)]
        for t_on, t_off in epochs:
            mods.append(ModEpoch(
                t_on=t_on,
                t_off=t_off,
                peak_rate_range=tuple(args.mod_peak),
                onset_jitter=args.mod_onset_jitter,
                offset_jitter=args.mod_offset_jitter,
                clamp_duration=args.mod_lock_duration,
                profile="cosine",
            ))

    spike_opts = SpikeGenOpts(
        n_units=args.n_units,
        T=args.duration,
        baseline_rate=tuple(args.baseline_rate),
        dt=dt,
        CV=args.cv,
        burst_alpha=args.burst_alpha,
        seed=args.seed,
        mod=mods,
        lock_onset_jitter_per_unit=args.mod_lock_onset_jitter,
    )
    print("[2P Sim] Generating spikes…")
    spike_result = generate_spikes(spike_opts)

    # ── Calcium dynamics ─────────────────────────────────────────────────────
    ind = INDICATOR_PRESETS[args.indicator]
    cal_params = CalciumParams(
        dt=dt,
        burst_tail_enabled=args.burst_tail,
        burst_tail_window=args.burst_tail_window,
        burst_tail_threshold=args.burst_tail_threshold,
        burst_tail_tau=args.burst_tail_tau,
        burst_tail_scale=args.burst_tail_scale,
        **ind,
    )
    print(f"[2P Sim] Computing calcium traces ({args.indicator})…")
    C = spikes_to_calcium(spike_result.spike_times, args.duration, cal_params)

    # ── Stack rendering ──────────────────────────────────────────────────────
    sim_params = SimParams(
        img_size=(args.img_h, args.img_w),
        F0=args.F0,
        F0_cv=args.F0_cv,
        cell_gain=args.cell_gain,
        cell_gain_cv=args.cell_gain_cv,
        inactive_frac=args.inactive_frac,
        inactive_count=args.inactive_count,
        cell_rad_px=tuple(args.cell_rad),
        psf_sigma=args.psf_sigma,
        neuropil_bleed_frac=args.np_bleed,
        neuropil_level=args.np_level,
        jitter_std=args.jitter_std,
        jump_px=args.jump_px,
        jump_rate=args.jump_rate,
        read_noise=args.read_noise,
        shot_coeff=args.shot_coeff,
        uint16_bit_depth=args.adc_bit_depth,
        uint16_counts_per_unit=args.counts_per_unit,
        seed=args.seed,
    )

    T_frames = C.shape[1]
    last_pct = [-1]

    def progress_cb(k, total):
        pct = int(100 * k / total)
        if pct != last_pct[0] and pct % 5 == 0:
            print(f"\r[2P Sim] Rendering {pct}%…", end="", flush=True)
            last_pct[0] = pct

    print("[2P Sim] Rendering stack…")
    result = simulate_stack(
        C=C,
        dt=dt,
        params=sim_params,
        extract_traces=False,
        return_movie=True,
        progress_cb=progress_cb,
    )
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"[2P Sim] Saving → {args.output}")
    result.spike_times = spike_result.spike_times
    result.rate_per_unit = spike_result.rate_per_unit
    result.rate_t = spike_result.rate_t
    result.baseline_rate_per_unit = spike_result.baseline_rate_per_unit
    result.mod_windows = spike_result.mod_windows

    save_tiff(result.movie_uint16, args.output)
    print(f"[2P Sim] Done.  Shape: {result.movie_uint16.shape}  "
          f"uint16_max={result.uint16_max:.4f}")


if __name__ == "__main__":
    main()
