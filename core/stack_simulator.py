"""
stack_simulator.py
==================
Python port of simulate_2p_stack_from_calcium_v3.m

Renders a realistic 2-photon TIFF stack from calcium traces:
  - Cell placement (with irregular shapes, PSF blurring)
  - Neuropil background field
  - Neuropil bleed into ROIs
  - Drift-free motion (AR(1) jitter + snap-back jumps)
  - Shot noise + read noise
  - Optional ΔF/F trace extraction
  - uint16 TIFF export
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# ─────────────────────────────────────────────────────────────────────────────
# Parameter dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimParams:
    # Image
    img_size: tuple[int, int] = (128, 256)   # (H, W)

    # Cell baseline
    F0: float = 1.0
    cell_gain: float = 1.0
    F0_cv: float = 0.35
    cell_gain_cv: float = 0.50
    inactive_frac: float = 0.15
    inactive_count: Optional[int] = 100

    # Cell morphology
    cell_rad_px: tuple[float, float] = (2.0, 3.5)
    cell_min_sep: float = 1.5
    shape_irreg_amp: float = 0.10
    shape_irreg_k: tuple[int, int] = (2, 5)
    donut_sigma: float = 0.0
    donut_contrast: float = 0.0
    psf_sigma: float = 0.9
    morphology_mode: str = "soma"       # soma, soma_process, dendrite
    process_prob: float = 0.50          # soma_process only: fraction of somas with visible processes
    process_count: int = 2              # axons/processes per ROI in process modes
    process_diameter_px: float = 0.65
    process_length_px: tuple[float, float] = (45.0, 180.0)
    process_orientation_deg: float = 90.0
    process_orientation_jitter_deg: float = 55.0
    process_continuity: float = 0.75    # 0 = short/gappy, 1 = spans toward FOV edge
    process_F0_scale: float = 0.10
    process_gain_scale: float = 0.30
    process_flow_speed_px_s: float = 60.0
    process_flow_bins: int = 8
    varicosity_density_per_px: float = 0.018
    varicosity_sigma_px: float = 0.85
    varicosity_strength: float = 3.0

    # Neuropil
    neuropil_blobs: int = 10
    neuropil_sigma_px: tuple[float, float] = (15.0, 35.0)
    neuropil_amp: float = 0.0
    neuropil_freq: float = 0.0
    neuropil_noise: float = 0.03
    neuropil_bleed_frac: float = 0.20
    neuropil_level: Optional[float] = 0.1   # None = 0.25 * F0

    # Motion (drift-free)
    jitter_std: float = 0.5     # px, AR(1) amplitude
    jitter_tau: float = 0.05    # s, AR(1) correlation time
    jump_px: float = 2.0        # px, saccade amplitude
    jump_rate: float = 0.05     # Hz
    jump_hold_sec: float = 0.02 # s

    # Noise
    read_noise: float = 0.10
    shot_coeff: float = 0.35

    # Detector digitization
    uint16_counts_per_unit: Optional[float] = 660.0
    uint16_bit_depth: int = 13

    # Seed
    seed: Optional[int] = None


@dataclass
class SimResult:
    movie: Optional[np.ndarray]          # (H, W, T) float64
    movie_uint16: Optional[np.ndarray]   # (T, H, W) uint16
    t: np.ndarray
    C: np.ndarray                        # (N_active, T) clean calcium
    masks: list[np.ndarray]
    cell_xyr: np.ndarray                 # (N_total, 3) x,y,r
    active_idx: np.ndarray
    inactive_idx: np.ndarray
    neuropil_base: np.ndarray
    neuropil_trace: np.ndarray
    motion_xy: np.ndarray                # (T, 2)
    uint16_scale: float
    uint16_max: float
    # extracted traces (may be None)
    F_cells: Optional[np.ndarray] = None
    Fraw: Optional[np.ndarray] = None
    dFF: Optional[np.ndarray] = None
    F0_per_cell: Optional[np.ndarray] = None
    cell_F0_per_cell: Optional[np.ndarray] = None
    cell_gain_per_cell: Optional[np.ndarray] = None
    process_F0_per_cell: Optional[np.ndarray] = None
    process_gain_per_cell: Optional[np.ndarray] = None
    soma_masks: Optional[list[np.ndarray]] = None
    process_masks: Optional[list[np.ndarray]] = None
    # spike-generation ground truth (attached by GUI/headless pipelines)
    spike_times: Optional[list[np.ndarray]] = None
    rate_t: Optional[np.ndarray] = None
    rate_per_unit: Optional[np.ndarray] = None
    baseline_rate_per_unit: Optional[np.ndarray] = None
    mod_windows: Optional[list[dict]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Main simulator
# ─────────────────────────────────────────────────────────────────────────────

def simulate_stack(
    C: np.ndarray,
    dt: float,
    params: SimParams,
    on_frame: Optional[Callable[[int, np.ndarray], None]] = None,
    extract_traces: bool = True,
    return_movie: bool = True,
    baseline_prct: float = 8.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> SimResult:
    """
    Build a 2P movie from pre-computed calcium traces C (N_active × T).

    Parameters
    ----------
    C            : (N_active, T) calcium traces (ΔF/F units)
    dt           : frame period (s)
    params       : SimParams
    on_frame     : optional callback(frame_idx, frame_array) called each frame
    extract_traces : compute dFF from simulated movie
    return_movie : keep full float64 movie in memory
    baseline_prct: percentile for F0 baseline estimation
    progress_cb  : optional callback(current_frame, total_frames)
    """
    rng = np.random.default_rng(params.seed)

    N_active, T = C.shape
    H, W = params.img_size
    t = np.arange(T) * dt

    # ── Place cells ──────────────────────────────────────────────────────────
    if params.inactive_count is None:
        N_inactive = round(params.inactive_frac * N_active)
    else:
        N_inactive = max(0, int(params.inactive_count))
    N_total = N_active + N_inactive
    (cell_xyr, mask_list, soma_mask_list, process_mask_list,
     soma_tmpl_list, process_tmpl_list, process_bin_tmpl_list,
     process_length_per_cell) = _place_cells(
        N_total, H, W, params.cell_rad_px, params.cell_min_sep,
        params.shape_irreg_amp, params.shape_irreg_k,
        params.donut_sigma, params.donut_contrast, params.psf_sigma,
        params.morphology_mode, params.process_prob, params.process_count, params.process_diameter_px,
        params.process_length_px, params.process_orientation_deg,
        params.process_orientation_jitter_deg, params.process_continuity,
        params.varicosity_density_per_px, params.varicosity_sigma_px,
        params.varicosity_strength, params.process_flow_bins, rng,
    )
    if len(mask_list) != N_total:
        placed_total = len(mask_list)
        if placed_total < N_active:
            warnings.warn(
                f"Only {placed_total}/{N_active} active ROIs were placed; truncating traces."
            )
            C = C[:placed_total]
            N_active = placed_total
        N_inactive = max(0, placed_total - N_active)
        N_total = placed_total

    active_idx = np.arange(N_active)
    inactive_idx = np.arange(N_active, N_total)

    # ── Per-cell baseline & gain ─────────────────────────────────────────────
    F0_per_cell = _sample_positive_per_cell(params.F0, params.F0_cv, N_total, rng)
    G_per_cell = _sample_positive_per_cell(params.cell_gain, params.cell_gain_cv, N_total, rng)
    use_soma = any(mask.any() for mask in soma_mask_list)
    use_process = any(mask.any() for mask in process_mask_list)
    process_F0_per_cell = _sample_positive_per_cell(
        params.F0 * params.process_F0_scale, params.F0_cv, N_total, rng
    ) if use_process else np.zeros(N_total, dtype=float)
    process_G_per_cell = _sample_positive_per_cell(
        params.cell_gain * params.process_gain_scale, params.cell_gain_cv, N_total, rng
    ) if use_process else np.zeros(N_total, dtype=float)
    G_per_cell[inactive_idx] = 0.0  # inactive cells: no dynamic drive
    process_G_per_cell[inactive_idx] = 0.0

    # ── Neuropil ─────────────────────────────────────────────────────────────
    neuropil_base = _make_neuropil_field(H, W, params.neuropil_blobs,
                                         params.neuropil_sigma_px, rng)
    neuropil_base = _normalize_map(neuropil_base)

    np_trace = 1.0 + params.neuropil_amp * np.sin(
        2 * np.pi * params.neuropil_freq * t + 2 * np.pi * rng.random()
    )
    win = min(T, max(1, round(0.5 / dt)))
    np_noise = np.convolve(rng.standard_normal(T), np.ones(win) / win, mode="same")
    np_trace = np_trace + params.neuropil_noise * np_noise

    neuropil_level = params.neuropil_level
    if neuropil_level is None:
        neuropil_level = 0.25 * params.F0

    # ── Vectorised templates ─────────────────────────────────────────────────
    Tsoma = np.zeros((H * W, N_total)) if use_soma else None
    Tprocess = np.zeros((H * W, N_total)) if use_process else None
    Tcells = np.zeros((H * W, N_total))
    Tcells_l2 = np.zeros((H * W, N_total))
    n_flow_bins = max(1, int(params.process_flow_bins)) if use_process else 1
    use_process_flow = use_process and params.process_flow_speed_px_s > 0 and n_flow_bins > 1
    Tprocess_bins = (
        [np.zeros((H * W, N_total), dtype=np.float32) for _ in range(n_flow_bins)]
        if use_process_flow else []
    )
    for i in range(N_total):
        if use_soma:
            Tsoma[:, i] = soma_tmpl_list[i].ravel()
        if use_process:
            Tprocess[:, i] = process_tmpl_list[i].ravel()
            Tcells[:, i] = Tprocess[:, i] if not use_soma else Tsoma[:, i] + Tprocess[:, i]
        else:
            Tcells[:, i] = Tsoma[:, i]
        norm = np.linalg.norm(Tcells[:, i])
        if norm > 0:
            Tcells_l2[:, i] = Tcells[:, i] / norm
        if use_process_flow:
            for b in range(n_flow_bins):
                Tprocess_bins[b][:, i] = process_bin_tmpl_list[i][b].ravel()

    # ── Motion path ──────────────────────────────────────────────────────────
    motion_xy = _make_motion_trace(
        t, params.jitter_std, params.jitter_tau,
        params.jump_px, params.jump_rate, params.jump_hold_sec, rng,
    )

    # ── Pad calcium for inactive cells ───────────────────────────────────────
    if N_inactive > 0:
        C_pad = np.vstack([C, np.zeros((N_inactive, T))])
    else:
        C_pad = C
    C_process_bins = (
        _make_flow_shifted_traces(
            C_pad, process_length_per_cell, params.process_flow_speed_px_s, dt, n_flow_bins
        )
        if use_process_flow else None
    )

    # ── Extraction buffers ───────────────────────────────────────────────────
    Fcells = np.zeros((N_total, T)) if extract_traces else None
    Fraw = np.zeros((N_total, T)) if extract_traces else None
    movie = np.zeros((H, W, T), dtype=np.float64) if return_movie else None
    mxAll = 0.0

    w0 = F0_per_cell
    g = G_per_cell
    proc_w0 = process_F0_per_cell
    proc_g = process_G_per_cell
    process_baseline_img = Tprocess @ proc_w0 if use_process else 0.0

    for k in range(T):
        Cflat = Tsoma @ (w0 + g * C_pad[:, k]) if use_soma else np.zeros(H * W, dtype=np.float64)
        if use_process:
            if use_process_flow:
                process_dyn = np.zeros(H * W, dtype=np.float64)
                for b in range(n_flow_bins):
                    process_dyn += Tprocess_bins[b] @ (proc_g * C_process_bins[b, :, k])
            else:
                process_dyn = Tprocess @ (proc_g * C_pad[:, k])
            Cflat = Cflat + process_baseline_img + process_dyn
        Cimg = Cflat.reshape(H, W)

        # Neuropil frame (visible background)
        np_frame = (neuropil_level * np_trace[k]) * (H * W) * neuropil_base

        # Composite: cells are added on top of the already-defined neuropil.
        F = np_frame + Cimg

        # Motion
        dx, dy = motion_xy[k]
        if dx != 0 or dy != 0:
            F = _translate(F, dx, dy)

        # Noise
        if params.shot_coeff != 0:
            F = F + params.shot_coeff * np.sqrt(np.maximum(F, 0)) * rng.standard_normal((H, W))
        if params.read_noise != 0:
            F = F + params.read_noise * rng.standard_normal((H, W))
        F = np.maximum(F, 0.0)

        mxAll = max(mxAll, F.max())

        # Extract traces (on motion-corrected frame)
        if extract_traces:
            Fr = F
            if dx != 0 or dy != 0:
                Fr = _translate(F, -dx, -dy)
            Fcells[:, k] = Tcells.T @ Fr.ravel()
            Fraw[:, k] = Tcells_l2.T @ Fr.ravel()

        if return_movie:
            movie[:, :, k] = F

        if on_frame is not None:
            try:
                on_frame(k, F)
            except Exception as e:
                warnings.warn(f"on_frame error at frame {k}: {e}")

        if progress_cb is not None:
            progress_cb(k + 1, T)

    # ── ΔF/F ─────────────────────────────────────────────────────────────────
    dFF = None
    F0_est = None
    if extract_traces:
        F0_est = np.percentile(Fcells, baseline_prct, axis=1, keepdims=True)
        dFF = (Fcells - F0_est) / np.maximum(F0_est, 1e-9)

    # ── uint16 scaling ────────────────────────────────────────────────────────
    if params.uint16_counts_per_unit is None:
        uint16_max = max(1.0, mxAll)
        uint16_scale = 65535.0 / uint16_max
        clip_max = 65535.0
    else:
        bit_depth = int(np.clip(params.uint16_bit_depth, 1, 16))
        clip_max = float((2 ** bit_depth) - 1)
        uint16_max = clip_max
        uint16_scale = float(params.uint16_counts_per_unit)

    movie_uint16 = None
    if return_movie and movie is not None:
        clipped = np.clip(movie * uint16_scale, 0, clip_max)
        movie_uint16 = clipped.astype(np.uint16).transpose(2, 0, 1)  # (T, H, W)

    return SimResult(
        movie=movie,
        movie_uint16=movie_uint16,
        t=t,
        C=C,
        masks=mask_list,
        cell_xyr=cell_xyr,
        active_idx=active_idx,
        inactive_idx=inactive_idx,
        neuropil_base=neuropil_base,
        neuropil_trace=np_trace,
        motion_xy=motion_xy,
        uint16_scale=uint16_scale,
        uint16_max=uint16_max,
        F_cells=Fcells,
        Fraw=Fraw,
        dFF=dFF,
        F0_per_cell=F0_est.squeeze() if F0_est is not None else None,
        cell_F0_per_cell=F0_per_cell,
        cell_gain_per_cell=G_per_cell,
        process_F0_per_cell=process_F0_per_cell,
        process_gain_per_cell=process_G_per_cell,
        soma_masks=soma_mask_list,
        process_masks=process_mask_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TIFF export
# ─────────────────────────────────────────────────────────────────────────────

def save_tiff(
    movie_uint16: np.ndarray,
    path: str | Path,
    bigtiff: bool = False,
) -> None:
    """Save (T, H, W) uint16 array as multi-page TIFF."""
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for TIFF export: pip install tifffile")
    tifffile.imwrite(
        str(path),
        movie_uint16,
        photometric="minisblack",
        compression=None,
        bigtiff=bigtiff,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _place_cells(N, H, W, rad_range, min_sep, irr_amp, irr_k, donut_sigma,
                 donut_contrast, psf_sigma, morphology_mode, process_prob, process_count,
                 process_diameter_px, process_length_px, process_orientation_deg,
                 process_orientation_jitter_deg, process_continuity,
                 varicosity_density_per_px, varicosity_sigma_px,
                 varicosity_strength, process_flow_bins, rng):
    """Place soma/process/dendrite ROIs and return masks plus compartment templates."""
    mode = (morphology_mode or "soma").lower()
    has_soma = mode != "dendrite"
    has_process = mode in {"soma_process", "soma+process", "soma + processes", "dendrite"}
    xyR = np.zeros((N, 3))
    soma_xyR = np.zeros((N, 3))
    masks = []
    soma_masks = []
    process_masks = []
    soma_tmpls = []
    process_tmpls = []
    process_bin_tmpls = []
    process_lengths = []
    n_flow_bins = max(1, int(process_flow_bins))
    ctr = 0
    max_tries = max(5000, 20 * N) if has_soma else N

    for _ in range(max_tries):
        if ctr >= N:
            break

        if has_soma:
            r = rng.uniform(rad_range[0], rad_range[1])
            x0 = rng.uniform(r + 2, W - r - 2)
            y0 = rng.uniform(r + 2, H - r - 2)

            ok = True
            for j in range(ctr):
                d = np.hypot(x0 - soma_xyR[j, 0], y0 - soma_xyR[j, 1])
                if d < min_sep * max(r, soma_xyR[j, 2]):
                    ok = False
                    break
            if not ok:
                continue
            soma_raw, soma_mask = _make_soma_template(
                H, W, x0, y0, r, irr_amp, irr_k, donut_sigma,
                donut_contrast, psf_sigma, rng,
            )
        else:
            r = max(1.0, process_diameter_px * 2.0)
            x0 = rng.uniform(0, W - 1)
            y0 = rng.uniform(0, H - 1)
            soma_raw = np.zeros((H, W), dtype=float)
            soma_mask = np.zeros((H, W), dtype=bool)

        process_raw = np.zeros((H, W), dtype=float)
        process_bins = np.zeros((n_flow_bins, H, W), dtype=float) if has_process else None
        branch_lengths = []
        draw_process = has_process and (not has_soma or rng.random() <= float(np.clip(process_prob, 0.0, 1.0)))
        if draw_process:
            n_process = max(1, int(process_count))
            for _branch in range(n_process):
                theta = np.deg2rad(
                    process_orientation_deg
                    + rng.normal(0.0, max(0.0, process_orientation_jitter_deg))
                )
                branch_raw, branch_bins, branch_length = _make_process_template(
                    H, W, x0, y0, theta, process_diameter_px, process_length_px,
                    process_continuity, psf_sigma, centered=True,
                    n_flow_bins=n_flow_bins,
                    varicosity_density_per_px=varicosity_density_per_px,
                    varicosity_sigma_px=varicosity_sigma_px,
                    varicosity_strength=varicosity_strength,
                    rng=rng,
                )
                process_raw += branch_raw
                process_bins += branch_bins
                branch_lengths.append(branch_length)

        soma_raw = _normalize_template(soma_raw)
        if has_process:
            process_raw, process_bins = _normalize_process_templates(process_raw, process_bins)
        else:
            process_bins = np.zeros((1, H, W), dtype=float)
        soma_mask_out = _template_mask(soma_raw)
        process_mask_out = _template_mask(process_raw)
        mask = soma_mask_out | process_mask_out
        if not mask.any():
            mask = soma_mask

        ys, xs = np.where(mask)
        if len(xs):
            x_c = float(np.median(xs))
            y_c = float(np.median(ys))
            r_eff = float(np.sqrt(len(xs) / np.pi))
        else:
            x_c, y_c, r_eff = x0, y0, r
        if has_soma:
            xyR[ctr] = [x0, y0, r]
            soma_xyR[ctr] = [x0, y0, r]
        else:
            xyR[ctr] = [x_c, y_c, r_eff]
            soma_xyR[ctr] = [x_c, y_c, r_eff]

        masks.append(mask.astype(bool))
        soma_masks.append(soma_mask_out.astype(bool))
        process_masks.append(process_mask_out.astype(bool))
        soma_tmpls.append(soma_raw)
        process_tmpls.append(process_raw)
        process_bin_tmpls.append(process_bins)
        process_lengths.append(float(np.mean(branch_lengths)) if branch_lengths else 0.0)
        ctr += 1

    if ctr < N:
        warnings.warn(f"Placed only {ctr}/{N} ROIs; loosen morphology placement params.")
        xyR = xyR[:ctr]
        masks = masks[:ctr]
        soma_masks = soma_masks[:ctr]
        process_masks = process_masks[:ctr]
        soma_tmpls = soma_tmpls[:ctr]
        process_tmpls = process_tmpls[:ctr]
        process_bin_tmpls = process_bin_tmpls[:ctr]
        process_lengths = process_lengths[:ctr]

    return (
        xyR, masks, soma_masks, process_masks, soma_tmpls, process_tmpls,
        process_bin_tmpls, np.asarray(process_lengths, dtype=float),
    )


def _make_soma_template(H, W, x0, y0, r, irr_amp, irr_k, donut_sigma,
                        donut_contrast, psf_sigma, rng):
    th = np.linspace(0, 2 * np.pi, 256)
    ks = rng.integers(irr_k[0], irr_k[1] + 1, 3)
    pert = 1 + irr_amp * (
        0.6 * np.sin(ks[0] * th + 2 * np.pi * rng.random()) +
        0.3 * np.sin(ks[1] * th + 2 * np.pi * rng.random()) +
        0.1 * np.sin(ks[2] * th + 2 * np.pi * rng.random())
    )
    rr = r * pert
    xs = x0 + rr * np.cos(th)
    ys = y0 + rr * np.sin(th)
    mask = _poly2mask(xs, ys, H, W)

    YY, XX = np.mgrid[0:H, 0:W]
    D = np.hypot(XX - x0, YY - y0)
    ring = np.exp(-0.5 * ((D - r) ** 2) / max(1e-9, donut_sigma ** 2)) * mask
    center = np.exp(-0.5 * D ** 2 / max(1e-9, (0.6 * r) ** 2)) * mask
    raw = (1 + donut_contrast) * ring + (1 - donut_contrast) * center
    if psf_sigma > 0:
        raw = gaussian_filter(raw.astype(float), sigma=psf_sigma)
    return np.maximum(raw, 0.0), mask.astype(bool)


def _make_process_template(H, W, x0, y0, theta, diameter_px, length_range,
                           continuity, psf_sigma, centered, n_flow_bins,
                           varicosity_density_per_px, varicosity_sigma_px,
                           varicosity_strength, rng):
    continuity = float(np.clip(continuity, 0.0, 1.0))
    diameter_px = max(0.2, float(diameter_px))
    min_len, max_len = sorted((float(length_range[0]), float(length_range[1])))
    base_len = rng.uniform(max(1.0, min_len), max(max_len, min_len + 1.0))
    edge_len_fwd = _distance_to_edge(x0, y0, theta, W, H)
    edge_len_back = _distance_to_edge(x0, y0, theta + np.pi, W, H)
    if centered:
        split = rng.uniform(0.35, 0.65)
        fwd_len = (1 - continuity) * base_len * split + continuity * edge_len_fwd
        back_len = (1 - continuity) * base_len * (1 - split) + continuity * edge_len_back
        direction = np.array([np.cos(theta), np.sin(theta)])
        start = np.array([x0, y0]) - back_len * direction
        end = np.array([x0, y0]) + fwd_len * direction
    else:
        length = (1 - continuity) * base_len + continuity * edge_len_fwd
        start = np.array([x0, y0])
        end = start + length * np.array([np.cos(theta), np.sin(theta)])

    length = max(1.0, float(np.hypot(*(end - start))))
    n = max(8, int(length * 2))
    u = np.linspace(0, 1, n)
    mid = 0.5 * (start + end)
    perp = np.array([-np.sin(theta), np.cos(theta)])
    curve_amp = rng.normal(0.0, 0.08 * length)
    points = (1 - u)[:, None] * start + u[:, None] * end
    wobble = rng.standard_normal(n)
    wobble_win = max(5, int(round(0.12 * n)))
    wobble = np.convolve(wobble, np.ones(wobble_win) / wobble_win, mode="same")
    wobble = wobble / max(float(np.std(wobble)), 1e-9)
    lateral = np.sin(np.pi * u) * (curve_amp + 0.035 * length * wobble)
    points += lateral[:, None] * perp

    keep_prob = 0.25 + 0.75 * continuity
    keep = rng.random(n) < keep_prob
    if continuity >= 0.98:
        keep[:] = True
    if not keep.any():
        keep[rng.integers(0, n)] = True

    gain_sigma = 0.45
    path_gain = rng.lognormal(-0.5 * gain_sigma ** 2, gain_sigma, n)
    smooth = max(3, int(round(0.04 * n)))
    if smooth > 1:
        path_gain = np.convolve(path_gain, np.ones(smooth) / smooth, mode="same")
    path_gain = path_gain / max(float(np.mean(path_gain)), 1e-9)

    raw = np.zeros((H, W), dtype=float)
    bins = np.zeros((n_flow_bins, H, W), dtype=float)
    rr = max(0.5, diameter_px / 2.0)
    pix_rad = int(np.ceil(rr + 1.0))
    kept_u = u[keep]
    for (x, y), ui in zip(points[keep], kept_u):
        xi = int(round(x))
        yi = int(round(y))
        x1, x2 = max(0, xi - pix_rad), min(W, xi + pix_rad + 1)
        y1, y2 = max(0, yi - pix_rad), min(H, yi + pix_rad + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        YY, XX = np.mgrid[y1:y2, x1:x2]
        d = np.hypot(XX - x, YY - y)
        amp = path_gain[int(np.clip(round(ui * (n - 1)), 0, n - 1))]
        spot = amp * np.exp(-0.5 * (d / max(0.35, rr)) ** 2)
        raw[y1:y2, x1:x2] += spot
        bin_idx = int(np.clip(np.floor(ui * n_flow_bins), 0, n_flow_bins - 1))
        bins[bin_idx, y1:y2, x1:x2] += spot

    n_varicosities = rng.poisson(max(0.0, float(varicosity_density_per_px)) * length)
    if n_varicosities > 0 and len(points) > 0 and varicosity_strength > 0:
        sigma = max(0.35, float(varicosity_sigma_px))
        pix_rad = int(np.ceil(3.0 * sigma))
        path_idx = rng.integers(0, len(points), size=n_varicosities)
        for idx in path_idx:
            x, y = points[idx]
            ui = u[idx]
            xi = int(round(x))
            yi = int(round(y))
            x1, x2 = max(0, xi - pix_rad), min(W, xi + pix_rad + 1)
            y1, y2 = max(0, yi - pix_rad), min(H, yi + pix_rad + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            YY, XX = np.mgrid[y1:y2, x1:x2]
            d = np.hypot(XX - x, YY - y)
            bead = float(varicosity_strength) * np.exp(-0.5 * (d / sigma) ** 2)
            raw[y1:y2, x1:x2] += bead
            bin_idx = int(np.clip(np.floor(ui * n_flow_bins), 0, n_flow_bins - 1))
            bins[bin_idx, y1:y2, x1:x2] += bead

    blur = max(psf_sigma * 0.6, diameter_px * 0.25)
    if blur > 0:
        raw = gaussian_filter(raw, sigma=blur)
        for b in range(n_flow_bins):
            bins[b] = gaussian_filter(bins[b], sigma=blur)
    return np.maximum(raw, 0.0), np.maximum(bins, 0.0), length


def _distance_to_edge(x, y, theta, W, H):
    dx = np.cos(theta)
    dy = np.sin(theta)
    candidates = []
    if abs(dx) > 1e-9:
        candidates.extend([(0 - x) / dx, ((W - 1) - x) / dx])
    if abs(dy) > 1e-9:
        candidates.extend([(0 - y) / dy, ((H - 1) - y) / dy])
    candidates = [c for c in candidates if c > 0]
    return max(1.0, min(candidates) if candidates else max(H, W))


def _normalize_template(raw):
    """Normalize spatial footprints by local peak brightness, not total area."""
    raw = np.maximum(raw, 0.0)
    peak = raw.max()
    if peak > 0:
        return raw / peak
    return raw


def _normalize_process_templates(raw, bins):
    raw = np.maximum(raw, 0.0)
    bins = np.maximum(bins, 0.0)
    peak = raw.max()
    if peak <= 0:
        return raw, bins
    return raw / peak, bins / peak


def _make_flow_shifted_traces(C, lengths_px, speed_px_s, dt, n_bins):
    n_bins = max(1, int(n_bins))
    N, T = C.shape
    out = np.zeros((n_bins, N, T), dtype=C.dtype)
    speed = float(speed_px_s)
    if speed <= 0 or n_bins == 1:
        out[:] = C[None, :, :]
        return out

    lengths = np.asarray(lengths_px, dtype=float)
    if lengths.size != N:
        lengths = np.resize(lengths, N)
    for b in range(n_bins):
        frac = 0.0 if n_bins == 1 else b / (n_bins - 1)
        delays = np.maximum(0, np.rint((frac * lengths) / max(speed, 1e-9) / dt).astype(int))
        for i, delay in enumerate(delays):
            if delay <= 0:
                out[b, i] = C[i]
            elif delay < T:
                out[b, i, delay:] = C[i, :T - delay]
    return out


def _template_mask(raw, frac=0.03):
    if raw is None or raw.size == 0 or raw.max() <= 0:
        return np.zeros_like(raw, dtype=bool)
    return raw > (float(frac) * float(raw.max()))


def _poly2mask(xs, ys, H, W):
    """Rasterise a polygon to a boolean mask."""
    from matplotlib.path import Path as MplPath
    poly = np.column_stack([xs, ys])
    path = MplPath(poly)
    YY, XX = np.mgrid[0:H, 0:W]
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    mask = path.contains_points(pts).reshape(H, W)
    return mask


def _make_neuropil_field(H, W, n_blobs, sig_range, rng):
    np_field = np.zeros((H, W))
    YY, XX = np.mgrid[0:H, 0:W]
    for _ in range(n_blobs):
        cx = rng.uniform(0, W)
        cy = rng.uniform(0, H)
        s = rng.uniform(sig_range[0], sig_range[1])
        g = np.exp(-0.5 * ((XX - cx) ** 2 + (YY - cy) ** 2) / max(1e-9, s ** 2))
        np_field += g
    smooth_sigma = np.mean(sig_range) * 0.75
    np_field = gaussian_filter(np_field, sigma=smooth_sigma)
    return np.maximum(np_field, 0.0)


def _normalize_map(M):
    M = np.maximum(M, 0.0)
    s = M.sum()
    if s > 0:
        M /= s
    return M


def _sample_positive_per_cell(mean, cv, n, rng):
    """Sample per-cell positive values with a specified mean and coefficient of variation."""
    mean = float(mean)
    cv = max(0.0, float(cv))
    if n <= 0:
        return np.array([], dtype=float)
    if cv == 0 or mean <= 0:
        return np.full(n, max(0.0, mean), dtype=float)
    sigma = np.sqrt(np.log1p(cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, n)


def _make_motion_trace(t, jitter_std, jitter_tau, jump_px, jump_rate, jump_hold_sec, rng):
    """Drift-free motion: zero-mean AR(1) jitter + snap-back jumps."""
    T = len(t)
    dt = float(t[1] - t[0]) if T > 1 else 1e-3
    rho = np.exp(-dt / max(jitter_tau, 1e-9))
    sigma_e = jitter_std * np.sqrt(max(0.0, 1 - rho ** 2))

    dx = np.zeros(T)
    dy = np.zeros(T)
    for k in range(1, T):
        dx[k] = rho * dx[k - 1] + sigma_e * rng.standard_normal()
        dy[k] = rho * dy[k - 1] + sigma_e * rng.standard_normal()

    dx -= dx.mean()
    dy -= dy.mean()

    # Snap-back jumps
    dur_f = max(1, round(jump_hold_sec / dt))
    lam = jump_rate * dt
    jump_mask = rng.random(T) < lam
    for k in np.where(jump_mask)[0]:
        th = rng.uniform(0, 2 * np.pi)
        jx = jump_px * np.cos(th)
        jy = jump_px * np.sin(th)
        k2 = min(T, k + dur_f)
        dx[k:k2] += jx
        dy[k:k2] += jy

    return np.column_stack([dx, dy])


def _translate(F, dx, dy):
    """Sub-pixel translation using scipy shift."""
    from scipy.ndimage import shift as nd_shift
    return nd_shift(F, shift=[-dy, -dx], order=1, mode="constant", cval=0.0)
