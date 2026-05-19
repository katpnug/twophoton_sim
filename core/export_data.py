from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from core.stack_simulator import SimResult


def export_sim_data(result: SimResult, path: str | Path) -> None:
    """Export simulated movie, traces, masks, and metadata for Python/MATLAB analysis."""
    path = Path(path)
    suffix = path.suffix.lower()
    payload = _result_payload(result)

    if suffix in {".npz", ".npy"}:
        if suffix == ".npy":
            np.save(path, payload, allow_pickle=True)
        else:
            np.savez_compressed(path, **payload)
        return

    if suffix == ".mat":
        savemat(path, payload, do_compression=True, long_field_names=True)
        return

    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5 export requires h5py: pip install h5py") from exc
        with h5py.File(path, "w") as h5:
            _write_h5_group(h5, payload)
        return

    raise ValueError("Unsupported export type. Use .npz, .npy, .mat, .h5, or .hdf5.")


def _result_payload(result: SimResult) -> dict[str, Any]:
    masks = _stack_masks(result.masks)
    soma_masks = _stack_masks(result.soma_masks)
    process_masks = _stack_masks(result.process_masks)
    spike_times = np.array(result.spike_times or [], dtype=object)
    mod_windows = np.array(result.mod_windows or [], dtype=object)
    active_idx = np.asarray(result.active_idx, dtype=np.int32)
    inactive_idx = np.asarray(result.inactive_idx, dtype=np.int32)

    payload: dict[str, Any] = {
        "movie": _none_to_empty(result.movie),
        "movie_uint16": _none_to_empty(result.movie_uint16),
        "t": result.t,
        "C": result.C,
        "masks": masks,
        "combined_masks": masks,
        "soma_masks": soma_masks,
        "process_masks": process_masks,
        "cell_xyr": result.cell_xyr,
        "active_idx": active_idx,
        "inactive_idx": inactive_idx,
        "neuropil_base": result.neuropil_base,
        "neuropil_trace": result.neuropil_trace,
        "motion_xy": result.motion_xy,
        "uint16_scale": np.array(result.uint16_scale),
        "uint16_max": np.array(result.uint16_max),
        "F_cells": _none_to_empty(result.F_cells),
        "Fraw": _none_to_empty(result.Fraw),
        "dFF": _none_to_empty(result.dFF),
        "F0_per_cell_extracted": _none_to_empty(result.F0_per_cell),
        "cell_F0_per_cell": _none_to_empty(result.cell_F0_per_cell),
        "cell_gain_per_cell": _none_to_empty(result.cell_gain_per_cell),
        "process_F0_per_cell": _none_to_empty(result.process_F0_per_cell),
        "process_gain_per_cell": _none_to_empty(result.process_gain_per_cell),
        "spike_times": spike_times,
        "rate_t": _none_to_empty(result.rate_t),
        "rate_per_unit": _none_to_empty(result.rate_per_unit),
        "baseline_rate_per_unit": _none_to_empty(result.baseline_rate_per_unit),
        "mod_windows": mod_windows,
    }
    return payload


def _none_to_empty(value):
    if value is None:
        return np.array([])
    return value


def _stack_masks(masks):
    if masks:
        return np.stack(masks).astype(np.uint8)
    return np.zeros((0, 0, 0), dtype=np.uint8)


def _write_h5_group(group, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.dtype == object:
            sub = group.create_group(key)
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    item_group = sub.create_group(str(i))
                    for sub_key, sub_value in item.items():
                        _write_h5_dataset(item_group, sub_key, sub_value)
                else:
                    _write_h5_dataset(sub, str(i), item)
        else:
            _write_h5_dataset(group, key, value)


def _write_h5_dataset(group, key: str, value: Any) -> None:
    if value is None:
        value = np.array([])
    if isinstance(value, str):
        group.create_dataset(key, data=np.bytes_(value))
        return
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.array([str(v) for v in arr], dtype="S")
    group.create_dataset(key, data=arr)
