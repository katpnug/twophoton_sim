# Two-Photon Simulation GUI

A PySide6 application and headless generator for creating synthetic two-photon calcium imaging recordings with known ground truth. The simulator is designed for testing and benchmarking segmentation, demixing, motion-correction, and trace-extraction workflows against realistic but fully controlled data.

The current GUI can simulate granule-cell-like soma recordings, soma plus axon/dendrite process recordings, and dendritic-branch-dominated recordings inspired by in vivo cerebellar two-photon examples.

This project was originally derived from MATLAB simulation scripts including `simgrc_2026_kpn.m` and `simulate_2p_stack_from_calcium_v3.m`.

## Highlights

- Interactive dark-theme GUI for configuring and previewing simulated recordings.
- Soma-only, soma + process, and dendritic-branch morphology modes.
- Active and inactive neuron populations with configurable firing-rate ranges.
- Multiple activity epochs with onset lists such as `1:5:30`, active duration, and onset/offset jitter.
- GCaMP-style calcium dynamics with selectable presets and an optional peak-preserving burst-tail model.
- Neuropil background, motion jitter/saccades, shot noise, read noise, and ADC quantization.
- Frame viewer with playback, zoom/pan, selectable ROIs, soma/process outline modes, outline width, and optional ROI fill.
- Trace viewer for extracted dF/F, raw ROI fluorescence, clean calcium ground truth, and spike rasters.
- Cell sorting by index, selected order, modulation onset, first spike, gain, baseline F0, and peak response.
- Export to TIFF and analysis data formats for Python and MATLAB.
- Import/export JSON parameter presets for reusing exact simulation settings.

## Example Data

During development, a local `examples/` folder can be used for real in vivo reference recordings from cerebellar imaging sessions:

- `granule_cells_8bit.tif`
- `molecular_layer_interneurons.gif`
- `purkinje_cell_dendrites_8bit.tif`

These examples motivated the current defaults and morphology controls. The granule-cell example is closest to the default soma-only settings. Molecular layer interneuron recordings motivate soma + process simulations, while Purkinje cell dendrite recordings motivate dendritic-branch simulations without cell bodies.

The `examples/` folder is ignored by default so private, large, or redistribution-restricted imaging data are not accidentally committed. If public example data are added later, remove or narrow the `examples/` rule in `.gitignore`.

## Installation

Clone or download the repository, create a Python environment, and install the dependencies:

```bash
# 1. Clone / download the repository
git clone https://github.com/katpnug/twophoton_sim.git
cd twophoton_sim

# 2. Create a virtual environment with venv
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Conda users can create an environment instead:

```bash
conda create -n 2psim python=3.11
conda activate 2psim
pip install -r requirements.txt
```

The project depends on:

- `PySide6`
- `numpy`
- `scipy`
- `matplotlib`
- `pyqtgraph`
- `tifffile`
- `h5py`

Python 3.10 or 3.11 is recommended.

## Running The GUI

From the repository root:

```bash
python main.py
```

The GUI is the primary interface and exposes the full current simulator feature set.

## Headless Generation

For batch generation with common soma-style parameters:

```bash
python run_headless.py --n_units 100 --duration 60 --fps 115 --indicator gcamp8f --output my_recording.tif
```

See all available command-line options:

```bash
python run_headless.py --help
```

Note: the headless script supports the common batch-generation path and calcium burst-tail options, but the GUI currently exposes the most complete set of morphology, process, dendrite, viewer, and export controls.

## Typical Workflow

1. Configure recording duration, frame rate, image size, and active/inactive neuron counts.
2. Define baseline firing-rate ranges and activity epochs.
3. Choose a calcium indicator preset and optional burst-tail behavior.
4. Select a morphology mode: soma only, soma + processes, or dendritic branches.
5. Tune neuropil, motion, and noise levels.
6. Optionally click `Save Params` to export the current settings as a reusable JSON preset.
7. Click `Generate` to simulate the recording.
8. Inspect the result in the frame viewer, trace viewer, histogram, neuropil map, and motion plot.
9. Export the movie as TIFF and/or save all analysis arrays using `Save Data`.

## GUI Controls

### Parameter Reference

Recording parameters:

- `N active neurons`: Number of modulated cells that can respond during activity epochs.
- `N inactive neurons`: Number of extra cells included in the movie with baseline fluorescence but no activity-epoch drive.
- `Duration`: Total simulated recording length in seconds.
- `Image H`: Output image height in pixels.
- `Image W`: Output image width in pixels.
- `Frame rate`: Simulated acquisition rate in frames per second.
- `Seed`: Random seed used to make placement, spikes, noise, and motion reproducible.

Spiking activity parameters:

- `Baseline rate min`: Lower bound of each cell's baseline firing-rate distribution.
- `Baseline rate max`: Upper bound of each cell's baseline firing-rate distribution.
- `ISI CV`: Controls spike-train irregularity, with `1` approximating Poisson-like intervals.
- `Burst alpha`: Hawkes/self-excitation strength that increases short-timescale burstiness.
- `Burst tau`: Time constant of the burst/self-excitation effect.
- `Enable epoch`: Turns activity-window modulation on or off.
- `Activity onsets`: List or range of activity start times, such as `1:5:30`.
- `Active Duration`: Duration of each activity window after onset.
- `Onset jitter`: Per-cell/window timing jitter applied to activity onset.
- `Lock onset jitter per cell`: Reuses each cell's onset-jitter offset across all activity epochs.
- `Offset jitter`: Per-cell/window timing jitter applied to activity offset when fixed duration is disabled.
- `Fixed duration after onset jitter`: Keeps each epoch duration fixed after onset jitter is applied.
- `Peak rate min`: Lower bound of the activity-epoch peak firing-rate distribution.
- `Peak rate max`: Upper bound of the activity-epoch peak firing-rate distribution.
- `Epoch profile`: Shape of the modulation window, currently `box` or `cosine`.

Calcium indicator parameters:

- `Preset`: Loads predefined calcium dynamics for common indicators or enables custom values.
- `tau rise`: Calcium transient rise time constant.
- `tau decay`: Calcium transient decay time constant.
- `Single-AP dF/F`: Peak clean calcium response amplitude for one action potential.
- `Prolong burst tail`: Adds a slow post-burst decay component without changing the burst peak.
- `Burst window`: Time window used to group spikes into a burst.
- `Burst threshold`: Minimum number of spikes required to recruit the prolonged burst tail.
- `Tail tau`: Decay time constant of the added post-burst tail.
- `Tail scale`: Strength of the post-burst tail relative to the burst event.

Cell morphology parameters:

- `Cell gain`: Mean local-brightness scale applied to each cell's activity-driven fluorescence.
- `F0 baseline`: Mean local baseline fluorescence for somatic compartments.
- `Gain CV`: Cell-to-cell coefficient of variation for activity gain.
- `F0 CV`: Cell-to-cell coefficient of variation for baseline fluorescence.
- `Morphology`: Selects soma-only, soma + process, or dendritic-branch ROI geometry.
- `Radius min`: Minimum soma radius in pixels.
- `Radius max`: Maximum soma radius in pixels.
- `Min separation`: Minimum soma-to-soma spacing in units of soma radius.
- `Shape irregularity`: Amount of non-circular soma boundary variation.
- `PSF sigma`: Gaussian blur width used to approximate optical point-spread smoothing.
- `Donut sigma`: Radius of ring-like fluorescence for hollow or donut-shaped somas.
- `Donut contrast`: Strength of the ring component relative to the soma center.
- `Process probability`: Fraction of soma-bearing ROIs that receive visible axon/dendrite processes.
- `Processes / ROI`: Number of process branches generated per process-bearing ROI.
- `Process diameter`: Approximate process thickness in pixels.
- `Length min`: Minimum generated process length in pixels.
- `Length max`: Maximum generated process length in pixels.
- `Orientation`: Preferred process direction in degrees.
- `Angle jitter`: Random angular spread around the preferred process orientation.
- `Continuity`: Controls whether processes are fragmented or span toward FOV edges.
- `Process F0 x`: Local baseline fluorescence multiplier for processes relative to soma `F0`.
- `Process gain x`: Local activity gain multiplier for processes relative to soma gain.
- `Flow speed`: Directional activity propagation speed along processes in pixels per second.
- `Flow bins`: Number of spatial bins used to model delayed process activity flow.
- `Varicosity density`: Expected number of bead-like varicosities per process pixel.
- `Varicosity size`: Spatial width of each varicosity in pixels.
- `Varicosity strength`: Brightness multiplier for varicosities along processes.

Neuropil parameters:

- `Blobs`: Number of smooth random components used to build the neuropil background.
- `Blob sigma min`: Minimum spatial scale of neuropil blobs in pixels.
- `Blob sigma max`: Maximum spatial scale of neuropil blobs in pixels.
- `Oscillation amp`: Amplitude of global neuropil brightness oscillation.
- `Oscillation freq`: Frequency of global neuropil oscillation in hertz.
- `Neuropil noise`: Amplitude of slow temporal neuropil fluctuations.
- `Bleed fraction`: Reserved neuropil contamination parameter for ROI/background modeling.
- `DC level`: Mean neuropil brightness level added behind cells.

Motion parameters:

- `Jitter sigma`: Standard deviation of continuous frame-to-frame motion in pixels.
- `Jitter tau`: Correlation time of the continuous motion jitter.
- `Saccade size`: Size of occasional jump-like motion events in pixels.
- `Saccade rate`: Frequency of jump-like motion events in hertz.
- `Saccade hold`: Duration that saccade-like displacements are held before returning.

Noise and digitization parameters:

- `Read noise sigma`: Additive camera/readout noise level.
- `Shot noise coeff`: Signal-dependent noise scale applied before digitization.
- `ADC bit depth`: Effective detector bit depth used for clipping the output range.
- `Counts scale`: Conversion factor from simulated fluorescence units to integer counts.

Viewer controls:

- `Load Params`: Imports a JSON parameter preset and restores the GUI controls.
- `Save Params`: Exports the current GUI controls as a reusable JSON parameter preset.
- `Colormap`: Selects the color lookup table used to display movie frames.
- `Auto contrast`: Rescales the displayed frame contrast for easier visual inspection.
- `Cell outlines`: Toggles ROI outline overlays in the frame viewer.
- `Outline`: Chooses whether overlays show combined ROIs, soma-only ROIs, or axon/process-only ROIs.
- `Width`: Sets the displayed ROI outline thickness.
- `Fill`: Fills ROI overlays instead of showing outlines only.
- `Clear`: Clears the current frame-viewer cell selection.
- `Reset view`: Restores the current viewer's pan/zoom limits.
- `Frame`: Selects the displayed movie frame with the bottom scrollbar.
- `Play/stop`: Starts or pauses frame playback in the frame viewer.
- `Show cells`: Controls how many traces or spike rows are displayed in the trace tab.
- `Trace source`: Selects dF/F, raw L2-normalized fluorescence, clean ground-truth calcium, or spike raster display.
- `Sort`: Orders displayed cells by index, selection order, onset timing, first spike, gain, baseline F0, or peak response.
- `Selected only`: Restricts the trace tab to cells selected in the frame viewer.
- `Overlay spikes`: Draws compact spike ticks underneath fluorescence or calcium traces.

### Parameter Presets

Use `Save Params` to write the current simulator controls to a `.json` file. Use `Load Params` to restore that file later or share the same configuration with another user.

The preset file stores recording, spiking, calcium, morphology, neuropil, motion, noise, and digitization settings. It does not store generated movie frames, extracted traces, masks, spike trains, or other simulation outputs; use `Save Data` for those arrays.

Parameter preset files are plain JSON, so they can be inspected, versioned, and edited outside the GUI if needed.

### Recording

Default recording parameters:

- `N active neurons`: `100`
- `N inactive neurons`: `100`
- `Duration`: `30 s`
- `Image H`: `128 px`
- `Image W`: `256 px`
- `Frame rate`: `115 Hz`
- `Seed`: `42`

Active neurons receive stimulus/activity epoch modulation. Inactive neurons remain in the recording as background biological units with baseline firing but no activity-epoch modulation.

### Spiking Activity

The spike generator supports heterogeneous baseline rates, repeated activity windows, and per-window temporal jitter.

Default spiking parameters:

- `Enable epoch`: on
- `Baseline min`: `0.5 Hz`
- `Baseline max`: `3 Hz`
- `ISI CV`: `1.0`
- `Burst alpha`: `0`
- `Burst tau`: `0.015 s`
- `Activity onsets`: `1:5:30`
- `Active Duration`: `0.25 s`
- `Onset jitter`: `0.1 s`
- `Lock onset jitter per cell`: off
- `Offset jitter`: `0 s`
- `Fixed duration after onset jitter`: off
- `Peak rate min`: `20 Hz`
- `Peak rate max`: `80 Hz`
- `Epoch profile`: `box`

`Activity onsets` accepts MATLAB-like range syntax:

```text
start:step:stop
```

For example, `1:5:30` creates activity windows starting at 1, 6, 11, 16, 21, and 26 seconds. Each window lasts for `Active Duration`, with optional onset and offset jitter applied per cell/window.

If `Lock onset jitter per cell` is enabled, each cell receives one onset-jitter offset and reuses that offset for every modulation epoch, preserving cell-specific response timing across repeated events. If it is disabled, each cell/window pair gets a new onset jitter sample. If `Fixed duration after onset jitter` is enabled, onset jitter shifts the window but the active duration stays fixed. If it is disabled, onset and offset jitter can independently change the exact window duration. The epoch profile can be `box` or `cosine`.

### Calcium Indicator

Built-in presets include:

- `GCaMP6f`
- `GCaMP6s`
- `GCaMP7f`
- `GCaMP8f`
- `GCaMP8m`
- `GCaMP8s`
- `Voltage (fast)`
- `Custom`

Current defaults use `GCaMP8f`:

- `tau_rise`: `0.007 s`
- `tau_decay`: `0.097 s`
- `single_ap_amp`: `0.41`

The base calcium trace is generated with a double-exponential AR(2)-style impulse response normalized to peak one and scaled by the selected single-action-potential amplitude.

The preset values are approximate single-AP somatic response settings from published or Janelia-reported measurements. Literature often reports half-decay time, while this simulator's `tau_decay` control is an exponential time constant, so half-decay values are converted with `tau = t_half / ln(2)`. `tau_rise` is used as a practical rise-shape parameter, not a perfect copy of every paper's half-rise or time-to-peak measurement.

| Preset | `tau_rise` | `tau_decay` | `single_ap_amp` | Basis |
| --- | ---: | ---: | ---: | --- |
| `GCaMP6f` | `0.026 s` | `0.202 s` | `0.22` | Chen et al. 2013 / GCaMP6 fast single-AP kinetics |
| `GCaMP6s` | `0.058 s` | `0.656 s` | `0.35` | Chen et al. 2013 / GCaMP6 slow single-AP kinetics |
| `GCaMP7f` | `0.025 s` | `0.262 s` | `0.21` | jGCaMP7f control values reported with jGCaMP8 screening |
| `GCaMP8f` | `0.007 s` | `0.097 s` | `0.41` | Janelia jGCaMP8f cultured-neuron 1AP screen |
| `GCaMP8m` | `0.007 s` | `0.171 s` | `0.76` | Janelia jGCaMP8m cultured-neuron 1AP screen |
| `GCaMP8s` | `0.010 s` | `0.442 s` | `1.11` | Janelia jGCaMP8s cultured-neuron 1AP screen |

These values are starting points, not absolutes. Indicator expression, temperature, cell type, imaging plane, extraction method, neuropil subtraction, and saturation can all change measured dF/F and kinetics. In the simulator, `single_ap_amp` is the clean latent calcium-trace peak before movie rendering, noise, neuropil, motion, and ROI extraction.

#### Prolonged Burst Tail

The optional `Prolong burst tail` control models slower decay after high-frequency spike bursts. It is intended to mimic nonlinear GCaMP behavior where large burst-driven calcium events can decay more slowly than isolated spikes.

Default burst-tail controls:

- `Burst window`: `0.20 s`
- `Burst threshold`: `3 spikes`
- `Tail tau`: `0.70 s`
- `Tail scale`: `0.25`

The burst-tail model is peak-preserving: it starts after a completed burst group, raises only the post-burst decay floor, leaves isolated spikes unchanged, and does not increase the original AR(2) peak amplitude.

### Cell Morphology

Default morphology parameters:

- `Morphology`: `Soma only`
- `Cell gain`: `1.0`
- `F0 baseline`: `1.0`
- `Gain CV`: `0.50`
- `F0 CV`: `0.35`
- `Radius min`: `2 px`
- `Radius max`: `3.5 px`
- `Min separation`: `1.5`
- `Shape irregularity`: `0.10`
- `PSF sigma`: `0.9 px`
- `Donut sigma`: `0`
- `Donut contrast`: `0`

Each cell receives its own gain and baseline fluorescence value. This models variability in viral expression, depth/focus, indicator concentration, and biological response amplitude.

Morphology modes:

- `Soma only`: compact round or slightly irregular somatic ROIs
- `Soma + processes`: soma ROIs with axon/dendrite-like process compartments.
- `Dendritic branches`: branch/band-like structures without soma ROIs, suitable for Purkinje dendrite-style recordings.

Process and dendrite controls include:

- `Process probability`
- `Processes / ROI`
- `Process diameter`
- `Length min`
- `Length max`
- `Orientation`
- `Angle jitter`
- `Continuity`
- `Process F0 x`
- `Process gain x`
- `Flow speed`
- `Flow bins`
- `Varicosity density`
- `Varicosity size`
- `Varicosity strength`

Process compartments can have independent baseline fluorescence and gain scaling relative to their parent soma. Soma-linked processes are drawn bidirectionally through or near the soma so axons can enter and leave the FOV rather than projecting from the soma toward only one edge. Directional activity flow is simulated by spatially binning each process and applying time delays along the process axis. Set `Flow speed` to `0` or `Flow bins` to `1` for uniform, non-propagating process activity.

When switching to `Soma + processes`, the GUI applies an MLI-like preset intended to better match the local molecular layer interneuron reference stack. It lowers the cell count and noise, uses a square field of view, brightens somas relative to axons, and produces sparser, more tangled, bead-like axonal structure:

- `N active neurons`: `45`
- `N inactive neurons`: `45`
- `Image H`: `256 px`
- `Image W`: `256 px`
- `F0 baseline`: `2.0`
- `Process probability`: `0.50`
- `Processes / ROI`: `2`
- `Process diameter`: `0.65 px`
- `Length min`: `45 px`
- `Length max`: `180 px`
- `Orientation`: `90 deg`
- `Angle jitter`: `55 deg`
- `Continuity`: `0.75`
- `Process F0 x`: `0.10`
- `Process gain x`: `0.30`
- `Flow speed`: `60 px/s`
- `Flow bins`: `8`
- `Varicosity density`: `0.018 / px`
- `Varicosity size`: `0.85 px`
- `Varicosity strength`: `3.0`
- `Neuropil blobs`: `120`
- `Neuropil sigma`: `25-80 px`
- `Neuropil noise`: `0.004`
- `DC level`: `0.045`
- `Read noise sigma`: `0.012`
- `Shot coeff`: `0.05`
- `Counts scale`: `300 ct/a.u.`

### Neuropil

Default neuropil parameters:

- `Blobs`: `10`
- `Sigma min`: `15 px`
- `Sigma max`: `35 px`
- `Oscillation amp`: `0`
- `Oscillation freq`: `0 Hz`
- `Neuropil noise`: `0.03`
- `Bleed fraction`: `0.20`
- `DC level`: `0.1`

The rendered movie is built by adding cell/process fluorescence on top of the neuropil field before applying motion and noise. Activity therefore increases local fluorescence without suppressing the surrounding neuropil or background noise.

### Motion

Motion controls include continuous jitter and occasional saccade-like displacements:

- `Jitter sigma`: `0.5 px`
- `Jitter tau`: `0.05 s`
- `Saccade size`: `2 px`
- `Saccade rate`: `0.05 Hz`
- `Saccade hold`: `0.02 s`

### Noise

Default noise and digitization parameters:

- `Read noise sigma`: `0.10`
- `Shot coeff`: `0.35`
- `ADC bit depth`: `13`
- `Counts scale`: `660 ct/a.u.`

Noise is applied after neuropil and cell fluorescence are combined. Output TIFF stacks are saved as unsigned 16-bit images.

## Viewer Tabs

### Frame Viewer

The frame viewer displays the generated movie with:

- Play/stop control.
- Frame scrollbar.
- Colormap selector.
- Auto contrast.
- ROI outline toggle.
- Outline target selector: soma + axons, soma only, or axons/processes only.
- Outline width control.
- Optional filled ROI overlay.
- Click selection for one or more cells.
- Pan/zoom support.

### dF/F Traces

The trace tab can display:

- `dFF (extracted)`
- `F raw (L2 ROI)`
- `clean C (ground truth)`
- spike rasters

Spike overlays are shown underneath the fluorescence/calcium traces with smaller amplitude so spikes remain visible without obscuring the trace shape.

Trace display options include:

- Number of cells to show.
- Selected-only display.
- Sort by cell index, selected order, modulation onset, first spike, gain, baseline F0, or peak response.
- Pan/zoom support.

### Other Views

Additional tabs show:

- Mean image and intensity histogram.
- Neuropil map.
- Motion trajectory.

## Fluorescence And Trace Computations

The simulator stores both clean ground truth and extracted traces.

For each frame, the rendered fluorescence image is approximately:

```text
F_frame = neuropil_frame + soma/process fluorescence
```

Soma and process spatial templates are normalized on a comparable local peak-brightness basis before `F0`, gain, `Process F0 x`, and `Process gain x` are applied. This keeps long or branched axons from becoming bright simply because of their template normalization, while still allowing the user to make processes brighter or dimmer with explicit parameters.

Motion, shot noise, read noise, clipping, and digitization are then applied.

Important trace arrays:

- `C`: clean ground-truth calcium activity before rendering and extraction.
- `F_cells`: ROI-extracted fluorescence using the simulator's native combined ROI footprint normalization.
- `Fraw`: raw ROI fluorescence using L2-normalized ROI spatial footprints, similar in spirit to CaImAn-style spatial component extraction.
- `F0_per_cell_extracted`: per-cell baseline estimated from `F_cells`.
- `dFF`: `(F_cells - F0_per_cell_extracted) / F0_per_cell_extracted`.

The dF/F trace is therefore an extracted signal from the rendered movie, while `C` is the clean latent calcium trace used to generate the movie.

## ROI Masks

The exported mask arrays separate soma and process compartments where possible:

- `masks`: combined ROI masks retained for compatibility.
- `combined_masks`: soma + process masks.
- `soma_masks`: soma-only masks; empty for dendritic-branch-only units.
- `process_masks`: axon/dendrite-only masks; empty for soma-only units.

This separation makes it easier to compare segmentation algorithms that recover soma-only ROIs against methods that recover larger soma + process or dendritic components.

## Exported Data

Use `Save TIFF` to export the rendered movie as an ImageJ/Fiji-readable TIFF stack.

Use `Save Data` to export analysis arrays. Supported formats include:

- `.npz`
- `.npy`
- `.mat`
- `.h5` / `.hdf5`

The export payload includes:

- `movie`
- `movie_uint16`
- `t`
- `C`
- `masks`
- `combined_masks`
- `soma_masks`
- `process_masks`
- `cell_xyr`
- `active_idx`
- `inactive_idx`
- `neuropil_base`
- `neuropil_trace`
- `motion_xy`
- `uint16_scale`
- `uint16_max`
- `F_cells`
- `Fraw`
- `dFF`
- `F0_per_cell_extracted`
- `cell_F0_per_cell`
- `cell_gain_per_cell`
- `process_F0_per_cell`
- `process_gain_per_cell`
- `spike_times`
- `rate_t`
- `rate_per_unit`
- `baseline_rate_per_unit`
- `mod_windows`

## Performance Notes

Soma-only simulations use a faster rendering path and skip process-flow computations. Soma + process and dendritic-branch modes are more expensive because they maintain additional spatial templates and, when enabled, directional flow bins.

For faster previews:

- Reduce image size or recording duration.
- Reduce frame rate.
- Reduce active/inactive neuron counts.
- Use `Soma only` morphology while tuning non-morphology parameters.
- Reduce `Flow bins`.
- Set `Flow speed` to `0` when process propagation is not needed.

## Project Structure

```text
twophoton_sim/
  main.py
  run_headless.py
  requirements.txt
  README.md
  core/
    calcium_dynamics.py
    export_data.py
    noise.py
    params.py
    spike_generator.py
    stack_simulator.py
  gui/
    main_window.py
    param_panel.py
    viz_panel.py
  examples/                 # optional local reference data; ignored by default
```

## Acknowledgements
Katrina P. Nguyen

Tool to create synthetic two-photon calcium imaging simulated tiff stacks to benchmark downstream analysis tools such as Suite2P and CaImAn.
