"""
test_visualization.py
=====================
1. Initialise simulation parameters and run the Rust DES binary.
2. Parse the HDF5 results file.
3. Display the print path as PyVista cylinders (tubes) coloured by peak
   temperature, with a heat-map legend.

Run from the repo root:
    python test_visualization.py

Set RUN_SIM = False to skip the simulation and re-visualise existing results.

Dependencies
------------
    conda install -c conda-forge pyvista h5py numpy
    # build the Rust binary first:
    $env:HDF5_DIR = "C:/Users/yblanchard/miniconda3/Library"
    cargo build --release
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# Make the python/ sub-package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from python.des_thermal_model import DESSimulation, read_results

# =============================================================================
# Configuration — edit here
# =============================================================================

WORKDIR     = Path(__file__).parent          # repo root
BINARY      = WORKDIR / "target/release/DES_thermal_simulation.exe"
HDF5_OUTPUT = "outputfiles/results.h5"

# Set False to skip re-running and jump straight to visualisation
RUN_SIM = True

# All simulation parameters (mirrors inputfiles/Input_file.txt)
SIM_PARAMS = dict(
    binary_path          = BINARY,
    gcode_file           = "inputfiles/3DBenchy.gcode",
    model_type           = 2,           # 2 = orthotropic, temperature-dependent SHC
    beadwidth            = 1.0e-3,   # metres
    beadheight           = 1.0e-3,     # metres
    divs_per_bead        = 1,
    divs_per_bead_z      = 1,
    sp_heat_cap_file     = "inputfiles/sp_heat_cap_data.csv",
    sp_heat_cap_step     = 5.0,         # K  spacing in the SHC file
    kx                   = 0.59,        # W/m·K  — print direction
    ky                   = 0.48,        # W/m·K  — transverse
    kz                   = 0.35,        # W/m·K  — build direction
    density              = 1271.1,      # kg/m³
    h                    = 3.0,         # W/m²·K  convective film coefficient
    temp_bed             = 363.15,      # K
    bed_k                = 0.17,        # W/m·K  bed interface conductivity
    ambient_temp         = 313.15,      # K
    extrusion_temp       = 473.15,      # K
    emissivity           = 0.92,
    time_step            = 1.0,         # s
    cooldown_period      = 100.0,       # s
    hdf5_output          = HDF5_OUTPUT,
    min_temp_diff_store  = 1.0,         # K  — snapshot threshold
    turn_off_layers_at   = 90,
    start_layer          = 1,           # first layer to simulate (0-based)
    end_layer            = 1,           # last layer inclusive (-1 = all)
    num_threads          = 0,           # 0 = all physical cores
)

# Visualisation settings
TUBE_SIDES      = 12     # cylinder facets (higher → smoother, slower)
CMAP            = "hot"  # matplotlib colormap: "hot", "plasma", "inferno", …
OPACITY_HEX     = 0.10   # transparency of the background hex mesh (0 = off)
BG_COLOR        = "white"

# Element to plot in the 2-D temperature-vs-time chart (None = hottest element)
PLOT_ELEMENT_ID: int | None = None

# =============================================================================
# 1.  Run simulation
# =============================================================================

if RUN_SIM:
    print("=" * 60)
    print("Starting DES Thermal Simulation …")
    sim = DESSimulation(**SIM_PARAMS)
    sim.run(workdir=WORKDIR)
    print("Simulation complete.")
else:
    print("RUN_SIM=False — loading existing HDF5 output.")

# =============================================================================
# 2.  Load and inspect HDF5 results
# =============================================================================

hdf5_path = WORKDIR / HDF5_OUTPUT
print(f"\nLoading results: {hdf5_path}")
results = read_results(hdf5_path)

nodes      = results["mesh/nodes"]            # float64[N, 3]  (x, y, z) metres
elements   = results["mesh/elements"]         # uint64 [E, 8]  0-based node indices
act_times  = results["activation/times"]      # float64[E]     activation time (s)
layer_nos  = results["activation/layer_nos"]  # uint64 [E]     layer number
elem_temps = results["results/elem_temps"]    # list[ndarray]  each is [k, 2]: (t, T)

n_nodes, n_elems = nodes.shape[0], elements.shape[0]
n_layers = int(layer_nos.max()) + 1
print(f"  Nodes:    {n_nodes:,}")
print(f"  Elements: {n_elems:,}")
print(f"  Layers:   {n_layers}")

# =============================================================================
# 3.  Derived quantities
# =============================================================================

# Element centroid = mean of 8 corner nodes
elem_centers = nodes[elements].mean(axis=1)   # [E, 3]

# Peak temperature per element in °C (NaN if no snapshot stored)
peak_temps_c = np.full(n_elems, np.nan, dtype=np.float64)
for i, arr in enumerate(elem_temps):
    if arr.shape[0] > 0:
        peak_temps_c[i] = arr[:, 1].max() - 273.15

# Fill any remaining NaN with ambient temperature
nan_mask = np.isnan(peak_temps_c)
if nan_mask.any():
    peak_temps_c[nan_mask] = SIM_PARAMS["ambient_temp"] - 273.15

temp_min, temp_max = float(peak_temps_c.min()), float(peak_temps_c.max())
print(f"  Peak T range: {temp_min:.1f} – {temp_max:.1f} °C")

# Estimate bead radius from element spacing (half of mean edge length).
# Use the first 10 elements to keep it fast.
_sample = nodes[elements[:min(10, n_elems)]]          # [S, 8, 3]
_edge_x = _sample[:, :, 0].max(axis=1) - _sample[:, :, 0].min(axis=1)
_edge_y = _sample[:, :, 1].max(axis=1) - _sample[:, :, 1].min(axis=1)
bead_radius = float(np.median(np.maximum(_edge_x, _edge_y))) / 2.0
bead_radius = max(bead_radius, 1e-4)
print(f"  Tube radius:  {bead_radius * 1e3:.2f} mm")

# Travel-move gap threshold: skip cylinder if consecutive element centres
# in the same layer are more than this distance apart (mm → m conversion).
TRAVEL_GAP = 3.0 * SIM_PARAMS["beadwidth"]

# =============================================================================
# 4.  2-D temperature–time plot for a single element
# =============================================================================

def plot_element_temperature(elem_id: int, elem_temps_list, act_times_arr) -> None:
    """Plot temperature vs. time curve for *elem_id* with peak annotation."""
    arr = elem_temps_list[elem_id]       # shape [k, 2]: columns = (time, temp)

    if arr.shape[0] == 0:
        print(f"[plot] Element {elem_id} has no temperature snapshots stored.")
        return

    times   = arr[:, 0]
    temps_c = arr[:, 1] - 273.15

    peak_idx  = int(np.argmax(temps_c))
    peak_time = times[peak_idx]
    peak_temp = temps_c[peak_idx]

    # Determine cooling rate: slope from peak to the last stored snapshot
    if peak_idx < len(times) - 1:
        dt = times[-1] - peak_time
        dT = temps_c[-1] - peak_temp
        cooling_rate = dT / dt if dt > 0 else 0.0
        cooling_label = f"Cooling rate ≈ {cooling_rate:.3f} °C/s"
    else:
        cooling_rate  = None
        cooling_label = "(no post-peak data)"

    act_time = float(act_times_arr[elem_id])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(times, temps_c, color="#C0392B", linewidth=1.8, label="Temperature")

    # Vertical line at activation
    ax.axvline(act_time, color="steelblue", linestyle="--",
               linewidth=1.2, label=f"Activation  t = {act_time:.1f} s")

    # Peak marker
    ax.scatter([peak_time], [peak_temp], color="#E74C3C", s=60,
               zorder=5, label=f"Peak  {peak_temp:.1f} °C")
    ax.annotate(
        f"  {peak_temp:.1f} °C\n  t = {peak_time:.1f} s",
        xy=(peak_time, peak_temp),
        xytext=(peak_time + (times[-1] - times[0]) * 0.04, peak_temp),
        fontsize=9, color="#C0392B",
        va="center",
    )

    # Ambient reference line
    ambient_c = float(SIM_PARAMS["ambient_temp"]) - 273.15
    ax.axhline(ambient_c, color="gray", linestyle=":", linewidth=1.0,
               label=f"Ambient  {ambient_c:.0f} °C")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title(
        f"Element {elem_id} — Temperature History\n{cooling_label}",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    plt.show()


_plot_id = PLOT_ELEMENT_ID if PLOT_ELEMENT_ID is not None else int(np.argmax(peak_temps_c))
print(f"\n[2-D plot] Showing temperature history for element {_plot_id}")
plot_element_temperature(_plot_id, elem_temps, act_times)

# =============================================================================
# 5.  Build PolyData: line segments along the print path (one per bead segment)
# =============================================================================

all_pts:   list[np.ndarray] = []
all_lines: list[int]        = []
all_temps: list[float]      = []
pt_offset = 0

for layer in np.unique(layer_nos):
    idx   = np.where(layer_nos == layer)[0]
    order = np.argsort(act_times[idx])   # chronological within layer
    seq   = idx[order]                   # element indices in print order

    centers = elem_centers[seq]          # [m, 3]
    temps   = peak_temps_c[seq]          # [m]  °C
    m       = len(seq)

    if m < 2:
        continue

    # Inter-element distances to detect travel moves
    dists = np.linalg.norm(np.diff(centers, axis=0), axis=1)

    # Walk the sequence and collect segments (split at travel-move gaps)
    seg_start = 0
    for j in range(m):
        at_last = (j == m - 1)
        at_gap  = (j < m - 1) and (dists[j] > TRAVEL_GAP)

        if at_gap or at_last:
            seg_end = j + 1  # exclusive
            seg_c   = centers[seg_start:seg_end]
            seg_t   = temps[seg_start:seg_end]
            n_seg   = len(seg_c)

            if n_seg >= 2:
                all_pts.append(seg_c)
                all_temps.extend(seg_t.tolist())
                for k in range(n_seg - 1):
                    all_lines += [2, pt_offset + k, pt_offset + k + 1]
                pt_offset += n_seg

            seg_start = j + 1  # next segment begins after the gap

if not all_pts:
    raise RuntimeError("No bead segments could be constructed from the results.")

pts_arr   = np.vstack(all_pts)
temps_arr = np.asarray(all_temps, dtype=np.float64)

path_mesh = pv.PolyData()
path_mesh.points = pts_arr
path_mesh.lines  = np.asarray(all_lines, dtype=np.int64)
path_mesh.point_data["Peak Temperature (°C)"] = temps_arr

print(f"\nPath mesh: {path_mesh.n_points:,} points, "
      f"{path_mesh.n_lines:,} line segments")

# Extrude lines into cylinders
tubes = path_mesh.tube(radius=bead_radius, n_sides=TUBE_SIDES)

# =============================================================================
# 6.  Build background hex mesh (translucent spatial reference)
# =============================================================================

if OPACITY_HEX > 0:
    cell_type = np.full(n_elems, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    cells = np.hstack(
        [np.full((n_elems, 1), 8, dtype=np.int64), elements.astype(np.int64)]
    ).ravel()
    hex_mesh = pv.UnstructuredGrid(cells, cell_type, nodes)
    hex_mesh.cell_data["Peak Temperature (°C)"] = peak_temps_c

# =============================================================================
# 7.  PyVista plotter
# =============================================================================

clim  = [temp_min, temp_max]

scalar_bar_args = dict(
    title           = "Peak Temperature (°C)",
    title_font_size = 14,
    label_font_size = 11,
    n_labels        = 6,
    fmt             = "%.0f",
    shadow          = True,
    vertical        = True,
    position_x      = 0.87,
    position_y      = 0.20,
    height          = 0.55,
    width           = 0.055,
    color           = "black",
)

pl = pv.Plotter(window_size=(1600, 950), title="Thermal Simulation")
pl.set_background(BG_COLOR)

# --- background hex mesh ---
if OPACITY_HEX > 0:
    pl.add_mesh(
        hex_mesh,
        scalars         = "Peak Temperature (°C)",
        cmap            = CMAP,
        clim            = clim,
        opacity         = OPACITY_HEX,
        show_scalar_bar = False,
        lighting        = False,
    )

# --- print-path tubes (main geometry) ---
pl.add_mesh(
    tubes,
    scalars         = "Peak Temperature (°C)",
    cmap            = CMAP,
    clim            = clim,
    smooth_shading  = True,
    show_scalar_bar = True,
    scalar_bar_args = scalar_bar_args,
    ambient         = 0.2,
    diffuse         = 0.8,
    specular        = 0.3,
    specular_power  = 20,
)

# --- annotations ---
pl.add_axes(line_width=3, labels_off=False)

# Temperature annotation at hottest element
hottest = int(np.argmax(peak_temps_c))
hot_pt  = elem_centers[hottest]
pl.add_point_labels(
    [hot_pt],
    [f"T_max = {peak_temps_c[hottest]:.0f} °C"],
    point_size    = 12,
    font_size     = 11,
    text_color    = "red",
    point_color   = "red",
    shape_opacity = 0.6,
    always_visible= True,
)

pl.add_title(
    "Thermal Simulation — Printing Path (Peak Temperature)",
    font_size=14,
    color="black",
)

# Isometric camera
pl.camera_position = "iso"
pl.camera.zoom(0.9)

print("\nRendering … close the window to exit.")
pl.show()
