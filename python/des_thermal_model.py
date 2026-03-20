"""
DES Thermal Model — Python wrapper
===================================
Drives the Rust binary, then reads the HDF5 output into numpy arrays.

HDF5 layout produced by the Rust binary
----------------------------------------
/mesh/
    nodes          float64[N, 3]   — (x, y, z) per node (metres)
    elements       uint64 [E, 8]   — 8 node indices (0-based) per element

/activation/
    times          float64[E]      — time at which each element is activated (s)
    element_ids    uint64 [E]      — original element IDs from the mesh generator
    layer_nos      uint64 [E]      — layer number for each element
    orientations   float64[E, 2]   — bead orientation angles

/results/
    elem_time_temp_data     float64[M]   — flat array: [t0, T0, t1, T1, ...] for all elements
    elem_time_temp_offsets  uint64 [E+1] — CSR row pointers; element i covers
                                           data[offsets[i] : offsets[i+1]]
    node_time_temp_data     float64[P]
    node_time_temp_offsets  uint64 [N+1]

Usage
-----
    from des_thermal_model import DESSimulation, read_results

    sim = DESSimulation(
        binary_path   = r"path/to/DES_thermal_simulation.exe",
        gcode_file    = r"inputfiles/wall.gcode",
        model_type    = 2,              # 1 = isotropic-TD, 2 = orthotropic-TD
        beadwidth     = 15.875e-3,
        beadheight    = 5.08e-3,
        divs_per_bead = 1,
        divs_per_bead_z = 1,
        sp_heat_cap_file = r"inputfiles/sp_heat_cap_data.csv",
        sp_heat_cap_step = 5.0,
        # model_type 2 conductivities (m1 uses a file instead):
        kx = 0.59, ky = 0.48, kz = 0.35,
        density              = 1271.1,
        h                    = 3.0,
        temp_bed             = 363.15,
        bed_k                = 0.17,
        ambient_temp         = 313.15,
        extrusion_temp       = 473.15,
        emissivity           = 0.92,
        time_step            = 1.0,
        cooldown_period      = 100.0,
        hdf5_output          = r"outputfiles/results.h5",
        min_temp_diff_store  = 1.0,
        turn_off_layers_at   = 90,
        start_layer          = 0,       # first layer to simulate (0-based)
        end_layer            = -1,      # last layer inclusive (-1 = all)
        num_threads          = 0,       # 0 = use all physical cores
    )

    sim.run(workdir=r"C:/YVAN/CODE/DES_thermal_model")
    results = sim.results          # dict of numpy arrays (see below)

    # Or just read an existing HDF5 file directly:
    results = read_results(r"outputfiles/results.h5")
    nodes    = results["mesh/nodes"]          # float64[N, 3]
    elements = results["mesh/elements"]       # uint64 [E, 8]
    act_t    = results["activation/times"]    # float64[E]
    elem_t   = results["results/elem_temps"]  # list[ndarray] — one per element
    node_t   = results["results/node_temps"]  # list[ndarray] — one per node
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Public helper — read an existing HDF5 result file
# ---------------------------------------------------------------------------

def read_results(hdf5_path: str | Path) -> Dict[str, object]:
    """
    Read the HDF5 file produced by the Rust binary and return a dict of
    numpy arrays.

    Keys
    ----
    "mesh/nodes"             float64[N, 3]
    "mesh/elements"          uint64 [E, 8]
    "activation/times"       float64[E]
    "activation/element_ids" uint64 [E]
    "activation/layer_nos"   uint64 [E]
    "activation/orientations"float64[E, 2]
    "results/elem_temps"     list of float64 arrays, shape[k, 2] each
                             — columns are (time, temperature)
    "results/node_temps"     list of float64 arrays, shape[k, 2] each
    """
    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    out: Dict[str, object] = {}

    with h5py.File(path, "r") as f:
        # mesh
        out["mesh/nodes"]    = f["mesh/nodes"][:]
        out["mesh/elements"] = f["mesh/elements"][:]

        # activation
        out["activation/times"]        = f["activation/times"][:]
        out["activation/element_ids"]  = f["activation/element_ids"][:]
        out["activation/layer_nos"]    = f["activation/layer_nos"][:]
        out["activation/orientations"] = f["activation/orientations"][:]

        # elemental temperatures — expand CSR → list of (time, temp) arrays
        elem_data    = f["results/elem_time_temp_data"][:]
        elem_offsets = f["results/elem_time_temp_offsets"][:]
        n_elems = len(elem_offsets) - 1
        elem_temps: List[np.ndarray] = []
        for i in range(n_elems):
            chunk = elem_data[elem_offsets[i] : elem_offsets[i + 1]]
            # reshape [t0, T0, t1, T1, ...] → [[t0, T0], [t1, T1], ...]
            if len(chunk) >= 2:
                arr = chunk.reshape(-1, 2)
            else:
                arr = np.empty((0, 2), dtype=np.float64)
            elem_temps.append(arr)
        out["results/elem_temps"] = elem_temps

        # nodal temperatures — same CSR expansion
        node_data    = f["results/node_time_temp_data"][:]
        node_offsets = f["results/node_time_temp_offsets"][:]
        n_nodes_res = len(node_offsets) - 1
        node_temps: List[np.ndarray] = []
        for i in range(n_nodes_res):
            chunk = node_data[node_offsets[i] : node_offsets[i + 1]]
            if len(chunk) >= 2:
                arr = chunk.reshape(-1, 2)
            else:
                arr = np.empty((0, 2), dtype=np.float64)
            node_temps.append(arr)
        out["results/node_temps"] = node_temps

    return out


# ---------------------------------------------------------------------------
# DESSimulation class
# ---------------------------------------------------------------------------

class DESSimulation:
    """
    High-level Python interface to the DES thermal simulation Rust binary.

    Parameters shared by all model types
    -------------------------------------
    binary_path         Path to the compiled Rust executable.
    gcode_file          G-code or toolpath file (relative to workdir).
    model_type          1 = isotropic TD conductivity,
                        2 = orthotropic TD conductivity (variable h).
    beadwidth           Bead width in metres.
    beadheight          Bead height in metres.
    divs_per_bead       Finite-element divisions per bead width.
    divs_per_bead_z     Finite-element divisions per bead height.
    sp_heat_cap_file    CSV file with temperature-vs-specific-heat-capacity.
    sp_heat_cap_step    Temperature spacing in that file (°C or K).
    density             Material density (kg/m³).
    h                   Convective heat transfer film coefficient (W/m²K).
    temp_bed            Bed temperature (K).
    bed_k               Bed-surface interface conductivity (W/mK).
    ambient_temp        Ambient temperature (K).
    extrusion_temp      Extrusion temperature (K).
    emissivity          Surface emissivity (0–1).
    time_step           Simulation time step (s).
    cooldown_period     Cooldown time after last deposition (s).
    hdf5_output         Path for the HDF5 output file.
    min_temp_diff_store Only record a temperature snapshot when the element
                        temperature changes by at least this many degrees.
    turn_off_layers_at  Number of layers above which distant layers are
                        deactivated (saves computation).
    num_threads         CPU threads to use (0 = all physical cores).

    Model-type-1 extra parameters
    ------------------------------
    conductivity_file   CSV file with temperature-vs-conductivity.
    conductivity_step   Temperature spacing in that file.

    Model-type-2 extra parameters
    ------------------------------
    kx, ky, kz         Orthotropic thermal conductivities (W/mK).
    """

    def __init__(
        self,
        binary_path: str | Path,
        gcode_file: str,
        model_type: int,
        beadwidth: float,
        beadheight: float,
        divs_per_bead: int,
        divs_per_bead_z: int,
        sp_heat_cap_file: str,
        sp_heat_cap_step: float,
        density: float,
        h: float,
        temp_bed: float,
        bed_k: float,
        ambient_temp: float,
        extrusion_temp: float,
        emissivity: float,
        time_step: float,
        cooldown_period: float,
        hdf5_output: str = "outputfiles/results.h5",
        min_temp_diff_store: float = 1.0,
        turn_off_layers_at: int = 90,
        num_threads: int = 0,
        start_layer: int = 0,
        end_layer: int = -1,
        # model_type 1
        conductivity_file: Optional[str] = None,
        conductivity_step: Optional[float] = None,
        # model_type 2
        kx: Optional[float] = None,
        ky: Optional[float] = None,
        kz: Optional[float] = None,
    ):
        self.binary_path        = Path(binary_path)
        self.gcode_file         = gcode_file
        self.model_type         = model_type
        self.beadwidth          = beadwidth
        self.beadheight         = beadheight
        self.divs_per_bead      = divs_per_bead
        self.divs_per_bead_z    = divs_per_bead_z
        self.sp_heat_cap_file   = sp_heat_cap_file
        self.sp_heat_cap_step   = sp_heat_cap_step
        self.density            = density
        self.h                  = h
        self.temp_bed           = temp_bed
        self.bed_k              = bed_k
        self.ambient_temp       = ambient_temp
        self.extrusion_temp     = extrusion_temp
        self.emissivity         = emissivity
        self.time_step          = time_step
        self.cooldown_period    = cooldown_period
        self.hdf5_output        = hdf5_output
        self.min_temp_diff_store = min_temp_diff_store
        self.turn_off_layers_at  = turn_off_layers_at
        self.start_layer         = start_layer
        self.end_layer           = end_layer
        self.num_threads         = num_threads
        self.conductivity_file  = conductivity_file
        self.conductivity_step  = conductivity_step
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self._results: Optional[Dict[str, object]] = None

    # ------------------------------------------------------------------
    # Input file generation
    # ------------------------------------------------------------------

    def _build_input_file(self) -> str:
        """Return the content of the Input_file.txt for the Rust binary."""
        lines = [
            "----Input File----",
            f"Number_of_threads: {self.num_threads}",
            f"GCode_file: {self.gcode_file}",
            f"divisions_per_bead_width: {self.divs_per_bead}",
            f"divisions_per_bead_height: {self.divs_per_bead_z}",
            f"beadwidth: {self.beadwidth}",
            f"beadheight: {self.beadheight}",
            f"model_type: {self.model_type}",
        ]

        if self.model_type == 1:
            if self.conductivity_file is None or self.conductivity_step is None:
                raise ValueError(
                    "model_type=1 requires conductivity_file and conductivity_step"
                )
            lines += [
                f"specific_heat_capacity_file: {self.sp_heat_cap_file}",
                f"specific_heat_capacity_file_temp_step: {self.sp_heat_cap_step}",
                f"conductivity_file: {self.conductivity_file}",
                f"conductivity_file_temp_step: {self.conductivity_step}",
                f"density: {self.density}",
                f"convective_heat_transfer_film_coefficient: {self.h}",
                f"bed_temperature: {self.temp_bed}",
                f"conductivity_at_bed_surface_interface: {self.bed_k}",
                f"ambient_temperature: {self.ambient_temp}",
                f"extrusion_temperature: {self.extrusion_temp}",
                f"emissivity: {self.emissivity}",
                f"time_step: {self.time_step}",
                f"cooldown_period: {self.cooldown_period}",
                f"hdf5_output: {self.hdf5_output}",
                f"element_min_temp_diff_store: {self.min_temp_diff_store}",
                f"turn_off_layers_at: {self.turn_off_layers_at}",
                f"start_layer: {self.start_layer}",
                f"end_layer: {self.end_layer}",
            ]

        elif self.model_type == 2:
            if None in (self.kx, self.ky, self.kz):
                raise ValueError("model_type=2 requires kx, ky, and kz")
            lines += [
                f"specific_heat_capacity_file: {self.sp_heat_cap_file}",
                f"specific_heat_capacity_file_temp_step: {self.sp_heat_cap_step}",
                f"kx: {self.kx}",
                f"ky: {self.ky}",
                f"kz: {self.kz}",
                f"density: {self.density}",
                f"convective_heat_transfer_film_coefficient: {self.h}",
                f"bed_temperature: {self.temp_bed}",
                f"conductivity_at_bed_surface_interface: {self.bed_k}",
                f"ambient_temperature: {self.ambient_temp}",
                f"extrusion_temperature: {self.extrusion_temp}",
                f"emissivity: {self.emissivity}",
                f"time_step: {self.time_step}",
                f"cooldown_period: {self.cooldown_period}",
                f"hdf5_output: {self.hdf5_output}",
                f"element_min_temp_diff_store: {self.min_temp_diff_store}",
                f"turn_off_layers_at: {self.turn_off_layers_at}",
                f"start_layer: {self.start_layer}",
                f"end_layer: {self.end_layer}",
            ]
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        workdir: str | Path,
        input_filename: str = "inputfiles/Input_file.txt",
        capture_output: bool = False,
        timeout: Optional[float] = None,
    ) -> "DESSimulation":
        """
        Write the input file, run the Rust binary, and load results.

        Parameters
        ----------
        workdir         Working directory passed to the subprocess (must
                        contain the inputfiles/ and outputfiles/ folders).
        input_filename  Path for the generated input file, relative to workdir.
        capture_output  If True, suppress binary stdout/stderr; otherwise
                        the binary prints to the terminal.
        timeout         Optional timeout in seconds for the subprocess.

        Returns
        -------
        self — so you can chain: sim.run(...).results
        """
        workdir = Path(workdir)
        if not workdir.is_dir():
            raise NotADirectoryError(f"workdir does not exist: {workdir}")

        # Write input file
        input_path = workdir / input_filename
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(self._build_input_file(), encoding="utf-8")

        # Ensure output directory exists
        hdf5_abs = workdir / self.hdf5_output
        hdf5_abs.parent.mkdir(parents=True, exist_ok=True)

        # Run the binary
        cmd = [str(self.binary_path)]
        kwargs = dict(cwd=str(workdir), timeout=timeout)
        if capture_output:
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.PIPE

        proc = subprocess.run(cmd, **kwargs)
        if proc.returncode != 0:
            err = proc.stderr.decode() if capture_output else "(see terminal)"
            raise RuntimeError(
                f"Rust binary exited with code {proc.returncode}.\n{err}"
            )

        # Read results
        self._results = read_results(hdf5_abs)
        return self

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    @property
    def results(self) -> Dict[str, object]:
        """
        Simulation results as a dict of numpy arrays.  Call run() first.

        Keys
        ----
        "mesh/nodes"             float64[N, 3]   — (x, y, z) in metres
        "mesh/elements"          uint64 [E, 8]   — 8 node indices (0-based)
        "activation/times"       float64[E]      — activation times (s)
        "activation/element_ids" uint64 [E]
        "activation/layer_nos"   uint64 [E]
        "activation/orientations"float64[E, 2]
        "results/elem_temps"     list[ndarray]   — shape [k, 2]: (time, temp) per element
        "results/node_temps"     list[ndarray]   — shape [k, 2]: (time, temp) per node
        """
        if self._results is None:
            raise RuntimeError("No results available — call run() first.")
        return self._results

    # Convenience accessors
    @property
    def nodes(self) -> np.ndarray:
        """Node coordinates, float64[N, 3]."""
        return self.results["mesh/nodes"]

    @property
    def elements(self) -> np.ndarray:
        """Element connectivity, uint64[E, 8]."""
        return self.results["mesh/elements"]

    @property
    def activation_times(self) -> np.ndarray:
        """Activation time per element, float64[E]."""
        return self.results["activation/times"]

    @property
    def elem_temps(self) -> List[np.ndarray]:
        """
        Temperature history per element as a list of float64[k, 2] arrays
        where column 0 is time (s) and column 1 is temperature (K).
        """
        return self.results["results/elem_temps"]

    @property
    def node_temps(self) -> List[np.ndarray]:
        """Temperature history per node, same layout as elem_temps."""
        return self.results["results/node_temps"]

    def __repr__(self) -> str:
        ran = self._results is not None
        return (
            f"DESSimulation(model_type={self.model_type}, "
            f"gcode='{self.gcode_file}', results_loaded={ran})"
        )


# ---------------------------------------------------------------------------
# Quick post-processing helpers
# ---------------------------------------------------------------------------

def max_temperature_per_element(sim_or_results) -> np.ndarray:
    """
    Return the peak temperature reached by each element (K).

    Parameters
    ----------
    sim_or_results   A DESSimulation instance, or the dict returned by
                     read_results().
    """
    if isinstance(sim_or_results, DESSimulation):
        elem_temps = sim_or_results.elem_temps
    else:
        elem_temps = sim_or_results["results/elem_temps"]

    peaks = np.empty(len(elem_temps), dtype=np.float64)
    peaks[:] = np.nan
    for i, arr in enumerate(elem_temps):
        if arr.size > 0:
            peaks[i] = arr[:, 1].max()
    return peaks


def cooling_rate_at_time(sim_or_results, query_time: float) -> np.ndarray:
    """
    Estimate the instantaneous cooling rate (dT/dt, K/s) for each element
    at the stored snapshot nearest to *query_time*.

    Returns NaN for elements with fewer than 2 stored snapshots or that were
    not yet activated at *query_time*.
    """
    if isinstance(sim_or_results, DESSimulation):
        elem_temps = sim_or_results.elem_temps
    else:
        elem_temps = sim_or_results["results/elem_temps"]

    rates = np.full(len(elem_temps), np.nan, dtype=np.float64)
    for i, arr in enumerate(elem_temps):
        if arr.shape[0] < 2:
            continue
        times = arr[:, 0]
        idx = np.searchsorted(times, query_time)
        if idx == 0 or idx >= len(times):
            continue
        dt = times[idx] - times[idx - 1]
        dT = arr[idx, 1] - arr[idx - 1, 1]
        if abs(dt) > 1e-12:
            rates[i] = dT / dt
    return rates
