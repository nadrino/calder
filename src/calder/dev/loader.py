import torch
import uproot
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from calder.core.samples.event import EventTable
from calder.core.samples.histogram import Histogram
from calder.core.globals.device import set_device


def uproot_to_tensors(arrays) -> dict[str, torch.Tensor]:
    """Convert uproot arrays to a dict of CPU torch tensors, memory efficiently."""
    tensors = {}

    for name, arr in arrays.items():
        # Skip non-arrays
        if not isinstance(arr, np.ndarray):
            print(f"[uproot_to_tensors] Skipping '{name}' (not ndarray)")
            continue

        # Case 1: simple numeric branch
        if np.issubdtype(arr.dtype, np.number):
            tensors[name] = torch.from_numpy(arr)
            continue

        # Case 2: object array (likely vector<T>)
        if arr.dtype == np.object_:
            # Try to see if all entries have same length
            first = arr[0]
            if not isinstance(first, np.ndarray):
                print(f"[uproot_to_tensors] Skipping '{name}' (non-numeric object)")
                continue
            n = len(first)
            if all(isinstance(x, np.ndarray) and len(x) == n for x in arr):
                # Regular array-of-arrays → stack view
                try:
                    stacked = np.stack(arr)
                    tensors[name] = torch.from_numpy(stacked)
                except Exception as e:
                    print(f"[uproot_to_tensors] Failed to stack '{name}': {e}")
            else:
                print(f"[uproot_to_tensors] Skipping '{name}' (jagged vector)")
            continue

        print(f"[uproot_to_tensors] Skipping '{name}' (unsupported dtype {arr.dtype})")

    return tensors


def load_root_events(
    file_path: str,
    tree_name: str,
    branches: list[str],
    selection: str | None = None,
):
    """
    Loads branches from a ROOT TTree and applies selection cuts with ROOT-style syntax.

    Example:
        events = load_root_events(
            "file.root", "tree",
            ["Enu", "theta", "weight"],
            selection="Enu > 0.5 && theta < 2.5",
            device="cuda"
        )
    """
    with uproot.open(file_path) as f:
        tree = f[tree_name]
        arrays = tree.arrays(branches, cut=selection, library="np")
        print(f"Loaded {len(arrays['Pmu'])} events from {file_path}")
        return arrays


def plot_flat_histogram(hist, ax=None, title=None, xlabel="Bin index", ylabel="Entries", **kwargs):
    """Plot a flattened (1D) view of an nD histogram.
    Useful for debugging binning or visualizing generic histograms."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    # Move data to CPU and flatten
    if isinstance(hist, torch.Tensor):
        y = hist.detach().flatten().cpu().numpy()
    elif isinstance(hist, dict) and "hist" in hist:
        y = hist["hist"].detach().flatten().cpu().numpy()
    else:
        y = hist

    x = range(len(y))
    ax.step(x, y, where="mid", **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_hist2d(hist2d, xedges, yedges, ax=None, title=None, xlabel=None, ylabel=None, cmap="viridis", **kwargs):
    """Plot a 2D histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    H = hist2d.detach().cpu().numpy() if isinstance(hist2d, torch.Tensor) else hist2d
    X = xedges.detach().cpu().numpy()
    Y = yedges.detach().cpu().numpy()
    

    mesh = ax.pcolormesh(X, Y, H.T, cmap=cmap, shading="auto", norm=LogNorm(), **kwargs)
    plt.colorbar(mesh, ax=ax, label="Entries")
    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax


if __name__ == "__main__":

    # set_device("cpu")
    set_device("mps")

    branches = [
        "Pmu",
        "CosThetamu",
        "ReactionCode"
    ]

    path = "/Users/nadrino/Documents/Work/Output/results/gundam/common/OA2024/ND280/Inputs/Splines/XSecAndNDSyst/P7/v12_Highland_3.22.4/MC_mirrored/run4wMCsplines.root"

    t0 = time.perf_counter()
    arrays = load_root_events(
        path,
        "sample_sum",
        branches,
        selection="SelectedSample == 157",
    )
    t1 = time.perf_counter()
    print(f"Load from disk: {t1 - t0:.3f} s")

    torch_arrays = uproot_to_tensors(arrays)
    print(torch_arrays["Pmu"].shape[0])
    torch_arrays["weight"] = torch.ones(torch_arrays["Pmu"].shape[0])

    t0 = time.perf_counter()
    events = EventTable(torch_arrays)
    t1 = time.perf_counter()
    print(f"To device: {t1 - t0:.3f} s")

    Pmu_edges = torch.linspace(100, 5000, 101)
    CosThetamu_edges = torch.linspace(0, 1, 101)

    # Create and fill 2D histogram
    hist = Histogram(["Pmu", "CosThetamu"], [Pmu_edges, CosThetamu_edges])

    t0 = time.perf_counter()
    hist.fill(events)
    t1 = time.perf_counter()
    print(f"Binning: {t1 - t0:.3f} s")

    print(hist.hist.shape)  # (50, 30)
    print(hist.hist.device)  # mps:0

    # plot_flat_histogram(hist.hist, title="hist", xlabel="bin index", ylabel="counts")

    plot_hist2d(hist.hist, Pmu_edges, CosThetamu_edges, title="2D hist", xlabel="Pmu [GeV]", ylabel="CosThetamu")
    plt.show()
