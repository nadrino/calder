import torch
import uproot
import numpy as np
from calder.core.samples.event import EventTable
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


if __name__ == "__main__":
    branches = [
        "Pmu",
        "CosThetamu",
        "ReactionCode"
    ]

    path = "/Users/nadrino/Documents/Work/Output/results/gundam/common/OA2024/ND280/Inputs/Splines/XSecAndNDSyst/P7/v12_Highland_3.22.4/MC_mirrored/run4wMCsplines.root"

    arrays = load_root_events(
        path,
        "sample_sum",
        branches,
        selection="SelectedSample == 157",
    )

    print(arrays["Pmu"][:5])

    set_device("mps")
    events = EventTable(uproot_to_tensors(arrays))
    print(events.data)


