from calder.core.samples.event import EventTable

import uproot


def load_root_events(
    file_path: str,
    tree_name: str,
    branches: list[str],
    selection: str | None = None,
    device: str = "cpu"
) -> EventTable:
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
        import numpy as np
        tree = f[tree_name]
        arrays = tree.arrays(branches, cut=selection, library="np")
        print(f"Loaded {len(arrays["Pmu"])} events from {file_path}")
        return EventTable(arrays, device=device)


if __name__ == "__main__":
    branches = [
        "Pmu",
        "CosThetamu",
        "ReactionCode"
    ]

    path = "/Users/nadrino/Documents/Work/Output/results/gundam/common/OA2024/ND280/Inputs/Splines/XSecAndNDSyst/P7/v12_Highland_3.22.4/MC_mirrored/run4wMCsplines.root"

    events = load_root_events(
        path,
        "sample_sum",
        branches,
        # selection="SelectedSample == 157",
        device="cpu"
    )

    print(events["Pmu"][:5])
