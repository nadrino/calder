import torch
import uproot
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from calder.core.samples.dataset import EventTable
from calder.core.samples.histogram import Histogram
from calder.core.globals.device import set_device
import calder.utils.style


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
    files: list[str],
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

    return uproot.concatenate(
        [f"{f}:{tree_name}" for f in files],
        cut=selection,
        filter_name=branches,
        library="np",
    )

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


def plot_hist2d(hist2d, xedges, yedges, ax=None, title=None, xlabel=None, ylabel=None, cmap="magma", **kwargs):
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

    files = [
        "/Users/nadrino/Documents/Work/Output/results/gundam/common/OA2024/ND280/Inputs/Splines/XSecAndNDSyst/P7/v12_Highland_3.22.4/MC_mirrored/run4wMCsplines.root",
        "/Users/nadrino/Documents/Work/Output/results/gundam/common/OA2024/ND280/Inputs/Splines/XSecAndNDSyst/P7/v12_Highland_3.22.4/MC_mirrored/run5MCsplines.root",
    ]

    t0 = time.perf_counter()
    arrays = load_root_events(
        files,
        "sample_sum",
        branches,
        # selection="SelectedSample == 157",
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

    Pmu_edges = torch.linspace(100, 5000, 51)
    CosThetamu_edges = torch.linspace(0, 1, 51)

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

    import torch
    import math
    import matplotlib.pyplot as plt


    # ============================================================
    # 1. Fonctions KDE et utilitaires depuis ton dictionnaire
    # ============================================================

    def stack_from_dict(x_data_dict, x_mc_dict):
        varnames = list(x_mc_dict.keys())
        x_mc_list = []
        x_data_list = []
        h_vec = []
        mu_vec = []
        std_vec = []
        for v in varnames:
            x_mc_list.append(x_mc_dict[v]["data"])
            x_data_list.append(x_data_dict[v])
            h_vec.append(x_mc_dict[v]["h"])
            mu_vec.append(x_mc_dict[v]["mean"])
            std_vec.append(x_mc_dict[v]["std"])
        x_mc = torch.stack(x_mc_list, dim=1)
        x_data = torch.stack(x_data_list, dim=1)
        h_vec = torch.tensor(h_vec, dtype=x_mc.dtype, device=x_mc.device)
        mu_vec = torch.tensor(mu_vec, dtype=x_mc.dtype, device=x_mc.device)
        std_vec = torch.tensor(std_vec, dtype=x_mc.dtype, device=x_mc.device)
        return x_data, x_mc, h_vec, mu_vec, std_vec


    import torch, math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm


    def robust_bandwidths(x_mc_dict, fp=0.05, fc=0.08):
        p = x_mc_dict["Pmu"]["data"].flatten()
        c = x_mc_dict["CosThetamu"]["data"].flatten()
        p05, p95 = torch.quantile(p, 0.05), torch.quantile(p, 0.95)
        c05, c95 = torch.quantile(c, 0.05), torch.quantile(c, 0.95)
        h_p = fp * (p95 - p05)  # e.g. 5% of central 90% range
        h_c = fc * (c95 - c05)  # e.g. 8% of central 90% range
        return torch.stack([h_p, h_c])


    import math
    import torch


    def logpdf_kde_batched_hist_ref(
          x_data_dict,  # dict {"Pmu": tensor(N,), "CosThetamu": tensor(N,)}
          x_mc_dict,  # dict {"Pmu": {"data": tensor(M,)}, "CosThetamu": {"data": tensor(M,)}}
          w_mc,  # tensor (M,)
          hist_ref,
          # dict with {"edges":[edges_Pmu, edges_CosThetamu], "counts":..., optional "h_vec":..., "alpha":...}
          batch_size=1200,
    ):
        """
        Returns log PDF values at x_data points using a KDE corrected by the
        reference histogram (hist_ref).
        """

        # --------------------------------------------------
        # Stack variables in fixed order
        # --------------------------------------------------
        vars_order = ["Pmu", "CosThetamu"]
        x_data = torch.stack([x_data_dict[v].flatten() for v in vars_order], dim=1)
        x_mc = torch.stack([x_mc_dict[v]["data"].flatten() for v in vars_order], dim=1)

        device = x_mc.device
        dtype = x_mc.dtype
        eps = torch.tensor(1e-12, device=device, dtype=dtype)

        N, d = x_data.shape
        M = x_mc.shape[0]

        assert d == len(vars_order)
        assert w_mc.shape[0] == M

        # --------------------------------------------------
        # Bandwidths h_vec (if not provided)
        # --------------------------------------------------
        h_vec = hist_ref.get("h_vec", None)
        if h_vec is None:
            W = w_mc.sum().clamp_min(eps)
            mean = (w_mc[:, None] * x_mc).sum(0) / W
            var = (w_mc[:, None] * (x_mc - mean) ** 2).sum(0) / W
            std = torch.sqrt(var.clamp_min(1e-24))
            n_eff = (W * W) / (w_mc.pow(2).sum().clamp_min(eps))
            h_vec = std * torch.pow(n_eff, -1.0 / (d + 4.0))
        h_vec = h_vec.to(device=device, dtype=dtype).clamp_min(torch.finfo(dtype).eps)

        # --------------------------------------------------
        # Core batched KDE log-density
        # --------------------------------------------------
        log_den = torch.log(w_mc.sum().clamp_min(eps))
        log_norm = -0.5 * d * math.log(2.0 * math.pi) - torch.log(h_vec).sum()

        out_chunks = []
        for i in range(0, N, batch_size):
            xb = x_data[i:i + batch_size]  # (B, d)
            diff = xb[:, None, :] - x_mc[None, :, :]  # (B, M, d)
            quad = (diff * diff / (h_vec * h_vec)).sum(-1)  # (B, M)
            logk = log_norm - 0.5 * quad
            log_num = torch.logsumexp(torch.log(w_mc.clamp_min(eps)) + logk, dim=1)
            out_chunks.append(log_num - log_den)
        logpdf_kde = torch.cat(out_chunks, dim=0)

        # --------------------------------------------------
        # Histogram-based multiplicative correction
        # --------------------------------------------------
        edges = [e.to(device=device, dtype=dtype) for e in hist_ref["edges"]]
        y_grid = hist_ref["counts"].to(device=device, dtype=dtype)
        y_tot = y_grid.sum().clamp_min(eps)
        y_frac = y_grid / y_tot

        sqrt2 = math.sqrt(2.0)
        mass_per_dim = []
        for k in range(d):
            ek = edges[k].contiguous()  # (B_k+1,)
            z = (ek[None, :] - x_mc[:, k:k + 1]) / h_vec[k]
            cdf = 0.5 * (1.0 + torch.erf(z / sqrt2))
            mk = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
            mass_per_dim.append(mk)

        letters = "abcdefghijklmnopqrstuvwxyz"
        in_spec = ["m"] + [f"m{letters[i]}" for i in range(d)]
        out_spec = "".join([letters[i] for i in range(d)])
        eq = ",".join(in_spec) + "->" + out_spec
        tensors = [w_mc] + mass_per_dim
        mu_grid_w = torch.einsum(eq, *tensors)
        W = w_mc.sum().clamp_min(eps)
        mu_grid = (mu_grid_w / W).contiguous()

        alpha = float(hist_ref.get("alpha", 1e-12))
        r_grid = (y_frac + alpha) / (mu_grid + alpha)
        Z = (r_grid * mu_grid).sum().clamp_min(eps)

        # --------------------------------------------------
        # Evaluate r(x) = r(bin) for each data point
        # --------------------------------------------------
        idxs = []
        for k in tqdm(range(d)):
            ek = edges[k]
            idx = torch.bucketize(x_data[:, k], ek, right=False) - 1
            idx = idx.clamp(min=0, max=ek.numel() - 2)
            idxs.append(idx)

        if d == 2:
            r_vals = r_grid[idxs[0], idxs[1]]
        else:
            r_vals = r_grid[tuple(idxs)]

        log_correction = torch.log(r_vals.clamp_min(eps)) - torch.log(Z)
        return logpdf_kde + log_correction


    def logpdf_kde_batched_raw(x_data_dict, x_mc_dict, w_mc, h_vec, batch_size=1500):
        # stack without standardization
        def stack_from_dict(x_data_dict, x_mc_dict):
            vars_order = ["Pmu", "CosThetamu"]
            x_mc = torch.stack([x_mc_dict[v]["data"].flatten() for v in vars_order], dim=1)
            x_dt = torch.stack([x_data_dict[v].flatten() for v in vars_order], dim=1)
            return x_dt, x_mc

        x_data, x_mc = stack_from_dict(x_data_dict, x_mc_dict)
        eps = 1e-12
        N, d = x_data.shape
        log_den = torch.log(w_mc.sum() + eps)
        log_norm = -0.5 * d * math.log(2 * math.pi) - torch.log(h_vec).sum()

        out = []
        for i in tqdm(range(0, N, batch_size)):
            xb = x_data[i:i + batch_size]  # (B, d)
            diff = xb[:, None, :] - x_mc[None, :, :]  # (B, M, d)
            quad = (diff ** 2 / (h_vec * h_vec)).sum(-1)  # (B, M)
            logk = log_norm - 0.5 * quad
            log_num = torch.logsumexp(torch.log(w_mc + eps) + logk, dim=1)
            out.append(log_num - log_den)
        return torch.cat(out, dim=0)


    # # Example usage with your grid
    # # Choose balanced bandwidths
    # h_vec = robust_bandwidths(x_mc)  # tensor([h_p, h_c]) on the same device/dtype
    # log_f = logpdf_kde_batched_raw(x_data, x_mc, w_mc, h_vec, batch_size=1200)
    # f = torch.exp(log_f).reshape(P.shape)

    M = events["Pmu"].shape[0]
    x_mc = {
        "Pmu":
            {"data": events["Pmu"], "mean":  events["Pmu"].mean(), "std":  events["Pmu"].std(), "h":  events["Pmu"].std() * M ** (-1 / 6)},
        "CosThetamu":
            {"data": events["CosThetamu"], "mean": events["CosThetamu"].mean(), "std": events["CosThetamu"].std(), "h": events["CosThetamu"].std() * M ** (-1 / 6)}
    }
    w_mc = torch.ones(M, device=events["Pmu"].device)

    p_grid = torch.linspace(100, 5000, 201, device=events["Pmu"].device)
    ct_grid = torch.linspace(0.0, 1.0, 201, device=events["Pmu"].device)
    P, C = torch.meshgrid(p_grid, ct_grid, indexing="xy")

    # Dictionnaire x_data pour évaluer la PDF sur la grille
    x_data = {
        "Pmu": P.flatten(),
        "CosThetamu": C.flatten()
    }

    # print(x_mc["Pmu"]["data"])
    x_mc["Pmu"]["data"] = x_mc["Pmu"]["data"].flatten()
    x_mc["CosThetamu"]["data"] = x_mc["CosThetamu"]["data"].flatten()
    x_mc_check = torch.stack([x_mc["Pmu"]["data"], x_mc["CosThetamu"]["data"]], dim=1)
    x_data_check = torch.stack([x_data["Pmu"], x_data["CosThetamu"]], dim=1)
    print("x_mc:", x_mc_check.shape, "x_data:", x_data_check.shape)

    h_vec = robust_bandwidths(x_mc)  # tensor([h_p, h_c]) on the same device/dtype
    log_f = logpdf_kde_batched_raw(x_data, x_mc, w_mc, h_vec, batch_size=1200)
    # f = torch.exp(log_f).reshape(P.shape)

    # log_f = logpdf_kde_from_dict_batched(x_data, x_mc, w_mc)
    # f = torch.exp(log_f).reshape(P.shape)

    # plt.figure(figsize=(7, 5), dpi=130)
    # plt.pcolormesh(P.to("cpu"), C.to("cpu"), f.to("cpu"), shading="auto", cmap="magma")
    # # plt.scatter(p_mu, cos_theta, s=6, c="white", edgecolors="k", lw=0.2, label="MC events")
    # plt.xlabel(r"$p_\mu$ [MeV]")
    # plt.ylabel(r"$\cos\theta_\mu$")
    # plt.title("PDF 2D KDE (pondérée)")
    # plt.colorbar(label="f(p_mu, cos_theta_mu)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Optionally, define per-dim bandwidths (for example, pre-determined)
    h_vec = torch.tensor([200.0, 0.02])  # in same units as your observables

    # Put everything into a dict
    hist_ref = {
        "edges": [Pmu_edges, CosThetamu_edges],
        "counts": hist.hist,
        "h_vec": h_vec,  # optional; can be omitted if you want auto bandwidth
        "alpha": 1e-6  # small pseudo-count to stabilize ratios
    }

    print("Calculating PDF")
    logpdf = logpdf_kde_batched_hist_ref(
        x_data,  # (N, 2)
        x_mc,  # (M, 2)
        w_mc,  # (M,)
        hist_ref,
        batch_size=1200,
    )

    print("EXP")
    f = torch.exp(logpdf).reshape(P.shape)

    print("to cpu...")
    P = P.cpu()
    C = C.cpu()
    f = f.cpu()

    f *= torch_arrays["Pmu"].shape[0]

    print("plotting...")
    # Plot in log scale so structure is visible
    plt.figure(figsize=(6, 5), dpi=150)
    plt.pcolormesh(P, C, np.clip(f.numpy(), 1e-12, None),
                   shading="auto", cmap="magma",
                   norm=LogNorm(
                       vmin=1,
                       # vmax=float(f.max())
                   )
                   )
    # plt.scatter(x_mc["Pmu"]["data"].cpu(), x_mc["CosThetamu"]["data"].cpu(),
    #             s=4, c="white", edgecolors="k", lw=0.2, alpha=0.6)
    plt.xlabel("p_mu [MeV]")
    plt.ylabel("cos_theta_mu")
    plt.title("PDF 2D KDE (weighted, per-dim bandwidths)")
    plt.colorbar(label="f(p_mu, cos_theta_mu)")
    plt.tight_layout()
    plt.show()


