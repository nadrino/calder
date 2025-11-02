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


max_hist = None
def plot_hist2d(hist2d, xedges, yedges, ax=None, title=None, xlabel=None, ylabel=None, cmap="magma", **kwargs):
    """Plot a 2D histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    H = hist2d.detach().cpu().numpy() if isinstance(hist2d, torch.Tensor) else hist2d
    X = xedges.detach().cpu().numpy()
    Y = yedges.detach().cpu().numpy()
    global max_hist
    max_hist = np.max(H.T)
    mesh = ax.pcolormesh(X, Y, H.T, cmap=cmap, shading="auto", norm=LogNorm(vmin=1, vmax=max_hist), **kwargs)
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

    import math
    import torch
    import torch.nn.functional as F

    import torch.nn.functional as F


    def _gauss_kernel_1d(sigma_bins: float, max_bins: int, device, dtype):
        # sigma_bins is in "bins". Limit it so that radius <= max_bins - 1
        if sigma_bins <= 0.0 or max_bins <= 1:
            return None
        # hard cap so that 3*sigma <= max_bins - 1
        max_sigma = max(1.0, (max_bins - 1) / 3.0)
        s = float(min(sigma_bins, max_sigma))
        if s < 1e-3:
            return None
        radius = min(int(math.ceil(3.0 * s)), max_bins - 1)
        xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (xs / s) ** 2)
        k = k / k.sum().clamp_min(torch.finfo(dtype).eps)
        return k


    def _smooth_grid_separable_2d(grid: torch.Tensor, sigma0_bins: float, sigma1_bins: float):
        # grid: (B0, B1), reflect padding but with safe kernel sizes
        B0, B1 = grid.shape
        x = grid.unsqueeze(0).unsqueeze(0)  # NCHW

        ky = _gauss_kernel_1d(sigma0_bins, B0, grid.device, grid.dtype)
        if ky is not None:
            p = min(ky.numel() // 2, B0 - 1)  # reflect needs pad < size
            if p > 0:
                x = F.pad(x, (0, 0, p, p), mode="reflect")
            x = F.conv2d(x, ky.view(1, 1, -1, 1))

        kx = _gauss_kernel_1d(sigma1_bins, B1, grid.device, grid.dtype)
        if kx is not None:
            p = min(kx.numel() // 2, B1 - 1)
            if p > 0:
                x = F.pad(x, (p, p, 0, 0), mode="reflect")
            x = F.conv2d(x, kx.view(1, 1, 1, -1))

        return x[0, 0]

    def _deposit_soft_hist2d(x0, x1, w, e0, e1, B0, B1, dtype, device):
        # Bilinear deposit into 4 neighbor cells on a rectilinear grid
        # Returns H shape (B0, B1) and sum_w (effective weight inside support)
        i1 = torch.bucketize(x0, e0)  # in [0..B0]
        j1 = torch.bucketize(x1, e1)  # in [0..B1]
        i0_raw = i1 - 1
        j0_raw = j1 - 1
        valid = (i0_raw >= 0) & (i0_raw < B0) & (j0_raw >= 0) & (j0_raw < B1)

        # clamp indices for safe gather; will zero out invalid later
        i0 = i0_raw.clamp(0, B0 - 1)
        j0 = j0_raw.clamp(0, B1 - 1)
        i1c = (i0 + 1).clamp(0, B0 - 1)
        j1c = (j0 + 1).clamp(0, B1 - 1)

        e0_l = e0[i0]
        e0_r = e0[i0 + 1]
        e1_l = e1[j0]
        e1_r = e1[j0 + 1]

        eps = torch.finfo(dtype).eps
        t0 = ((x0 - e0_l) / (e0_r - e0_l).clamp_min(eps)).clamp(0.0, 1.0)
        t1 = ((x1 - e1_l) / (e1_r - e1_l).clamp_min(eps)).clamp(0.0, 1.0)

        w_eff = w * valid.to(dtype)

        w00 = w_eff * (1.0 - t0) * (1.0 - t1)
        w10 = w_eff * (t0) * (1.0 - t1)
        w01 = w_eff * (1.0 - t0) * (t1)
        w11 = w_eff * (t0) * (t1)

        idx00 = (i0 * B1 + j0).view(-1)
        idx10 = (i1c * B1 + j0).view(-1)
        idx01 = (i0 * B1 + j1c).view(-1)
        idx11 = (i1c * B1 + j1c).view(-1)

        H = torch.zeros(B0 * B1, device=device, dtype=dtype)
        H.scatter_add_(0, idx00, w00.view(-1))
        H.scatter_add_(0, idx10, w10.view(-1))
        H.scatter_add_(0, idx01, w01.view(-1))
        H.scatter_add_(0, idx11, w11.view(-1))
        H = H.view(B0, B1)

        sum_w_eff = w_eff.sum()
        return H, sum_w_eff


    def _interp_bilinear_centers_2d(Fgrid, c0, c1, x):
        # Fgrid defined at cell centers (B0,B1), c0 shape (B0,), c1 shape (B1,)
        N = x.shape[0]
        eps = torch.finfo(Fgrid.dtype).eps
        i1 = torch.bucketize(x[:, 0], c0)  # in [0..B0]
        j1 = torch.bucketize(x[:, 1], c1)  # in [0..B1]
        i0 = (i1 - 1).clamp(0, c0.numel() - 1)
        j0 = (j1 - 1).clamp(0, c1.numel() - 1)
        i1 = i1.clamp(0, c0.numel() - 1)
        j1 = j1.clamp(0, c1.numel() - 1)

        c0_i0 = c0[i0];
        c0_i1 = c0[i1]
        c1_j0 = c1[j0];
        c1_j1 = c1[j1]

        t0 = ((x[:, 0] - c0_i0) / (c0_i1 - c0_i0).clamp_min(eps)).clamp(0.0, 1.0)
        t1 = ((x[:, 1] - c1_j0) / (c1_j1 - c1_j0).clamp_min(eps)).clamp(0.0, 1.0)

        f00 = Fgrid[i0, j0]
        f10 = Fgrid[i1, j0]
        f01 = Fgrid[i0, j1]
        f11 = Fgrid[i1, j1]

        return (1 - t0) * (1 - t1) * f00 + t0 * (1 - t1) * f10 + (1 - t0) * t1 * f01 + t0 * t1 * f11


    def logpdf_kde_batched_hist_ref(
          x_data_dict,
          x_mc_dict,
          w_mc,
          hist_ref,  # {"edges":[eP,eC], "counts": H_ref, opt "h_vec","alpha","smooth_sigma_bins","smooth_log"}
          batch_size=1200,  # kept for signature compatibility, not used anymore
    ):
        # 1) stack inputs
        vars_order = ["Pmu", "CosThetamu"]
        x_data = torch.stack([x_data_dict[v].flatten() for v in vars_order], dim=1)
        x_mc = torch.stack([x_mc_dict[v]["data"].flatten() for v in vars_order], dim=1)

        device = x_mc.device
        dtype = x_mc.dtype
        eps = torch.tensor(1e-12, device=device, dtype=dtype)

        # 2) reference grid
        e0, e1 = [t.to(device=device, dtype=dtype).contiguous() for t in hist_ref["edges"]]
        B0 = e0.numel() - 1
        B1 = e1.numel() - 1
        y_grid = hist_ref["counts"].to(device=device, dtype=dtype).contiguous()
        y_tot = y_grid.sum().clamp_min(eps)
        y_frac = y_grid / y_tot

        # per cell areas (supports non uniform bins)
        dx0 = (e0[1:] - e0[:-1]).contiguous()
        dx1 = (e1[1:] - e1[:-1]).contiguous()
        area = dx0[:, None] * dx1[None, :]

        # 3) bandwidths and kernel sizes in bin units
        h_vec = hist_ref.get("h_vec", None)
        if h_vec is None:
            W = w_mc.sum().clamp_min(eps)
            mean = (w_mc[:, None] * x_mc).sum(0) / W
            var = (w_mc[:, None] * (x_mc - mean) ** 2).sum(0) / W
            std = torch.sqrt(var.clamp_min(1e-24))
            n_eff = (W * W) / (w_mc.pow(2).sum().clamp_min(eps))
            h_vec = std * torch.pow(n_eff, -1.0 / (x_mc.shape[1] + 4.0))
        h_vec = h_vec.to(device=device, dtype=dtype)

        # map to bin sigmas using mean bin width per dim
        dx0_mean = float(dx0.mean().item())
        dx1_mean = float(dx1.mean().item())
        sigma0_bins = float((h_vec[0] / dx0_mean).item())
        sigma1_bins = float((h_vec[1] / dx1_mean).item())

        # optional extra smoothing strength of r, in bins
        smooth_sigma_bins = hist_ref.get("smooth_sigma_bins", 1.0)
        if isinstance(smooth_sigma_bins, (list, tuple)):
            r_sy, r_sx = float(smooth_sigma_bins[0]), float(smooth_sigma_bins[1])
        else:
            r_sy = r_sx = float(smooth_sigma_bins)

        # 4) deposit weighted MC to grid with soft binning
        H_counts, W_eff = _deposit_soft_hist2d(
            x_mc[:, 0], x_mc[:, 1], w_mc.to(dtype),
            e0, e1, B0, B1, dtype, device
        )
        W_eff = W_eff.clamp_min(eps)

        # 5) smooth counts by separable Gaussian in bin space
        H_smooth = _smooth_grid_separable_2d(H_counts, sigma0_bins, sigma1_bins)

        # 6) convert to pdf on grid (units 1/(x0*x1)), normalize
        f_grid = (H_smooth / W_eff) / area
        f_grid = f_grid / (f_grid * area).sum().clamp_min(eps)

        # 7) build r_grid = y_bin / mu_bin with matching "mu" from smoothed counts
        mu_grid = (H_smooth / W_eff).contiguous()  # probability per bin
        alpha = float(hist_ref.get("alpha", 1e-12))
        r_grid = (y_frac + alpha) / (mu_grid + alpha)

        # smooth r on the grid to remove blockiness
        if hist_ref.get("smooth_log", True):
            r_grid = torch.log(r_grid.clamp_min(alpha))
            r_grid = _smooth_grid_separable_2d(r_grid, r_sy, r_sx).exp()
        else:
            r_grid = _smooth_grid_separable_2d(r_grid, r_sy, r_sx)

        # renormalize corrected pdf
        f_corr_grid = (r_grid * f_grid).contiguous()
        Z = (f_corr_grid * area).sum().clamp_min(eps)
        f_corr_grid = f_corr_grid / Z

        # 8) interpolate pdf at data coordinates
        c0 = 0.5 * (e0[:-1] + e0[1:])
        c1 = 0.5 * (e1[:-1] + e1[1:])
        f_vals = _interp_bilinear_centers_2d(f_corr_grid, c0, c1, x_data)

        # 9) return log pdf
        return torch.log(f_vals.clamp_min(float(alpha))).to(x_data.dtype)


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
    # log_f = logpdf_kde_batched_raw(x_data, x_mc, w_mc, h_vec, batch_size=1200)
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

    # Put everything into a dict
    hist_ref = {
        "edges": [Pmu_edges, CosThetamu_edges],
        "counts": hist.hist,
        # "h_vec": torch.tensor([200.0, 0.02]),  # optional; can be omitted if you want auto bandwidth
        "alpha": 1e-6,  # small pseudo-count to stabilize ratios
        "smooth_sigma_bins": 1.0,  # try 0.8 to 1.5 (in bins)
        "smooth_log": True  # smooth log(r) for stability
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
                       # vmax=np.max(max_hist)*0.9
                       vmax=float(f.max())
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


