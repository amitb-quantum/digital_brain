"""Figures written next to each run's metrics.json."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def receptive_fields(w, size, path, title):
    n = w.shape[1]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.full((rows * (size + 1), cols * (size + 1)), np.nan)
    for k in range(n):
        r, c = divmod(k, cols)
        grid[r * (size + 1):r * (size + 1) + size,
             c * (size + 1):c * (size + 1) + size] = w[:, k].reshape(size, size)
    fig, ax = plt.subplots(figsize=(8, 8 * rows / cols))
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, label="weight")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def selectivity(pre, post, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bins = np.linspace(0, 1, 21)
    for ev, label, color in ((pre, "before training", "#888888"),
                             (post, "after training", "#1f77b4")):
        axes[0].hist(ev["osi"][~np.isnan(ev["osi"])], bins=bins, alpha=0.6,
                     label=f"{label} (mean {ev['summary']['mean_osi']:.2f})", color=color)
    axes[0].set_xlabel("orientation selectivity index")
    axes[0].set_ylabel("neurons")
    axes[0].legend()

    pref = np.degrees(post["pref"][~np.isnan(post["pref"])])
    axes[1].hist(pref, bins=np.linspace(0, 180, 9), color="#1f77b4", alpha=0.8)
    axes[1].set_xlabel("preferred orientation after training (deg)")
    axes[1].set_ylabel("neurons")

    tuning = post["tuning"]
    n_or = tuning.shape[1]
    degs = np.arange(n_or) * 180 / n_or
    best = np.argsort(np.nan_to_num(post["osi"], nan=-1))[::-1][:6]
    for k in best:
        axes[2].plot(degs, tuning[k], marker="o", label=f"n{k} OSI {post['osi'][k]:.2f}")
    axes[2].set_xlabel("stimulus orientation (deg)")
    axes[2].set_ylabel("mean spikes / presentation")
    axes[2].set_title("most selective neurons")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
