import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt


# the base directory for results
BASE_DIR = "residual_network/residual_network_results"
DATA_DIR = os.path.join(BASE_DIR, "data")
REPLOT_DIR = os.path.join(BASE_DIR, "replots")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.edgecolor': 'black',
    'axes.grid': True
})

color_map = {
    "Q-LINK(Fixed)":     {"line": "#FB8500", "marker": "#E85D04"},
    "Q-LINK(Adaptive)":  {"line": "#52B788", "marker": "#2D6A4F"},
    "Vallina":           {"line": "#2E86AB", "marker": "#A23B72"},
}


# change the model name if there is a typo
def fix_model_name(name):
    if isinstance(name, str) and name.strip() == "Q-LINK(Adptive)":
        return "Q-LINK(Adaptive)"
    return name


def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9_\-\(\)\.]+", "_", s)


# replot loss comparison
def replot_loss_comparison(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    n = int(data.get("n", -1))
    models = data.get("models", {})

    plt.figure(figsize=(10, 6))
    for model, payload in models.items():
        model = fix_model_name(model)

        avg_loss = np.asarray(payload["avg_loss"], dtype=float)
        std_loss = np.asarray(payload["std_loss"], dtype=float)

        x = np.arange(len(avg_loss))
        if model in color_map:
            plt.plot(x, avg_loss, linewidth=2, color=color_map[model]["line"], label=model)
        else:
            plt.plot(x, avg_loss, linewidth=2, label=model)

        plt.fill_between(x, avg_loss - std_loss, avg_loss + std_loss, alpha=0.1)

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # add "replots_" prefix to distinguish from original plots
    out_name = f"replots_n{n}_loss_comparison.pdf"
    out_path = os.path.join(REPLOT_DIR, safe_filename(out_name))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# replot average stopping iteration vs qubits
def replot_avgiter(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    qubits = [int(x) for x in data["qubit_list"]]
    iteration_mean = data["iteration_mean"]

    plt.figure(figsize=(6.5, 4.5))
    for model, y_vals in iteration_mean.items():
        model = fix_model_name(model)
        y_vals = [int(v) for v in y_vals]

        if model in color_map:
            plt.plot(qubits, y_vals, linewidth=2, color=color_map[model]["line"], label=model)
            plt.scatter(qubits, y_vals, s=80,
                        color=color_map[model]["marker"],
                        edgecolor=color_map[model]["line"],
                        linewidth=1.5, zorder=3)
        else:
            plt.plot(qubits, y_vals, linewidth=2, label=model)
            plt.scatter(qubits, y_vals, s=80, zorder=3)

        for x, y in zip(qubits, y_vals):
            plt.annotate(str(y), (x, y), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=10)

    plt.xlabel("qubits")
    plt.ylabel("average iterations")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = os.path.join(REPLOT_DIR, "replots_avgiter_vs_qubits.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# replot loss landscape
def scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.shape == () else v


def replot_landscape(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        num_qubits = int(scalar(d["num_qubits"]))
        model = fix_model_name(str(scalar(d["model"])))
        alphas = d["alphas"]
        betas = d["betas"]
        loss_landscape = d["loss_landscape"]

    X, Y = np.meshgrid(alphas, betas)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, loss_landscape, cmap="terrain", edgecolor="none")

    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.set_zlabel("loss")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12)

    plot_n = num_qubits if model == "Vallina" else (num_qubits - 1)
    out_name = f"replots_n{plot_n}_{model}_landscape.pdf"
    out_path = os.path.join(REPLOT_DIR, safe_filename(out_name))

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()



# scan the data directory and replot all
def scan_and_replot_all(default_std):
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"DATA_DIR not found: {DATA_DIR}")
    if not os.path.isdir(REPLOT_DIR):
        raise RuntimeError(f"REPLOT_DIR not found: {REPLOT_DIR}")

    files = sorted(os.listdir(DATA_DIR))

    for f in files:
        path = os.path.join(DATA_DIR, f)

        if f.endswith("_loss_comparison_data.json"):
            replot_loss_comparison(path, default_std)

        elif f == "avg_stopping_iteration_vs_qubits.json":
            replot_avgiter(path)

        elif f.endswith("_landscape_data.npz"):
            replot_landscape(path)


if __name__ == "__main__":
    scan_and_replot_all()
