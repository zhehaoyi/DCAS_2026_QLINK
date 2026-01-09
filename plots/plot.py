import os
import torch
from circuit import quantum_res_circuit as qrc
from metrics import cost_function as cf
from matplotlib import pyplot as plt
import numpy as np
import tensorcircuit as tc

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
        "Q-LINK(Adaptive)":   {"line": "#52B788", "marker": "#2D6A4F"},
        "Vanilla":  {"line": "#2E86AB", "marker": "#A23B72"},
}

def plot_loss_comparison(num_qubits, results_dict, save_dir):
    """
    :param num_qubits: number of qubits
    :param results_dict: results dictionary with model names as keys and (avg_loss, std_loss) tuples as values
    :param save_dir: directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    for model, (avg_l, std_l) in results_dict.items():
        x = np.arange(len(avg_l))
        plt.plot(x, avg_l, label=f'{model}', linewidth=2, color=color_map[model]["line"])
        plt.fill_between(x, avg_l - std_l, avg_l + std_l, alpha=0.1)

    plt.xlabel('iteration', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(save_dir, f"n{num_qubits - 1}_loss_comparison.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_loss_landscape(num_qubits, depth, input_state, final_u, final_res, model, landscape_dir, data_dir,num_points):
    """
    :params num_qubits: number of qubits
    :params depth: circuit depth
    :params input_state: input quantum state
    :params final_u: trained u_params
    :params final_res: trained res_params
    :params model: model type
    :params landscape_dir: directory to save the plot
    :params num_points: how many points in each direction
    """
    # choose two random directions
    u_d1 = torch.randn_like(final_u)
    u_d2 = torch.randn_like(final_u)
    res_d1 = torch.randn_like(final_res)
    res_d2 = torch.randn_like(final_res)

    u_d1 = u_d1 / (torch.norm(u_d1)) * torch.norm(final_u)
    u_d2 = u_d2 / (torch.norm(u_d2)) * torch.norm(final_u)
    res_d1 = res_d1 / (torch.norm(res_d1)) * torch.norm(final_res)
    res_d2 = res_d2 / (torch.norm(res_d2)) * torch.norm(final_res)

    # define grid
    alphas = np.linspace(-3, 3, num_points)
    betas = np.linspace(-3, 3, num_points)
    x_axis, y_axis = np.meshgrid(alphas, betas)
    loss_landscape = np.zeros((num_points, num_points))

    print(f"Generating Landscape for n={num_qubits}, model={model}")

    # get device and dtype
    device = final_u.device
    dtype = final_u.dtype

    # convert alphas and betas to tensors
    alphas_t = torch.tensor(alphas, device=device, dtype=dtype)
    betas_t  = torch.tensor(betas, device=device, dtype=dtype)

    # torch meshgrid
    A_t, B_t = torch.meshgrid(alphas_t, betas_t, indexing="xy")  # A: (num_points,num_points) for alphas, B for betas
    a_flat = B_t.reshape(-1)  # a corresponds to betas (outer loop i)
    b_flat = A_t.reshape(-1)  # b corresponds to alphas (inner loop j)

    N = num_points * num_points
    a_u = a_flat.view(N, 1, 1, 1)
    b_u = b_flat.view(N, 1, 1, 1)
    a_r = a_flat.view(N, 1, 1)
    b_r = b_flat.view(N, 1, 1)

    # build batched params
    curr_u = final_u + a_u * u_d1 + b_u * u_d2
    curr_res = final_res + a_r * res_d1 + b_r * res_d2

    # single-point loss
    def one_loss(u, r):
        probs = qrc(num_qubits, depth, u, r, input_state, model, return_state=False)
        loss = cf(probs, num_qubits, model)
        return loss

    # compute all losses
    with torch.no_grad():
        try:
            # torch vmap over first dim
            losses = torch.vmap(one_loss, in_dims=(0, 0))(curr_u, curr_res)  # (N,)
        except Exception as e:
            # fallback: batch loop
            batch_size = 128
            losses = torch.empty((N,), device=device, dtype=dtype)
            for s in range(0, N, batch_size):
                e2 = min(s + batch_size, N)
                losses[s:e2] = torch.vmap(one_loss, in_dims=(0, 0))(curr_u[s:e2], curr_res[s:e2])

        Z = losses.view(num_points, num_points)  # matches (betas, alphas) due to a_flat=b_flat mapping above

    loss_landscape[:, :] = Z.detach().cpu().numpy()

    # save the landscape data for future use, may be we need to re-plot later
    n_for_name = num_qubits - 1 if model != "Vanilla" else num_qubits
    npz_path = os.path.join(data_dir, f"n{n_for_name}_{model}_landscape_data.npz")
    np.savez_compressed(
        npz_path,
        num_qubits=num_qubits,
        depth=depth,
        model=model,
        num_points=num_points,
        alphas=alphas,
        betas=betas,
        loss_landscape=loss_landscape,
        u_d1=u_d1.detach().cpu().numpy(),
        u_d2=u_d2.detach().cpu().numpy(),
        res_d1=res_d1.detach().cpu().numpy(),
        res_d2=res_d2.detach().cpu().numpy(),
        final_u=final_u.detach().cpu().numpy(),
        final_res=final_res.detach().cpu().numpy(),
        input_state=input_state.detach().cpu().numpy() if torch.is_tensor(input_state) else input_state,
    )

    # plot the 3d surface of loss landscape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_axis, y_axis, loss_landscape, cmap='terrain', edgecolor='none', antialiased=True)

    ax.set_xlabel('direction 1', fontsize=10)
    ax.set_ylabel('direction 2', fontsize=10)
    ax.set_zlabel('loss', fontsize=10)

    zmin = loss_landscape.min()
    zmax = loss_landscape.max()
    ax.set_zlim(zmin, zmax * 1.1)
    ax.zaxis.set_label_coords(-0.08, 0.5)

    ax.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12)    
    ax.view_init(elev=35, azim=45)

    if model != "Vanilla":
        num_qubits = num_qubits - 1 # to match the plot title
    save_path = os.path.join(landscape_dir, f"n{num_qubits}_{model}_landscape.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def plot_iter_vs_qubits_scatter(qubits_list, iteration_mean, base_dir):
    """
    :param qubits_list: qubit numbers
    :param iteration_mean: dictionary with model names as keys and list of average iterations as values
    :param base_dir: directory to save the plot
    """

    plt.figure(figsize=(6.5, 4.5))

    for model, y_vals in iteration_mean.items():
        x_vals = qubits_list

        # line
        plt.plot(x_vals, y_vals, color=color_map[model]["line"], linewidth=2, alpha=0.9, label=model)

        # scatter
        plt.scatter(x_vals, y_vals, s=80, color=color_map[model]["marker"], edgecolor=color_map[model]["line"], linewidth=1.5, zorder=3)

        # annotate each point
        for x, y in zip(x_vals, y_vals):
            plt.annotate(f"{y}", xy=(x, y),fontsize=10, ha="center", va="bottom", textcoords="offset points", xytext=(0, 5), color=color_map[model]["line"], fontweight="bold")

    plt.xticks(x_vals)
    plt.xlabel("qubits", fontsize=12)
    plt.ylabel("average iterations", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()

    save_path = os.path.join(base_dir, "avgiter_vs_qubits.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()