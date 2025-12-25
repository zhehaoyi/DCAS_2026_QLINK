import json
import os
import random
from matplotlib import pyplot as plt
from numpy import pi
from circuit import create_random_state, quantum_res_circuit, circuit_depth
from metrics import compute_avg_loss_gradient_stopiter, ComputeExpress, cost_function
from plots import plot_loss_comparison, plot_loss_landscape, plot_iter_vs_qubits_scatter
from create_file import create_file
import tensorcircuit as tc
import torch
import numpy as np
tc.set_backend("pytorch")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# set the random seed for reproducibility
set_seed(42)

def main():
    # create necessary directories
    dirs = create_file()
    base_dir = dirs["base"]
    landscape_dir = dirs["landscape"]
    loss_compare_dir = dirs["loss_compare"]
    avgiter_dir = dirs["avgiter"]  
    summary_dir = dirs["summary"]
    data_dir = dirs["data"]
    replot_dir = dirs["replots"]


    qubit_list = list(range(5, 11)) # qubit numbers to simulate
    models = ["Q-LINK(Fixed)", "Q-LINK(Adaptive)", "Vallina"]
    n_repeats = 5
    num_iterations = 1500
    n_loss_stats = {}
    iteration_mean = {m: [] for m in models} # record the mean iteration for each model

    for n in qubit_list:
        print(f"\n{'='*20} Simulation n = {n} (include proxy qubit) {'='*20}")     
        summary_file = os.path.join(summary_dir, f"n{n}_summary.txt")
        with open(summary_file, "w") as f: f.write(f"Results for {n} qubits experiment\n")

        for model in models:
            all_qubits = n if model != "Vallina" else n - 1 # all qubits
            data_qubits = n - 1 # data qubits

            depth = circuit_depth(data_qubits)
            
            all_losses = [] # record the loss of each run
            all_grads = [] # record the gradient of u_params
            stop_iteration = [] # record the stopping iteration of each run

            for r in range(n_repeats):
                input_state = create_random_state(data_qubits, model)
                u_params = torch.nn.Parameter(torch.randn(depth, data_qubits, 3) * 0.1) # operting layer parameters
                res_params = torch.nn.Parameter(torch.randn(depth + 1, data_qubits) * 0.1) # residual layer parameters
                optimizer = torch.optim.SGD([u_params, res_params], lr=0.1)
                
                single_loss = [] # record the loss of this run
                single_grad = [] # record the gradient of this run

                stop_it = num_iterations
                for i in range(num_iterations):
                    optimizer.zero_grad()
                    probs = quantum_res_circuit(all_qubits, depth, u_params, res_params, input_state, model, return_state=False)
                    loss = cost_function(probs, all_qubits, model)
                    loss.backward()

                    # only recode the gradient of u_params
                    single_grad.append(u_params.grad.clone().detach().numpy())
                    single_loss.append(loss.item())
                    optimizer.step()

                    # early stopping
                    if loss < 1e-3:
                        stop_it = i + 1
                        break
                    if i % 10 == 0:
                        print(f"Model: {model}, Qubits: {n}, Run: {r+1}, Iteration: {i}, Loss: {loss.item():.6f}")

                all_losses.append(single_loss)
                all_grads.append(single_grad)
                stop_iteration.append(stop_it)

            # compute average loss and std of loss for each model
            # compute average gradient of u_params for each model
            # compute average stopping iteration for each model
            avg_l, std_l, grad_variance, avg_i = compute_avg_loss_gradient_stopiter(all_losses, all_grads, stop_iteration)

            # recode the avg loss and std loss for plotting later
            n_loss_stats[model] = (avg_l, std_l)
            iteration_mean[model].append(avg_i)


            # Expressibility
            compute_expressibility = ComputeExpress(num_bins=20, num_qubits=all_qubits, num_fidelity=500, depth=depth, model=model)
            expressibility = compute_expressibility.compute_express()

            # record summary
            with open(summary_file, "a") as f:
                f.write(f"Model {model}: KL={expressibility:.6f}, Final Loss={avg_l[-1]:.6f}, Avg Stopping Iteration={avg_i}, Gradient Variance={grad_variance}\n")
            
            plot_loss_landscape(all_qubits, depth, input_state, u_params, res_params, model, landscape_dir, data_dir,num_points=200)
        
        # save loss comparison data for this qubit number
        loss_compare_data_dir = os.path.join(data_dir, f"n{n}_loss_comparison_data.json")
        loss_compar_dataload = {"n":n, "models":{}}
        for model, (avg_l, std_l) in n_loss_stats.items():
            loss_compar_dataload["models"][model] = {
                "iteration": list(range(len(avg_l))),
                "avg_loss": avg_l.tolist(),
                "std_loss": std_l.tolist()
            }
        with open(loss_compare_data_dir, "w") as f:
            json.dump(loss_compar_dataload, f, indent=2)
        plot_loss_comparison(n, n_loss_stats, loss_compare_dir)


    # plot average stopping iteration vs qubit number
    # record data for future use
    avgiter_data_dir = os.path.join(data_dir, f"avg_stopping_iteration_vs_qubits.json")
    avgiter_dataload = {"qubit_list":  [x for x in qubit_list], "iteration_mean": {m: [int(v) for v in vals] for m, vals in iteration_mean.items()}}
    with open(avgiter_data_dir, "w") as f:
        json.dump(avgiter_dataload, f, indent=2)
    plot_iter_vs_qubits_scatter(qubit_list, iteration_mean, avgiter_dir)


if __name__ == "__main__":
    main()