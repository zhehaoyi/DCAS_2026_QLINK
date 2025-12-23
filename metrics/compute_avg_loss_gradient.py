import numpy as np

def compute_avg_loss_gradient_stopiter(all_losses, all_grads, stop_iterations):
    """
    :param all_losses: A list where each element is a list of loss values from a model
    :param all_grads: A list where each element is a list of gradient arrays from a model
    :param stop_iterations: A list where each element is the stopping iteration for a run
    :returns: avg_loss: The average loss across all runs, aligned to the longest run.
    :returns: std_loss: The standard deviation of the loss across all runs.
    :returns: avg_grad: The average gradient across all runs, truncated to the shortest run length.
    """

    # compute average and std of loss for each model
    max_l = max(len(l) for l in all_losses)
    padded_l = np.array([l + [l[-1]]*(max_l-len(l)) for l in all_losses])
    avg_l, std_l = np.mean(padded_l, axis=0), np.std(padded_l, axis=0)
            
    # compute average gradient of gradient for each model (truncate to the shortest length)
    min_g = min(len(g) for g in all_grads)
    avg_g = np.mean([g[:min_g] for g in all_grads], axis=0) # (iters, depth, data_n, 3)
    grad_variance = np.var(avg_g)

    # compute average stopping iteration
    avg_i = int(np.mean(stop_iterations))

    return avg_l, std_l, grad_variance, avg_i