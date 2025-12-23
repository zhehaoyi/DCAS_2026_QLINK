import os
def create_file():
    """
    create all the necessary directories for saving results
    :return: dictionary of directory paths
    """
    base_dir = "Q-LINK_results"

    dirs = {
        "base": base_dir,
        "landscape": os.path.join(base_dir, "loss_landscapes"),
        "loss_compare": os.path.join(base_dir, "loss_comparisons"),
        "avgiter": os.path.join(base_dir, "avg_stopping_iterations"),
        "summary": os.path.join(base_dir, "summaries"),
        "data": os.path.join(base_dir, "data"),
        "replots": os.path.join(base_dir, "replots"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs
