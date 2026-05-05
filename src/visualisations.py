"""File containing the logic of visualising of the networks performance.

Usage Example:
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# folder of this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# go back one step (into the project root), then into the data folder
ROOT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def __smoothen(data: list) -> list:
    """Smoothens data.

    Args:
        data: The original data.

    Returns:
        list: the smoothened data.
    """
    smoothed: list = []
    curr = data[0]

    for x in data:
        curr = 0.05 * x + (1 - 0.05) * curr
        smoothed.append(curr)

    return smoothed

def save_cost(cost_history: list, name: str) -> None:
    """Save the cost over iterations as a plot.

    Args:
        cost_history: A list of costs.
        name: The name of the folder the plot should be saved under.
    """
    # get the filepath using provided name
    save_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    file_dir = os.path.join(save_dir, "cost_over_iterations.png")

    # create smoothened data
    smooth: list = __smoothen(cost_history)

    # plot
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 5))

    sns.lineplot(data=cost_history, label="Raw", linewidth=1, alpha=0.7, color="purple")
    sns.lineplot(data=smooth, label="Smoothened", linewidth=1.5, color="black")

    plt.title("Training Cost Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()
    # log scaling for better visibility
    plt.xscale("symlog", linthresh=500)
    plt.xlim(left=-5)

    plt.show()
    plt.savefig(file_dir, dpi=200)
    plt.close()
