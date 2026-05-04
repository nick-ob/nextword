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
