"""
Collection of utility functions and modules.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from auto_circuit.data import PromptDataLoader
from auto_circuit.metrics.prune_metrics.answer_diff_percent import (
    measure_answer_diff_percent,
)
from auto_circuit.metrics.prune_metrics.correct_answer_percent import (
    measure_correct_ans_percent,
)
from auto_circuit.metrics.prune_metrics.kl_div import measure_kl_div
from auto_circuit.types import CircuitOutputs, PruneScores
from auto_circuit.visualize import draw_seq_graph
from transformer_lens import HookedTransformer


def setup_logging(verbosity: int = 0) -> None:
    """
    Set up logging based on the verbosity level.

    Args:
        verbosity (int): Verbosity level from command line arguments.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generation in torch, numpy, and random libraries.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True


def get_save_dir_name(prefix: str, template: str) -> str:
    """
    Based on template, get name of directory where results should be saved.

    Args:
        prefix (str): Prefix to path.
        template (str): Name of template.

    Returns:
        str: Name of save directory.
    """
    if template == "full":
        return f"{prefix}/all_tokens"

    return f"{prefix}/tokenwise/template_{template}"


def save_dict_to_json(data: dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a JSON file, ensuring that the folder exists.

    Args:
        data (Dict[str, Any]): The dictionary to save.
        file_path (str): The path to the JSON file, including the file name.

    Raises:
        ValueError: If the provided file_path is invalid.
        OSError: If there is an error creating the folder or writing the file.
    """
    if not file_path.endswith(".json"):
        raise ValueError("The file path must end with '.json'")

    # Extract the directory from the file path
    directory = os.path.dirname(file_path)

    try:
        # Check if the directory exists, create it if necessary
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Write the dictionary to a JSON file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        logging.info(f"Data successfully saved to: {file_path}")

    except OSError as e:
        raise OSError(f"An error occurred while handling the file or directory: {e}")


def load_json_to_dict(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        Dict[str, Any]: A dictionary containing the data from the JSON file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(file_path, "r") as file:
            data: Dict[str, Any] = json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Error decoding JSON from file at {file_path}: {str(e)}", e.doc, e.pos
        )


def save_results(file_path: str, result_dict: dict[str, Any]) -> None:
    """
    Save the results of the circuit discovery experiment to a csv file.

    Args:
        file_path (str): Path to file where results should be saved.
        result_dict (dict[str, Any]): Dictionary summarizing results.
    """
    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(",".join(result_dict.keys()) + "\n")

    with open(file_path, "a") as f:
        f.write(",".join([str(value) for value in result_dict.values()]) + "\n")


def draw_circuit_graph(
    model: HookedTransformer,
    learned_prune_scores: PruneScores,
    score_threshold: float,
    file_path: str,
    seq_labels: list[str] | None = None,
) -> None:
    """
    Draw a graph of the model with the edges that are included in the circuit highlighted.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        learned_prune_scores (PruneScores): Dictionary mapping module names to edge scores.
        score_threshold (float): Threshold score that includes edges.
        file_path (str): Path to where image should be stored.
        seq_labels (list[str] | None, optional): Labels. Defaults to None.
    """
    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    orientation: Literal["h", "v"] = "h" if seq_labels is not None else "v"
    fig = draw_seq_graph(
        model,
        learned_prune_scores,
        score_threshold,
        layer_spacing=True,
        orientation=orientation,
        file_path=None,
        seq_labels=seq_labels,
    )
    fig.write_image(file_path)
    logging.info("Plot of circuit created.")


def draw_sparsity_vs_faithfulness_graph(
    model_name: str,
    metric: str,
    perturbation: str,
) -> None:
    """
    Draw a graph of the sparsity vs faithfulness metric value.

    Args:
        model_name (str): Name of model.
        metric (str): Name of metric.
        perturbation (str): Name of perturbation.
    """
    save_dir = "results/sparsity-vs-faithfulness/tokenwise"
    save_model_name = model_name.split("/")[-1].lower()
    save_dir_model = f"{save_dir}/{save_model_name}"
    save_filename = Path(f"{save_dir_model}/{metric}_{perturbation}.csv").absolute()

    df = pd.read_csv(save_filename)

    plt.figure(figsize=(10, 6))
    for template in df["template"].unique():
        template_data = df[df["template"] == template]
        plt.plot(
            template_data["sparsity"],
            template_data["metric_value"],
            marker="o",
            label=template,
        )

    plt.xlabel("% Edges")
    plt.ylabel("Faithfulness")
    plt.title("Edges included in the circuit vs Faithfulness")
    plt.legend(title="Template")

    save_img_path = Path(f"{save_dir_model}/{metric}_{perturbation}.png").absolute()
    plt.savefig(save_img_path)
    plt.close()


def draw_train_vs_faithfulness_graph(
    model_name: str,
    metric: str,
    perturbation: str,
) -> None:
    """
    Draw a graph of the train_size vs edges needed to reach a target faithfulness metric value,
    with different lines for each template.

    Args:
        model_name (str): Name of model.
        metric (str): Name of metric.
        perturbation (str): Name of perturbation
    """
    save_model_name = model_name.split("/")[-1].lower()
    save_dir = f"results/train-vs-faithfulness/tokenwise/{save_model_name}"
    save_filename = Path(f"{save_dir}/{metric}_{perturbation}.csv").absolute()

    df = pd.read_csv(save_filename)

    plt.figure(figsize=(10, 6))

    for template in df["template"].unique():
        template_data = df[df["template"] == template]
        plt.plot(
            template_data["train_size"],
            template_data["sparsity"],
            marker="o",
            label=template,
        )

    plt.xlabel("Train Size")
    plt.ylabel("Sparsity")
    plt.title("Train Size vs Sparsity needed for target Faithfulness")
    plt.legend()

    save_img_path = Path(f"{save_dir}/{metric}_{perturbation}.png").absolute()
    plt.savefig(save_img_path)
    plt.close()


def faithfulness_score(
    model: HookedTransformer,
    test_loader: PromptDataLoader,
    circ_outs: CircuitOutputs,
    metric: Literal["correct", "diff", "kl"],
) -> tuple[int | float, int | float]:
    """
    Compute the faithfulness of the model using the given metric.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        test_loader (PromptDataLoader): The test dataloader.
        circ_outs (CircuitOutputs): The circuit outputs.
        metric (Literal["correct", "diff", "kl"]): Metric to consider.

    Returns:
        tuple: Number of edges in the circuit, value of metric.
    """

    metric_functions = {
        "kl": measure_kl_div,
        "diff": measure_answer_diff_percent,
        "correct": measure_correct_ans_percent,
    }
    n_edge, metric_value = metric_functions[metric](model, test_loader, circ_outs)[0]

    return n_edge, metric_value


def plot_dual_y_axis(
    data: Dict[str, Dict[str, Union[Tuple, List]]],
    file_path: str,
    x_axis_label: str = "X-axis",
    x_axis_ticks: Optional[List[str]] = None,
    y_lim: List[int] = [0, 120],
    title: str = "Dual Y-axis Plot",
) -> None:
    """
    Creates and saves a dual y-axis plot based on the input dictionary.

    Parameters:
        data (Dict[str, Dict[str, Union[Tuple, List]]]):
            A dictionary where the first-level keys are x-axis values (as strings),
            and the values are dictionaries with two keys: "faithfulness" and "n_edges".
            Each of these keys maps to a tuple (mean, std_dev).
        file_path (str):
            The path where the plot will be saved. Ensures the directory exists.
        x_axis_label (Optional[str]):
            Label for the x-axis. Defaults to "X-axis".
        x_axis_ticks (Optional[List[str]]): Tick descriptions for x-axis. Defaults to None.
        y_lim (List[int]): Range of y-axis for faithfulness plotting. Defatults to [0, 120].
        title (Optional[str]):
            Title of the plot. Defaults to "Dual Y-axis Plot".

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        x_values = list(data.keys())
        x_numeric = np.arange(len(x_values))

        if x_axis_ticks is None:
            x_axis_ticks = [str(val) for val in x_numeric]
        else:
            assert len(x_axis_ticks) == len(
                x_values
            ), "'x_axis_ticks' is non-empty but of unequal length to x_values!"

        faithfulness_mean = np.array([data[x]["faithfulness"][0] for x in x_values])
        faithfulness_std = np.array([data[x]["faithfulness"][1] for x in x_values])
        n_edges_mean = np.array([data[x]["n_edges"][0] for x in x_values])
        n_edges_std = np.array([data[x]["n_edges"][1] for x in x_values])

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the first line (faithfulness)
        ax1.errorbar(
            x_numeric,
            faithfulness_mean,
            yerr=faithfulness_std,
            fmt="-o",
            label="Faithfulness",
            capsize=5,
        )
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylim(
            min(y_lim[0], min(1.2 * (faithfulness_mean - faithfulness_std))),
            max(y_lim[1], max(1.2 * (faithfulness_mean + faithfulness_std))),
        )
        ax1.set_ylabel("Faithfulness", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_xticks(x_numeric)
        ax1.set_xticklabels(x_axis_ticks, rotation=0, ha="center")

        # Create the second y-axis
        ax2 = ax1.twinx()
        ax2.errorbar(
            x_numeric,
            n_edges_mean,
            yerr=n_edges_std,
            fmt="-s",
            label="Number of Edges",
            color="green",
            capsize=5,
        )
        ax2.set_ylim(0, max(1.2 * (n_edges_mean + n_edges_std)))
        ax2.set_ylabel("Number of Edges", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # Add title and legend
        fig.suptitle(title)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(file_path, format="png", dpi=300)
        plt.close()

    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the plot: {e}")


def plot_edge_overlap_heatmap(
    triplets: List[Tuple[str, str, float]],
    labels: List[str],
    file_path: str,
    title: str = "Heatmap",
    colormap: str = "Blues",
) -> None:
    """
    Create and save a 3x3 heatmap from triplets of the form (label1, label2, value).

    Args:
        triplets (List[Tuple[str, str, float]]): A list of triplets, where each triplet is of the
            form (label1, label2, value), representing the value for the combination of label1 and label2.
        labels (List[str]): A list of unique labels in the order they should appear on the heatmap.
        file_path (str): The file path where the heatmap image will be saved.
        title (str, optional): The title of the heatmap. Defaults to "Heatmap".
        colormap (str, optional): The colormap to choose. Defaults to "Blues".

    Raises:
        ValueError: If the number of labels is not 3 or if the triplets do not include all combinations
            of the labels.
    """
    if len(labels) != 3:
        raise ValueError("The number of labels must be exactly 3.")

    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    heatmap = np.zeros((3, 3))

    # Populate the heatmap using the triplets
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    for label1, label2, value in triplets:
        i, j = label_to_index[label1], label_to_index[label2]
        heatmap[i, j] = value
        heatmap[j, i] = value  # Symmetric assignment

    # Create the heatmap plot
    plt.figure(figsize=(6, 5))
    img = plt.imshow(heatmap, cmap=colormap, vmin=0, vmax=1)  # Use the "Blues" colormap
    plt.colorbar(img, label="Value")  # Add colorbar

    # Set axis labels
    plt.xticks(ticks=range(3), labels=labels)
    plt.yticks(ticks=range(3), labels=labels)

    # Add values inside the heatmap cells
    for i in range(3):
        for j in range(3):
            plt.text(
                j,
                i,
                f"{heatmap[i, j]:.2f}",  # noqa: E231
                ha="center",
                va="center",
                color=(
                    "white" if heatmap[i, j] > 0.5 else "black"
                ),  # Use contrasting text color
            )

    # Add title
    plt.title(title)

    # Save the plot
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def bar_plot_attention_interventions(
    original_accuracies: List[float],
    intervened_accuracies: List[float],
    random_accuracies: List[float],
    file_path: str,
) -> None:
    """
    Create a bar plot comparing the accuracies of the original, intervened, and random interventions
    on attention heads within the error detection circuit of a model.

    Args:
        original_accuracies (List[float]): List of accuracies for the original model.
        intervened_accuracies (List[float]): List of accuracies for the intervened model.
        random_accuracies (List[float]): List of accuracies for the random interventions.
        file_path (str): File path to save the plot.
    """
    plt.rcParams.update({"font.size": 18})

    means = [
        np.mean(original_accuracies),
        np.mean(intervened_accuracies),
        np.mean(random_accuracies),
    ]
    stds = [
        np.std(original_accuracies),
        np.std(intervened_accuracies),
        np.std(random_accuracies),
    ]

    labels = ["Original", "Intervention", "Random"]
    x_positions = list(range(len(labels)))

    plt.style.use("seaborn-v0_8-paper")
    plt.figure(figsize=(6, 5))

    ax = plt.gca()
    ax.set_facecolor("white")

    ax.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, zorder=0)

    colors = ["#56b356", "#4c92c3", "tab:orange"]
    bars = plt.bar(
        x_positions, means, color=colors, align="center", width=0.8, zorder=3
    )

    for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
        x_center = x_positions[i]
        top = m + s
        if top > 1:
            capped_error = 1 - m
            plt.errorbar(
                x_center,
                m,
                yerr=capped_error,
                fmt="none",
                ecolor="black",
                capsize=5,
                capthick=1,
                zorder=4,
            )
        else:
            plt.errorbar(
                x_center,
                m,
                yerr=s,
                fmt="none",
                ecolor="black",
                capsize=5,
                capthick=1,
                zorder=4,
            )

    plt.xticks(x_positions, labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.tick_params(bottom=False, left=False)

    plt.ylim(0, 1.10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("grey")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def bar_plot_residual_interventions(
    original_accuracies_both: List[float],
    intervened_accuracies_both: List[float],
    original_accuracies_z1: List[float],
    intervened_accuracies_z1: List[float],
    file_path: str,
) -> None:
    """
    Create a bar plot comparing the accuracies of the original and intervened models
    on the residual stream of a model for both data and z1 data.

    Args:
        original_accuracies_both (List[float]): List of accuracies for the original model on both data.
        intervened_accuracies_both (List[float]): List of accuracies for the intervened model on both data.
        original_accuracies_z1 (List[float]): List of accuracies for the original model on z1 data.
        intervened_accuracies_z1 (List[float]): List of accuracies for the intervened model on z1 data.
        file_path (str): File path to save the plot.
    """
    plt.rcParams.update({"font.size": 18})

    means_both = [
        np.mean(original_accuracies_both),
        np.mean(intervened_accuracies_both),
    ]
    stds_both = [np.std(original_accuracies_both), np.std(intervened_accuracies_both)]
    means_z1 = [np.mean(original_accuracies_z1), np.mean(intervened_accuracies_z1)]
    stds_z1 = [np.std(original_accuracies_z1), np.std(intervened_accuracies_z1)]

    plt.style.use("seaborn-v0_8-paper")
    plt.figure(figsize=(7, 5))

    ax = plt.gca()
    ax.set_facecolor("white")

    ax.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, zorder=0)

    bar_width = 0.17
    r1 = [0, 0.8]
    r2 = [x + bar_width for x in r1]

    plt.bar(
        r1, [means_both[0], means_z1[0]], width=bar_width, color="#56b356", zorder=3
    )
    plt.bar(
        r2, [means_both[1], means_z1[1]], width=bar_width, color="#4c92c3", zorder=3
    )

    # For "both" data
    for x, m, s in zip(r1[:1] + r2[:1], means_both, stds_both):
        top = m + s
        if top > 1:
            capped_error = 1 - m
            plt.errorbar(
                x,
                m,
                yerr=capped_error,
                fmt="none",
                ecolor="black",
                capsize=5,
                capthick=1,
                zorder=4,
            )
        else:
            plt.errorbar(
                x,
                m,
                yerr=s,
                fmt="none",
                ecolor="black",
                capsize=5,
                capthick=1,
                zorder=4,
            )

    # For "z1" data
    for x, m, s in zip(r1[1:] + r2[1:], means_z1, stds_z1):
        top = m + s
        if top > 1:
            capped_error = 1 - m
            plt.errorbar(
                x,
                m,
                yerr=capped_error,
                fmt="none",
                ecolor="black",
                capsize=5,
                capthick=1,
                zorder=4,
            )
        else:
            plt.errorbar(
                x,
                m,
                yerr=s,
                fmt="none",
                ecolor="black",
                capsize=5,
                capthick=1,
                zorder=4,
            )

    plt.xticks(
        [r + bar_width / 2 for r in r1],
        ["Invalid\nResult & Answer", "Invalid Result"],
        fontsize=14,
    )
    plt.yticks(fontsize=16)
    plt.ylabel("Accuracy", fontsize=18)
    plt.tick_params(bottom=False, left=False)

    plt.ylim(0, 1.10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("grey")

    original_patch = mpatches.Patch(facecolor="#56b356", label="Original")
    intervention_patch = mpatches.Patch(facecolor="#4c92c3", label="Intervention")
    plt.legend(handles=[original_patch, intervention_patch], fontsize=11)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
