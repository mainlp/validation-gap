"""
Functions and modules to run circuit discovery.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Literal

import torch as t
from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import AblationType, CircuitOutputs, PatchType, PruneScores
from tqdm import tqdm
from transformer_lens import HookedTransformer

from llm_error_detection.circuits.circuit_analysis import Circuit
from llm_error_detection.utils import faithfulness_score


def load_model(model_name: str, device: t.device, cache_dir: str) -> HookedTransformer:
    """
    Load a pretrained HookedTransformer model in lower precision (bfp16).

    Args:
        model_name (str): Name of the model to load from the Hugging Face model hub.
        device (t.device): Device to use for computation.
        cache_dir (str): Directory to cached model weights.

    Returns:
        HookedTransformer: A HookedTransformer model loaded from the Hugging Face model hub.
    """
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        cache_dir=cache_dir,
        dtype="bfloat16",
    )
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_attn_in = True
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.eval()
    for param in tl_model.parameters():
        param.requires_grad = False
    return tl_model


def load_data(
    model: HookedTransformer,
    data_dir: str,
    save_model_name: str,
    template: str,
    perturbation: Literal["z1", "z2", "both", "shortened_z1", "computation"],
    num_train: int,
    num_test: int,
    batch_size: int,
    device: t.device,
    filtered: bool = True,
) -> tuple[PromptDataLoader, PromptDataLoader, List[str]]:
    """
    Load a filtered dataset of clean and corrupt paired input and outputs from a json file.
    Data is filtered to only include the examples of the task that a model can classify correctly.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        data_dir (str): Data directory.
        save_model_name (str): Name under which to save model.
        template (str): Template name.
        perturbation (Literal["z1", "z2", "both", "shortened_z1", "computation"]): Type of perturbation.
        num_train (int): Number of train samples.
        num_test (int): Number of test samples.
        batch_size (int): Batch size.
        device (t.device): Device to use for computation.
        filtered (bool): Whether to use filtered data where the model can classify correctly.

    Returns:
        tuple[PromptDataLoader, PromptDataLoader]: A tuple of trainloader, testloader, and sequence labels.
    """

    template = "full" if template == "full" else f"template_{template}"
    return_seq_length = False if template == "full" else True

    path = (
        Path(f"{data_dir}/{template}/math_{save_model_name}_{perturbation}.json")
        if filtered
        else Path(
            f"{data_dir}/{template}/math_{save_model_name}_{perturbation}_no_check.json"
        )
    )

    # Check that enough promtps are available for the specified train and test sizes
    with open(path, "r") as f:
        data = json.load(f)
    num_prompts = len(data["prompts"])

    if num_prompts < (num_train + num_test):
        if num_prompts < num_train:
            num_train = int(num_prompts * 0.8)  # Use 80% for training
            num_test = num_prompts - num_train
        else:
            num_test = num_prompts - num_train
        logging.warning(
            f"Not enough prompts for the specified train and test sizes!\nAdjusting sizes: train={num_train}, test={num_test}"
        )

    train_loader, test_loader = load_datasets_from_json(
        model=model,
        path=path,
        device=device,
        prepend_bos=False,
        batch_size=batch_size,
        train_test_size=(num_train, num_test),
        shuffle=True,
        random_seed=42,
        return_seq_length=return_seq_length,
        tail_divergence=False,
    )
    return train_loader, test_loader, data["seq_labels"]


def find_top_k_edges_threshold(prune_scores: PruneScores, top_k: int) -> float:
    """
    Retrieve the top k edges based on the absolute values of scores in prune_scores.

    Args:
        prune_scores (PruneScores): Dictionary mapping module names to edge scores.
        top_k (int): Top-k edges to keep.

    Returns:
        float: Threshold score that includes top-k edges.
    """
    edge_scores = prune_scores.values()
    listed_ts = [v.abs().flatten() for v in edge_scores]
    ordered_ts = t.sort(t.cat(listed_ts), descending=True)
    return ordered_ts.values[top_k - 1].item()


def learn_edge_scores(
    model: HookedTransformer,
    train_loader: PromptDataLoader,
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "avg_val", "mse"],
) -> PruneScores:
    """
    Assign a prune score to each edge in the model.
    Prune scores equal to the gradient of the mask values that interpolates the edges between
    the clean activations and the ablated activations.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        train_loader (PromptDataLoader): The train dataloader.
        grad_function (Literal["logit", "prob", "logprob", "logit_exp"]): The gradient function to use.
        answer_function (Literal["avg_diff", "avg_val", "mse"]): Loss function of the model output which the gradient is taken
            with respect to.

    Returns:
        PruneScores: The edge scores.
    """
    return mask_gradient_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
        grad_function=grad_function,
        answer_function=answer_function,
        mask_val=0.0,
    )


def find_threshold_edges(
    model: HookedTransformer,
    test_loader: PromptDataLoader,
    learned_prune_scores: PruneScores,
    initial_num_edges: int,
    min_threshold: float,
    max_threshold: float,
    metric: Literal["correct", "diff", "kl"],
    step_size: int,
) -> tuple[int | float, int | float, int]:
    """
    Find the number of edges to include in the circuit based on a target faithfulness value.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        test_loader (PromptDataLoader): The test dataloader.
        learned_prune_scores (PruneScores): Dictionary mapping module names to edge scores.
        initial_num_edges (int): Number of edges to consider initially.
        min_threshold (float): Minimum acceptable faithfulness score.
        max_threshold (float): Maximum acceptable faithfulness score.
        metric (Literal["correct", "diff", "kl"]): Metric to consider.
        step_size (int): Additional number of edges to prune each iteration.

    Returns:
        tuple[int | float, int | float, int]: Number of remaining edges, metric value, number of edges pruned.
    """
    test_count = [initial_num_edges]
    metric_value = 0.0

    while metric_value < float(min_threshold) or metric_value > float(max_threshold):
        circ_outs: CircuitOutputs = run_circuits(
            model=model,
            dataloader=test_loader,
            test_edge_counts=test_count,
            prune_scores=learned_prune_scores,
            patch_type=PatchType.EDGE_PATCH,
            ablation_type=AblationType.RESAMPLE,
            render_graph=False,
        )

        n_edge, metric_value = faithfulness_score(model, test_loader, circ_outs, metric)

        test_count[0] += step_size

        if test_count[0] > model.n_edges:
            break

    return n_edge, metric_value, test_count[0]


def save_prune_scores(prune_scores: PruneScores, file_path: str) -> None:
    """
    Save a PruneScores dictionary to a .pth file.

    Args:
        prune_scores (PruneScores): The dictionary containing module names as keys and
                                    edge scores as PyTorch tensors.
        file_path (str): The file path where the .pth file will be saved.

    Raises:
        ValueError: If the provided prune_scores is not a dictionary of the expected format.
        IOError: If the file cannot be saved to the specified location.
    """
    if not isinstance(prune_scores, dict):
        raise ValueError("prune_scores must be a dictionary.")
    if not all(
        isinstance(k, str) and isinstance(v, t.Tensor) for k, v in prune_scores.items()
    ):
        raise ValueError(
            "All keys in prune_scores must be strings, and all values must be torch.Tensors."
        )

    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    try:
        # Save the dictionary to the specified file path
        t.save(prune_scores, file_path)
        logging.info("PruneScores saved successfully.")
    except Exception as e:
        raise IOError(f"Failed to save prune_scores to {file_path}: {e}")


def load_prune_scores(file_path: str) -> PruneScores:
    """
    Load a PruneScores dictionary from a .pth file.

    Args:
        file_path (str): The file path from which the .pth file will be loaded.

    Returns:
        PruneScores: The dictionary containing module names as keys and
                     edge scores as PyTorch tensors.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the loaded data is not in the expected format.
        IOError: If the file cannot be loaded due to an I/O error.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        # Load the dictionary from the specified file path
        prune_scores = t.load(file_path)
    except Exception as e:
        raise IOError(f"Failed to load prune_scores from {file_path}: {e}")

    if not isinstance(prune_scores, dict):
        raise ValueError("Loaded data is not a dictionary.")
    if not all(
        isinstance(k, str) and isinstance(v, t.Tensor) for k, v in prune_scores.items()
    ):
        raise ValueError(
            "All keys in the loaded prune_scores must be strings, and all values must be torch.Tensors."
        )

    logging.info("PruneScores loaded successfully.")
    return prune_scores


def obtain_circuit(
    model: HookedTransformer,
    prune_scores: PruneScores,
    score_threshold: float,
    seq_labels: list[str],
) -> Circuit:
    """
    Obtain the circuit from the given model and prune scores.

    Args:
        model (HookedTransformer): The model to extract the circuit from.
        prune_scores (PruneScores): The prune scores used to determine the circuit.
        score_threshold (float): The score below which an edge is not included in the circuit.

    Returns:
        Circuit: The circuit of the given model, with edges that have a score below the
                 score threshold removed.
    """
    circuit = Circuit()

    for seq_idx, edges in tqdm(
        model.edge_dict.items(),
        total=len(model.edge_dict),
        desc="Extracting circuit",
    ):
        seq_label = seq_labels[seq_idx]
        edge_set = set(edges)

        for edge in edge_set:
            edge_score = edge.prune_score(prune_scores).item()
            if abs(edge_score) < score_threshold:
                continue
            circuit.add_edge(seq_label, edge)

    return circuit


def faithfulness_given_sparsity(
    model: HookedTransformer,
    test_loader: PromptDataLoader,
    learned_prune_scores: PruneScores,
    sparsity: float,
    metric: Literal["correct", "diff", "kl"],
) -> tuple[int | float, int | float]:
    """
    Find the faithfulness obtained by the circuit that includes an amount of edges
    defined by the sparsity parameter.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        test_loader (PromptDataLoader): The test dataloader.
        learned_prune_scores (PruneScores): Dictionary mapping module names to edge scores.
        sparsity (float): The model's desired sparsity (0.0 represents full model).
        metric (Literal["correct", "diff", "kl"]): Metric to consider.

    Returns:
        tuple: Number of remaining edges, value of metric.
    """
    test_count = [int(model.n_edges * sparsity)]

    circ_outs: CircuitOutputs = run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=test_count,
        prune_scores=learned_prune_scores,
        patch_type=PatchType.EDGE_PATCH,
        ablation_type=AblationType.RESAMPLE,
        render_graph=False,
    )

    n_edge, metric_value = faithfulness_score(model, test_loader, circ_outs, metric)

    return n_edge, metric_value
