"""
Script to intervene on attention components on the arithmetic error detection task.
"""

import argparse
import json
import logging
import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch as t
from transformer_lens import HookedTransformer

from llm_error_detection.circuits.circuit_discovery import load_data, load_model
from llm_error_detection.circuits.intervene.attention_intervention import (
    intervene_on_attention_heads,
)
from llm_error_detection.utils import (
    bar_plot_attention_interventions,
    set_seed,
    setup_logging,
)

TEMPLATES = ["0", "1", "2", "3", "4", "5", "6", "7"]


Z1_AND_Z2_LABELS = [
    ("[z1-second]_occ_1", "[z2-second]_occ_1"),
    ("[z1]_occ_1", "[z2]_occ_1"),
]


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line input arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # General config
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/math",
        help="Directory to load the data from",
    )
    parser.add_argument(
        "--cache_dir", type=str, help="Directory of cached model weights",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )

    # Data config
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for data loading"
    )

    # Model & circuit config
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Name of the model to use"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=500,
        help="Number of samples used to compute edge scores",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of samples used to select edges to include in circuit based",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="diff",
        choices=["correct", "diff", "kl"],
        help="Faithfulness metric to use",
    )
    parser.add_argument(
        "--grad_function",
        type=str,
        default="logit",
        choices=["logit", "prob", "logprob", "logit_exp"],
        help="Function to apply to logits for finding edge scores before computing gradient",
    )
    parser.add_argument(
        "--answer_function",
        type=str,
        default="avg_diff",
        choices=["avg_diff", "avg_val", "mse"],
        help="Loss function to apply to answer and wrong answer for finding edge scores",
    )
    parser.add_argument(
        "--single_perturbation_data",
        type=str,
        default="z1",
        choices=["z1", "z2"],
        help="Type of perturbation data to used to gather activations on single perturbation data",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Scaling factor for the attention head intervention",
    )
    parser.add_argument(
        "--save_plot_dir",
        type=str,
        default="results/attention-interventions",
        help="Directory to save the attention intervention plots",
    )
    args = parser.parse_args()

    return args


def subsample_math_data(data: dict, num_samples: int) -> dict:
    """
    Subsample math data to a specified number of samples.

    Args:
        data (dict): The input data to be subsampled.
        num_samples (int): The number of samples to include in the subsampled data.

    Returns:
        dict: The subsampled data.
    """
    sampled_idx = random.sample(range(len(data["prompts"])), num_samples)
    prompts = [data["prompts"][i] for i in sampled_idx]
    data["prompts"] = prompts
    return data


def get_random_heads(
    model_name: str, num_samples: int, num_heads: int, num_layers: int
) -> List[Tuple[int, int]]:
    """
    Get a list of random attention heads with no duplicates.

    Args:
        model_name (str): The name of the model.
        num_samples (int): The number of samples to select.
        num_heads (int): The number of heads to select.
        num_layers (int): The number of layers in the model.

    Returns:
        List[Tuple[int, int]]: A list of unique (layer, head) tuples.
    """
    # Create random heads dictionary if it doesn't exist
    if not os.path.exists("annotated_heads/random_heads.json"):
        with open("annotated_heads/consistency_heads.json", "r") as f:
            important_heads_dict = json.load(f)
        random_heads_dict: Dict[str, List[Tuple[int, int]]] = {
            key: [] for key in important_heads_dict.keys()
        }
        with open("annotated_heads/random_heads.json", "w") as f:
            json.dump(random_heads_dict, f, indent=2)

    # Load random heads dictionary
    with open("annotated_heads/random_heads.json", "r") as f:
        random_heads_dict = json.load(f)

    if random_heads_dict[model_name] != []:  # Return random heads if they already exist
        return random_heads_dict[model_name]
    else:  # Generate random heads and save them to the dictionary
        with open("annotated_heads/consistency_heads.json", "r") as f:
            important_heads_dict = json.load(f)
        all_heads = [
            (layer, head)
            for layer in range(num_layers)
            for head in range(num_heads)
            if [layer, head] not in important_heads_dict[model_name]
        ]
        random_heads = random.sample(all_heads, num_samples)
        random_heads_dict[model_name] = random_heads
        with open("annotated_heads/random_heads.json", "w") as f:
            json.dump(random_heads_dict, f, indent=2)
        return random_heads


def get_model_heads(model_name: str) -> List[Tuple[int, int]]:
    """
    Get a list of attention heads to patch for the error detection task
    for a specific model.

    Args:
        model_name (str): The name of the model.

    Returns:
        List[Tuple[int, int]]: A list of (layer, head) tuples.
    """
    # Load important heads from JSON file
    with open("annotated_heads/consistency_heads.json", "r") as f:
        heads_dict = json.load(f)

    # Get heads for specified model
    if model_name not in heads_dict:
        raise ValueError(f"Model {model_name} not found in heads dictionary")

    # Convert string tuples to actual tuples
    heads = heads_dict[model_name]
    return [(int(h[0]), int(h[1])) for h in heads]


def prepare_data(
    model: HookedTransformer,
    model_name: str,
    template: str,
    args: argparse.Namespace,
) -> Tuple[List[t.Tensor], List[t.Tensor]]:
    """
    Prepare data for causal intervention on attention heads.
    This function loads the filtered math data for a single perturbation
    (z1 or z2) and create a paired dataloader with the same data but
    perturbation in both z1 and z2 positions.

    Args:
        model (HookedTransformer): The model to use for the intervention.
        model_name (str): The name of the model.
        template (str): The template to load the math data for.
        args (argparse.Namespace): The input arguments.

    Returns:
        Tuple[List[t.Tensor], List[t.Tensor]]: The data loaders for the single perturbation
        and both perturbation data.
    """

    _, promptloader_single, seq_labels = load_data(
        model=model,
        data_dir=args.data_dir,
        save_model_name=model_name,
        template=template,
        perturbation=args.single_perturbation_data,
        num_train=args.train_size,
        num_test=args.test_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    if "llama-3.2" in model_name.lower():
        seq_labels = ["[bos]"] + seq_labels

    test_loader_single: List[t.Tensor] = [batch.clean for batch in promptloader_single]
    test_loader_both: List[t.Tensor] = []

    indexes_z1_z2 = [
        (seq_labels.index(z1), seq_labels.index(z2))
        for z1, z2 in Z1_AND_Z2_LABELS
        if z1 in seq_labels and z2 in seq_labels
    ]

    for batch in test_loader_single:
        new_batch = deepcopy(batch)
        if args.single_perturbation_data == "z1":
            new_batch[:, indexes_z1_z2[0][1]] = batch[:, indexes_z1_z2[0][0]]
        else:
            new_batch[:, indexes_z1_z2[0][0]] = batch[:, indexes_z1_z2[0][1]]
        test_loader_both.append(new_batch)

    return test_loader_single, test_loader_both


def main() -> None:
    """
    Main script.
    """
    args = parse_arguments()
    args.device = (
        t.device("cuda")
        if t.cuda.is_available() and args.device == "cuda"
        else t.device("cpu")
    )

    setup_logging(args.verbose)
    set_seed(args.seed)

    # model & data
    model = load_model(args.model, args.device, args.cache_dir)
    model.tokenizer.add_bos_token = False
    save_model_name = args.model.split("/")[-1].lower()

    important_heads = get_model_heads(save_model_name)
    random_heads = get_random_heads(
        model_name=save_model_name,
        num_samples=len(important_heads),
        num_heads=model.cfg.n_heads,
        num_layers=model.cfg.n_layers,
    )
    logging.info("Attention heads to intervene on loaded.")

    # Main loop
    original_accuracies = []
    intervened_accuracies = []
    random_accuracies = []

    for template in TEMPLATES:

        test_loader_z1, test_loader_both = prepare_data(
            model=model,
            model_name=save_model_name,
            template=template,
            args=args,
        )
        logging.info(f"Loaded data for template {template}")

        # Intervene on attention heads
        original_acc, intervened_acc = intervene_on_attention_heads(
            single_perturbation_prompts=test_loader_z1,
            both_perturbation_prompts=test_loader_both,
            model=model,
            head_indices=important_heads,
            alpha=args.alpha,
        )

        # Intervene on random attention heads
        _, random_acc = intervene_on_attention_heads(
            single_perturbation_prompts=test_loader_z1,
            both_perturbation_prompts=test_loader_both,
            model=model,
            head_indices=random_heads,
            alpha=args.alpha,
        )
        logging.info(f"Original accuracy: {original_acc}")
        logging.info(f"Intervened accuracy: {intervened_acc}")
        logging.info(f"Random intervention accuracy: {random_acc}")

        original_accuracies.append(original_acc)
        intervened_accuracies.append(intervened_acc)
        random_accuracies.append(random_acc)

    logging.info(f"Original mean accuracies: {np.mean(original_accuracies)}")
    logging.info(f"Intervened mean accuracies: {np.mean(intervened_accuracies)}")
    logging.info(f"Random intervention mean accuracies: {np.mean(random_accuracies)}")

    os.makedirs(args.save_plot_dir, exist_ok=True)
    uid = f"alpha_{args.alpha}_{args.single_perturbation_data}-to-both_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"
    file_path = f"{args.save_plot_dir}/{save_model_name}_{uid}.png"

    bar_plot_attention_interventions(
        original_accuracies=original_accuracies,
        intervened_accuracies=intervened_accuracies,
        random_accuracies=random_accuracies,
        file_path=file_path,
    )


if __name__ == "__main__":
    main()
