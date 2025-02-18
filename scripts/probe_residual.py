"""
Script to probe the residual stream on computing the result of a computation.
"""

import argparse
import logging
import os
from typing import List, Tuple

import torch as t
from transformer_lens import HookedTransformer

from llm_error_detection.circuits.circuit_analysis import load_circuit
from llm_error_detection.circuits.circuit_discovery import load_data, load_model
from llm_error_detection.circuits.probe.probing import probe_residual_components
from llm_error_detection.utils import set_seed, setup_logging

TEMPLATES = ["0", "1", "2", "3", "4", "5", "6", "7"]


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
        "--train_size_per_template",
        type=int,
        default=500,
        help="Number of samples used to compute edge scores",
    )
    parser.add_argument(
        "--test_size_per_template",
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
        "--intersection_overlap_z1",
        type=str,
        default="1.0",
        choices=["1.0", "0.875", "0.75", "0.625", "0.5", "0.375", "0.25", "0.125"],
        help="Overlapping templates parameter in the intersection circuit for z1",
    )
    parser.add_argument(
        "--intersection_overlap_z2",
        type=str,
        default="1.0",
        choices=["1.0", "0.875", "0.75", "0.625", "0.5", "0.375", "0.25", "0.125"],
        help="Overlapping templates parameter in the intersection circuit for z2",
    )
    args = parser.parse_args()

    return args


def get_circuit_tokens(save_model_name: str, args: argparse.Namespace) -> List[str]:
    """
    Get the tokens positions relevant for the error identification
    circuit for prompts with z1 and z2 positions perturbed.

    Args:
        save_model_name (str): Name of the model to save the circuit for.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[str]: List of tokens in the circuit.
    """
    # load circuits
    circuits = []
    for perturbation in ["z1", "z2"]:
        intersection_overlap = (
            args.intersection_overlap_z1
            if perturbation == "z1"
            else args.intersection_overlap_z2
        )
        circuit = load_circuit(
            save_model_name=save_model_name,
            perturbation=perturbation,
            template_name="intersection",
            grad_function=args.grad_function,
            answer_function=args.answer_function,
            train_size=5000,
            intersection_overlap=intersection_overlap,
        )
        circuits.append(circuit)
    logging.info("Loaded circuits for z1 and z2 perturbations.")

    tokens = set(circuits[0].tok_pos_edges.keys())
    for circuit in circuits[1:]:
        tokens.intersection_update(circuit.tok_pos_edges.keys())

    return list(tokens)


def prepare_data(
    model: HookedTransformer,
    model_name: str,
    train_size_per_template: int,
    test_size_per_template: int,
    data_dir: str,
    batch_size: int,
    device: t.device,
) -> Tuple[List[Tuple[t.Tensor, List[str]]], List[Tuple[t.Tensor, List[str]]]]:
    """
    Prepare the data for probing.

    Args:
        model (HookedTransformer): The model to use for probing.
        model_name (str): Name of the model.
        train_size_per_template (int): Size of the training set.
        test_size_per_template (int): Size of the test set.
        data_dir (str): Directory containing the data.
        batch_size (int): Batch size for data loading.
        device (t.device): Device to use for computation.

    Returns:
        Tuple[List[Tuple[t.Tensor, List[str]]], List[Tuple[t.Tensor, List[str]]]]: Batched training and test data with sequence labels wrt template.
    """
    train_batches_with_seq_labels: List[Tuple[t.Tensor, List[str]]] = []
    test_batches_with_seq_labels: List[Tuple[t.Tensor, List[str]]] = []

    for template in TEMPLATES:

        train_loader, test_loader, seq_labels = load_data(
            model=model,
            data_dir=data_dir,
            save_model_name=model_name,
            template=template,
            perturbation="both",
            num_train=train_size_per_template,
            num_test=test_size_per_template,
            batch_size=batch_size,
            device=device,
            filtered=False,
        )
        clean_train_batches = [batch.clean for batch in train_loader]
        clean_test_batches = [batch.clean for batch in test_loader]

        if "llama-3.2" in model_name.lower():
            seq_labels = ["[bos]"] + seq_labels

        train_batches_with_seq_labels.append((clean_train_batches, seq_labels))
        test_batches_with_seq_labels.append((clean_test_batches, seq_labels))

    return train_batches_with_seq_labels, test_batches_with_seq_labels


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

    circuit_tokens = get_circuit_tokens(save_model_name, args)

    train_batches_with_seq_labels, test_batches_with_seq_labels = prepare_data(
        model=model,
        model_name=save_model_name,
        train_size_per_template=args.train_size_per_template,
        test_size_per_template=args.test_size_per_template,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    logging.info("Data prepared for probing.")

    # probe residual components
    save_dir = "results/probing"
    os.makedirs(save_dir, exist_ok=True)

    probe_residual_components(
        model_name=save_model_name,
        model=model,
        train_batches_with_seq_labels=train_batches_with_seq_labels,
        test_batches_with_seq_labels=test_batches_with_seq_labels,
        circuit_tokens=circuit_tokens,
        save_dir=save_dir,
    )
    logging.info("Probing results saved.")


if __name__ == "__main__":
    main()
