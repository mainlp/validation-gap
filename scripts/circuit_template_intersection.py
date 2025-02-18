"""
Compute the intersection of circuits identified via different templates
"""

import argparse
import logging
from typing import List

import torch as t

from llm_error_detection.circuits.circuit_analysis import (
    Circuit,
    circuit_intersection,
    load_circuit_from_file,
)
from llm_error_detection.circuits.visualize.graph_utils import (
    plot_circuits_for_all_positions,
)
from llm_error_detection.utils import get_save_dir_name, set_seed, setup_logging

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

    # Circuit config
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        default="z1",
        choices=["z1", "z2", "both", "shortened_z1"],
        help="Type of perturbation applied to data",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help="Number of samples used to compute edge scores",
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

    args = parser.parse_args()

    return args


def load_circuit(
    save_model_name: str,
    perturbation: str,
    template_name: str,
    grad_function: str,
    answer_function: str,
    train_size: int,
) -> Circuit:
    result_path = get_save_dir_name(
        prefix="results/discovered-circuits", template=template_name
    )
    result_path += f"/{save_model_name}"
    uid = f"{perturbation}_template_{template_name}_gradfunc_{grad_function}_ansfunc_{answer_function}_train_size_{train_size}"
    file_path = f"{result_path}/circuit_{uid}.json"

    return load_circuit_from_file(file_path=file_path)


def main() -> None:
    args = parse_arguments()

    args.device = (
        t.device("cuda")
        if t.cuda.is_available() and args.device == "cuda"
        else t.device("cpu")
    )

    setup_logging(args.verbose)
    set_seed(args.seed)

    # load circuits
    save_model_name = args.model.split("/")[-1].lower()

    circuits: List[Circuit] = []

    for template_name in TEMPLATES:
        circuit = load_circuit(
            save_model_name=save_model_name,
            perturbation=args.perturbation,
            template_name=template_name,
            grad_function=args.grad_function,
            answer_function=args.answer_function,
            train_size=args.train_size,
        )
        logging.info(f"Circuit loaded: \n{circuit}")
        circuits.append(circuit)

    # compute intersection
    intersection_circuit = circuit_intersection(circuits)
    logging.info(f"Circuit intersection: \n{intersection_circuit}")

    # save circuit intersection
    result_path = get_save_dir_name(
        prefix="results/discovered-circuits", template="intersection"
    )
    result_path += f"/{save_model_name}"
    uid = f"{args.perturbation}_template_intersection_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"
    file_path = f"{result_path}/circuit_{uid}.json"

    intersection_circuit.save_to_file(
        file_path=f"{result_path}/circuit_{uid}.json",
    )
    logging.info(f"Circuit intersection saved to {file_path}")

    # plot circuit
    plot_circuits_for_all_positions(
        circuit=intersection_circuit,
        file_path=f"{result_path}/circuit_{uid}.png",
        minimum_penwidth=1.0,
        num_layers=32,
    )


if __name__ == "__main__":
    main()
