"""
Script to evaluate a saved circuit on the arithmetic error detection task.
"""

import argparse
import logging

import torch as t
from auto_circuit.utils.graph_utils import patchable_model

from llm_error_detection.circuits.circuit_analysis import eval_circuit, load_circuit
from llm_error_detection.circuits.circuit_discovery import load_data, load_model
from llm_error_detection.utils import set_seed, setup_logging


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
        "--perturbation",
        type=str,
        default="z1",
        choices=["z1", "z2", "both", "shortened_z1"],
        help="Type of perturbation applied to data",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="0",
        choices=["0", "1", "2", "3", "4", "5", "6", "7", "intersection"],
        help="Template of the math task",
    )
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
    parser.set_defaults(cplot=True)
    args = parser.parse_args()

    return args


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

    train_loader, test_loader, seq_labels = load_data(
        model=model,
        data_dir=args.data_dir,
        save_model_name=save_model_name,
        template=args.template,
        perturbation=args.perturbation,
        num_train=args.train_size,
        num_test=args.test_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        seq_len=test_loader.seq_len,
        separate_qkv=True,
        device=args.device,
    )

    # load circuit
    circuit = load_circuit(
        save_model_name=save_model_name,
        perturbation=args.perturbation,
        template_name=args.template,
        grad_function=args.grad_function,
        answer_function=args.answer_function,
        train_size=args.train_size,
    )
    logging.info("Circuit loaded.")

    # run circuit
    n_edge, metric_value = eval_circuit(
        model=model,
        dataloader=test_loader,
        circuit=circuit,
        metric=args.metric,
    )

    logging.info(f"n_edge: {n_edge}, metric_value: {metric_value}")


if __name__ == "__main__":
    main()
