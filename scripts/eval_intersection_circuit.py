"""
Script to evaluate the soft intersection circuit across different overlap fractions
"""

import argparse
import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch as t
from auto_circuit.utils.graph_utils import patchable_model

from llm_error_detection.circuits.circuit_analysis import (
    Circuit,
    circuit_soft_intersection,
    eval_circuit,
    load_circuit,
    update_edge_patch_idx,
)
from llm_error_detection.circuits.circuit_discovery import load_data, load_model
from llm_error_detection.circuits.visualize.graph_utils import (
    plot_circuits_for_all_positions,
)
from llm_error_detection.utils import (
    get_save_dir_name,
    plot_dual_y_axis,
    save_dict_to_json,
    set_seed,
    setup_logging,
)

OVELRAP = [1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 8 / 8]
CIRCUIT_TEMPLATES = ["0", "1", "2", "3", "4", "5", "6", "7"]
EVAL_TEMPLATES = ["0", "1", "2", "3", "4", "5", "6", "7"]


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
        choices=["z1", "z2", "both", "shortened_z1", "computation"],
        help="Type of perturbation applied to data",
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
    parser.add_argument(
        "--no-cplot",
        dest="cplot",
        action="store_false",
        help="Disable circuit plotting.",
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

    # model
    model = load_model(args.model, args.device, args.cache_dir)
    model.tokenizer.add_bos_token = False

    # load circuits
    save_model_name = args.model.split("/")[-1].lower()
    circuits: List[Circuit] = []

    for template_name in CIRCUIT_TEMPLATES:
        circuit = load_circuit(
            save_model_name=save_model_name,
            perturbation=args.perturbation,
            template_name=template_name,
            grad_function=args.grad_function,
            answer_function=args.answer_function,
            train_size=args.train_size,
        )
        logging.info(f"Circuit for template {template_name} loaded.")
        circuits.append(circuit)

    circuit_faithfulness: Dict[str, Dict[str, Union[Tuple, List]]] = {}

    for overlap_val in OVELRAP:
        # compute soft intersection
        intersection_circuit = circuit_soft_intersection(circuits, overlap=overlap_val)
        logging.info(f"Circuit intersection for overlap {overlap_val} computed.")

        if args.cplot:
            # save circuit intersection
            result_path = get_save_dir_name(
                prefix="results/discovered-circuits", template="intersection"
            )
            result_path += f"/{save_model_name}"
            uid = f"{args.perturbation}_template_intersection_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}_overlap_{overlap_val}"
            intersection_circuit.save_to_file(
                file_path=f"{result_path}/circuit_{uid}.json",
            )

            # plot circuit
            plot_circuits_for_all_positions(
                circuit=intersection_circuit,
                file_path=f"{result_path}/circuit_{uid}.png",
                minimum_penwidth=1.0,
                num_layers=32,
            )

        # eval intersection of templates
        faithfulness_scores: List[float] = []
        n_edges: List[int] = []

        for template_name in EVAL_TEMPLATES:
            _, test_loader, seq_labels = load_data(
                model=model,
                data_dir=args.data_dir,
                save_model_name=save_model_name,
                template=template_name,
                perturbation=args.perturbation,
                num_train=args.train_size,
                num_test=args.test_size,
                batch_size=args.batch_size,
                device=args.device,
            )

            if "llama-3.2" in save_model_name.lower():
                seq_labels = ["[bos]"] + seq_labels

            # align circuit sequence labels
            aligned_circuit = update_edge_patch_idx(
                circuit=intersection_circuit, target_seq_labels=seq_labels
            )

            patched_model = patchable_model(
                deepcopy(model),
                factorized=True,
                slice_output="last_seq",
                seq_len=test_loader.seq_len,
                separate_qkv=True,
                device=args.device,
            )

            # run circuit
            n_edge, metric_value = eval_circuit(
                model=patched_model,
                dataloader=test_loader,
                circuit=aligned_circuit,
                metric=args.metric,
            )
            faithfulness_scores.append(metric_value)
            n_edges.append(n_edge)
            logging.info(
                f"overlap: {overlap_val}, template: {template_name}, faithfulness: {metric_value}"
            )

        avg_faithfulness = np.mean(faithfulness_scores)
        std_faithfulness = np.std(faithfulness_scores)
        avg_n_edges = np.mean(n_edges)
        std_n_edges = np.std(n_edges)

        circuit_faithfulness[f"{overlap_val}"] = {
            "faithfulness": (avg_faithfulness, std_faithfulness),
            "faithfulness_all": faithfulness_scores,
            "n_edges": (avg_n_edges, std_n_edges),
            "n_edges_all": n_edges,
        }

        logging.info(
            f"overlap: {overlap_val}, avg: {avg_faithfulness} +/- {std_faithfulness}"
        )

    file_name = f"intersection_faithfulness_{args.perturbation}_template_intersection_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"
    save_dict_to_json(circuit_faithfulness, f"{result_path}/{file_name}.json")
    plot_dual_y_axis(
        data=circuit_faithfulness,
        file_path=f"{result_path}/{file_name}.png",
        x_axis_label="overlap",
        x_axis_ticks=["1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"],
        title=save_model_name,
    )


if __name__ == "__main__":
    main()
