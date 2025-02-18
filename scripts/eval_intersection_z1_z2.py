"""
Evaluate the intersection and union of the final z1 and z2 circuits on the z1 and z2 tasks.
"""

import argparse
import logging
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch as t
from auto_circuit.utils.graph_utils import patchable_model

from llm_error_detection.circuits.circuit_analysis import (
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
    save_dict_to_json,
    set_seed,
    setup_logging,
)

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
        "--batch_size", type=int, default=2, help="Batch size for data loading"
    )

    # Model & circuit config
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Name of the model to use"
    )
    parser.add_argument(
        "--z1_circuit_overlap",
        type=float,
        required=True,
        help="The overlap of the z1 circuit.",
    )
    parser.add_argument(
        "--z2_circuit_overlap",
        type=float,
        required=True,
        help="The overlap of the z1 circuit.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=5000,
        help="Number of samples used to compute edge scores",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
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

    final_z1_circuit = load_circuit(
        save_model_name=save_model_name,
        perturbation="z1",
        template_name="intersection",
        grad_function=args.grad_function,
        answer_function=args.answer_function,
        train_size=args.train_size,
        intersection_overlap=args.z1_circuit_overlap,
    )
    final_z2_circuit = load_circuit(
        save_model_name=save_model_name,
        perturbation="z2",
        template_name="intersection",
        grad_function=args.grad_function,
        answer_function=args.answer_function,
        train_size=args.train_size,
        intersection_overlap=args.z2_circuit_overlap,
    )
    z1_z2_circuits = [final_z1_circuit, final_z2_circuit]

    intersection_z1_z2 = circuit_soft_intersection(z1_z2_circuits, overlap=1.0)
    union_z1_z2 = circuit_soft_intersection(z1_z2_circuits, overlap=0.0)
    logging.info("Circuits loaded and intersection computed.")

    if args.cplot:
        result_path = get_save_dir_name(
            prefix="results/discovered-circuits", template="final_z1_z2_intersection"
        )
        result_path += f"/{save_model_name}"
        plot_circuits_for_all_positions(
            circuit=intersection_z1_z2,
            file_path=f"{result_path}/intersection_z1_z2.png",
            minimum_penwidth=1.0,
            num_layers=model.cfg.n_layers,
        )
        plot_circuits_for_all_positions(
            circuit=union_z1_z2,
            file_path=f"{result_path}/union_z1_z2.png",
            minimum_penwidth=1.0,
            num_layers=model.cfg.n_layers,
        )

    circuit_faithfulness: Dict[str, Dict[str, Any]] = {"intersection": {}, "union": {}}

    for perturbation in ["z1", "z2"]:
        logging.info(f"Eval on data for {perturbation} error...")
        faithfulness_scores_intersection: List[float] = []
        faithfulness_scores_union: List[float] = []
        n_edges_intersection: List[int] = []
        n_edges_union: List[int] = []

        for template_name in EVAL_TEMPLATES:
            _, test_loader, seq_labels = load_data(
                model=model,
                data_dir=args.data_dir,
                save_model_name=save_model_name,
                template=template_name,
                perturbation=perturbation,  # type: ignore
                num_train=args.train_size,
                num_test=args.test_size,
                batch_size=args.batch_size,
                device=args.device,
            )

            if "llama-3.2" in save_model_name.lower():
                seq_labels = ["[bos]"] + seq_labels

            # align circuit sequence labels
            patched_model = patchable_model(
                deepcopy(model),
                factorized=True,
                slice_output="last_seq",
                seq_len=test_loader.seq_len,
                separate_qkv=True,
                device=args.device,
            )

            aligned_intersection_circuit = update_edge_patch_idx(
                circuit=intersection_z1_z2, target_seq_labels=seq_labels
            )

            aligned_union_circuit = update_edge_patch_idx(
                circuit=union_z1_z2, target_seq_labels=seq_labels
            )

            # run circuit
            n_edge_intersection, metric_value_intersection = eval_circuit(
                model=patched_model,
                dataloader=test_loader,
                circuit=aligned_intersection_circuit,
                metric=args.metric,
            )
            faithfulness_scores_intersection.append(metric_value_intersection)
            n_edges_intersection.append(n_edge_intersection)

            n_edge_union, metric_value_union = eval_circuit(
                model=patched_model,
                dataloader=test_loader,
                circuit=aligned_union_circuit,
                metric=args.metric,
            )
            faithfulness_scores_union.append(metric_value_union)
            n_edges_union.append(n_edge_union)

            logging.info(
                f"template: {template_name}\nfaithfulness union: {metric_value_union}\nfaithfulness intersection: {metric_value_intersection}"
            )
            logging.info(
                f"Num edges union: {n_edge_union}\nNum edges intersection: {n_edge_intersection}"
            )

        avg_faithfulness_intersection = np.mean(faithfulness_scores_intersection)
        std_faithfulness_intersection = np.std(faithfulness_scores_intersection)
        avg_n_edges_intersection = np.mean(n_edges_intersection)
        std_n_edges_intersection = np.std(n_edges_intersection)
        logging.info(
            f"Intersection: \nfaithfulness avg: {avg_faithfulness_intersection} +/- {std_faithfulness_intersection}"
        )
        logging.info(
            f"Num edges avg: {avg_n_edges_intersection} +/- {std_n_edges_intersection}"
        )

        avg_faithfulness_union = np.mean(faithfulness_scores_union)
        std_faithfulness_union = np.std(faithfulness_scores_union)
        avg_n_edges_union = np.mean(n_edge_union)
        std_n_edges_union = np.std(n_edge_union)
        logging.info(
            f"Union: \nfaithfulness avg: {avg_faithfulness_union} +/- {std_faithfulness_union}"
        )
        logging.info(f"Num edges avg: {avg_n_edges_union} +/- {std_n_edges_union}")

        circuit_faithfulness["intersection"][perturbation] = {
            "faithfulness": (
                avg_faithfulness_intersection,
                std_faithfulness_intersection,
            ),
            "faithfulness_all": faithfulness_scores_intersection,
            "n_edges": (avg_n_edges_intersection, std_n_edges_intersection),
            "n_edges_all": n_edges_intersection,
        }
        circuit_faithfulness["union"][perturbation] = {
            "faithfulness": (avg_faithfulness_union, std_faithfulness_union),
            "faithfulness_all": faithfulness_scores_union,
            "n_edges": (avg_n_edges_union, std_n_edges_union),
            "n_edges_all": n_edge_union,
        }

    # save stats to file
    file_name = f"final_z1_z2_intersection_faithfulness_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"
    save_dict_to_json(circuit_faithfulness, f"{result_path}/{file_name}.json")


if __name__ == "__main__":
    main()
