"""
Script computing the Intersection over Union (IoU) and Intersection over Minimum (IoM)
to analyze the overlap of different circuits.
"""

import argparse
import itertools
import logging
from typing import List, Tuple

from llm_error_detection.circuits.circuit_analysis import (
    Circuit,
    intersection_over_minimum,
    intersection_over_union,
    load_circuit_from_file,
)
from llm_error_detection.utils import plot_edge_overlap_heatmap, set_seed, setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line input arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser("Evaluate circuit overlap.")

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
        "--circuit_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to circuits that should be loaded.",
    )
    parser.add_argument(
        "--circuit_labels",
        type=str,
        nargs="+",
        required=True,
        help="Names or labels for circuits.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/edge_overlap",
        help="Output directory.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable plotting.",
    )

    # Overlap config
    parser.add_argument(
        "--token_pos",
        action="store_false",
        help="Consider token positions when computing overlap.",
    )

    parser.set_defaults(plot=True)
    args = parser.parse_args()

    return args


def load_circuits(circuit_paths: List[str]) -> List[Circuit]:
    """
    Load multiple Circuit objects from the specified file paths.

    Args:
        circuit_paths (List[str]): A list of file paths where the Circuit JSON files are located.

    Returns:
        List[Circuit]: A list of Circuit objects loaded from the specified paths.

    Raises:
        AssertionError: If less than two circuit paths are provided or if the number of loaded circuits
                        does not match the number of specified paths.
    """
    assert (
        len(circuit_paths) > 1
    ), f"Must specify more than one circuit to load! {circuit_paths}"
    circuits: List[Circuit] = []

    for circuit_path in circuit_paths:
        circuit = load_circuit_from_file(file_path=circuit_path)
        circuits.append(circuit)

    assert len(circuit_paths) == len(circuits), (
        "Number of loaded circuits is unequal to number of specified circuits!\n"
        f"{circuit_paths}\n{circuits}"
    )

    return circuits


def main() -> None:
    """
    Main script.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    # load circuits
    circuits = load_circuits(args.circuit_paths)

    # compute overlap
    circuit_pairs = list(itertools.combinations_with_replacement(circuits, 2))
    label_pairs = list(itertools.combinations_with_replacement(args.circuit_labels, 2))

    iou_list: List[Tuple[str, str, float]] = []
    iom_list: List[Tuple[str, str, float]] = []

    for label, circuit_pair in zip(label_pairs, circuit_pairs):
        iou = intersection_over_union(
            circuit_list=list(circuit_pair), token_pos=args.token_pos
        )
        iom = intersection_over_minimum(
            circuit_list=list(circuit_pair), token_pos=args.token_pos
        )

        iou_list.append((label[0], label[1], iou))
        iom_list.append((label[0], label[1], iom))

        if label[0] != label[1]:
            logging.info(f"IoU between {label[0]} and {label[1]}: {iou}")
            logging.info(f"IoM between {label[0]} and {label[1]}: {iom}")
            logging.info("===" * 15)

    if args.plot:
        token_pos_spec = "token_pos" if args.token_pos else "no_token_pos"

        plot_edge_overlap_heatmap(
            triplets=iou_list,
            labels=args.circuit_labels,
            file_path=f"{args.output_dir}/{token_pos_spec}/iou_{token_pos_spec}_{'_'.join(args.circuit_labels)}.png",
            title="Intersection over Union",
        )
        plot_edge_overlap_heatmap(
            triplets=iom_list,
            labels=args.circuit_labels,
            file_path=f"{args.output_dir}/{token_pos_spec}/iom_{token_pos_spec}_{'_'.join(args.circuit_labels)}.png",
            title="Intersection over Minimum",
        )


if __name__ == "__main__":
    main()
