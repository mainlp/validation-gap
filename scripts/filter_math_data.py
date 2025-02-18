"""
Script to filter math data according to samples where models succeeds or fails to detect the arithmetic error.
"""

import argparse

import torch as t
from transformer_lens import HookedTransformer

from llm_error_detection.data_prep.math_filterer import filter_math_data
from llm_error_detection.utils import set_seed, setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line input arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # General configs
    parser.add_argument(
        "--cache_dir", type=str, help="Directory of cached model weights"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/math",
        help="Directory to load the data from",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )

    # Data configs
    parser.add_argument(
        "--template",
        type=str,
        default="full",
        choices=["full", "0", "1", "2", "3", "4", "5", "6", "7"],
        help="Type of template to use",
    )

    # Model configs
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        choices=[
            "microsoft/Phi-3-mini-4k-instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
        ],
        help="Name of the model to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default="32", help="Batch size for data loading"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type to use for model weights",
    )
    args = parser.parse_args()
    return args


def main():
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

    template = "full" if args.template == "full" else f"template_{args.template}"

    model = HookedTransformer.from_pretrained(
        args.model,
        device=args.device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
    )
    tokenizer = model.tokenizer
    tokenizer.add_bos_token = False  # Do not add BOS token
    tokenizer.padding_side = "left"

    for perturbation, check_answer in [
        ("z1", True),
        ("z2", True),
        ("both", True),
        ("shortened_z1", True),
        ("computation", True),
        ("both", False),
    ]:
        filter_math_data(
            data_dir=args.data_dir,
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            perturbation=perturbation,
            template=template,
            device=args.device,
            batch_size=args.batch_size,
            check_answers=check_answer,
        )


if __name__ == "__main__":
    main()
