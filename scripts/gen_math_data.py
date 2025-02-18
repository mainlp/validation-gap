"""
Script to generate data for arithmetic error detection. Clean inputs are mathematical reasoning traces that contain an arithmetic error.
Corrupted prompts are respective reasoning traces without the error present.
"""

import argparse

from llm_error_detection.data_prep.math_generator import (
    filter_variables,
    gen_samples,
    load_tokenizers,
    randomly_sample_variables,
    samples_to_chat_format,
    subsample_prompts,
)
from llm_error_detection.utils import save_dict_to_json, set_seed, setup_logging

MODELS = models = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]


def parse_arguments():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument(
        "--seed", type=int, default=5, help="Random seed for reproducibility"
    )
    parser.add_argument("--cache_dir", type=str, help="Directory of cached tokenizers")
    parser.add_argument(
        "--data_dir", type=str, default="./data/math", help="Directory to store data"
    )

    # Data configs
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help=(
            "Number of total samples to generate. "
            "If bigger than the total number of samples, all samples will be generated."
        ),
    )
    parser.add_argument(
        "--subsamples_parameters",
        type=int,
        default=10,
        help=(
            "Number of samples to subsample from the list of names, objects, and verbs. "
            "If bigger than the total number of samples, all samples will be generated."
        ),
    )
    parser.add_argument(
        "--template",
        type=str,
        default="full",
        choices=["full", "0", "1", "2", "3", "4", "5", "6", "7"],
        help="Template to use for generating prompts",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main script.
    """
    # Parse arguments
    args = parse_arguments()

    # Fix random seed for reproducibility
    setup_logging(args.verbose)
    set_seed(args.seed)

    # Load tokenizers
    tokenizers = load_tokenizers(MODELS, args.cache_dir)

    # Filter variables based on token length and subsample them
    filtered_variables = filter_variables(tokenizers, args.template)
    filtered_variables = randomly_sample_variables(
        filtered_variables, args.subsamples_parameters
    )

    (
        correct_list,
        z1_error_list,
        z2_error_list,
        both_error_list,
        shortened_z1_error_list,
    ) = gen_samples(
        variables=filtered_variables,
        current_template=args.template,
    )

    (
        correct_list,
        z1_error_list,
        z2_error_list,
        both_error_list,
        shortened_z1_error_list,
    ) = subsample_prompts(
        correct_list=correct_list,
        z1_error_list=z1_error_list,
        z2_error_list=z2_error_list,
        both_error_list=both_error_list,
        shortened_z1_error_list=shortened_z1_error_list,
        num_samples=args.samples,
    )

    # Convert to chat format
    (
        prompt_z1,
        prompt_z2,
        prompt_both,
        prompt_shortened_z1,
        prompt_computation_one_digit,
        prompt_computation_two_digits,
    ) = samples_to_chat_format(
        correct_list=correct_list,
        z1_error_list=z1_error_list,
        z2_error_list=z2_error_list,
        both_error_list=both_error_list,
        shortened_z1_error_list=shortened_z1_error_list,
    )

    for data, name in zip(
        [
            prompt_z1,
            prompt_z2,
            prompt_both,
            prompt_shortened_z1,
            prompt_computation_one_digit,
            prompt_computation_two_digits,
        ],
        [
            "z1",
            "z2",
            "both",
            "shortened_z1",
            "computation_one_digit",
            "computation_two_digits",
        ],
    ):
        data_json = {"prompts": data}

        template_save_name = (
            f"template_{args.template}" if args.template != "full" else args.template
        )
        data_dir = f"{args.data_dir}/{template_save_name}"
        file_path = f"{data_dir}/math_prompts_{name}.json"
        save_dict_to_json(data=data_json, file_path=file_path)


if __name__ == "__main__":
    main()
