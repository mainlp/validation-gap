"""
Evaluate models on all templates and get their average accuracy on the error detection task.
"""

import argparse
import logging
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Tuple, Union

import numpy as np
import torch as t
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from llm_error_detection.circuits.circuit_discovery import load_data
from llm_error_detection.data_prep.math_filterer import (
    get_prompt_strs,
    load_math_data,
    logits_to_number_pred,
    logits_to_valid_pred,
    prep_math_data,
)
from llm_error_detection.utils import save_results, set_seed, setup_logging

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
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )

    # Data configs
    parser.add_argument(
        "--batch_size", type=int, default="32", help="Batch size for data loading"
    )
    parser.add_argument(
        "--samples_per_template",
        type=int,
        default="1000",
        help="Number of samples per template",
    )

    # Model configs
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Name of the model to use for inference. The model must be available in the Transformer Lens. If the model is not available in the Transformer Lens, it will be loaded from Hugging Face.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type to use for model weights",
    )
    args = parser.parse_args()

    return args


def load_baseline_model(
    model_name: str, device: t.device, cache_dir: str, dtype: str
) -> Tuple[
    Union[AutoModelForCausalLM, HookedTransformer],
    Union[PreTrainedTokenizer, AutoTokenizer],
]:
    """
    Load a baseline model and its tokenizer, either from the Transformer Lens or Hugging Face.

    Attempts to load the model using the HookedTransformer from the Transformer Lens.
    If the model is not found, it falls back to loading from Hugging Face's AutoModelForCausalLM.
    Correspondingly, either the HookedTransformer tokenizer or AutoTokenizer is loaded.

    Args:
        model_name (str): The name of the model to load.
        device (t.device): The device to use for model loading.
        cache_dir (str): Directory where the model weights are cached.
        dtype (str): Data type for the model weights.

    Returns:
        Tuple: A tuple containing the loaded model (either HookedTransformer or AutoModelForCausalLM)
               and its tokenizer (either PreTrainedTokenizer or AutoTokenizer).
    """
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            cache_dir=cache_dir,
            dtype=dtype,
        )
        tokenizer = model.tokenizer

    except ValueError:
        logging.warning(
            f"Could not find the model: {model_name} in Tranformer Lens. Loading from HF instead."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    return model, tokenizer


def correct_vs_error_evaluation(
    model_name: str,
    model: Union[HookedTransformer, AutoModelForCausalLM],
    tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
    clean_batches: Tuple[Tensor, ...],
    corrupted_batches: Tuple[Tensor, ...],
    perturbation: str,
) -> Tuple[float, float, float]:
    """
    Evaluate a model on a set of clean and corrupted prompts.

    Args:
        model_name (str): The name of the model.
        model (Union[HookedTransformer, AutoModelForCausalLM]): The model to evaluate.
        tokenizer (Union[PreTrainedTokenizer, AutoTokenizer]): The tokenizer used to
            decode the predictions.
        clean_batches (Tuple[Tensor, ...]): A tuple of tensors, each containing a batch
            of clean prompts.
        corrupted_batches (Tuple[Tensor, ...]): A tuple of tensors, each containing a batch
            of corrupted prompts.
        perturbation (str): A string describing the type of perturbation used to corrupt
            the prompts.

    Returns:
        Tuple[float, float, float]: A tuple of three floats, representing the accuracy of
            the model on the clean prompts, the accuracy of the model on the corrupted
            prompts, and the accuracy of the model on both the clean and corrupted prompts.
    """

    tokenization_type = number_tokenization_type(model_name)

    with t.inference_mode():
        valid_clean_and_corrupted: List[int] = []
        clean_pred_accuracy: List[int] = []
        corrputed_pred_accuracy: List[int] = []

        for clean_batch, corr_batch in tqdm(
            zip(clean_batches, corrupted_batches),
            total=len(clean_batches),
            desc=f"Evaluating on permutation {perturbation}",
        ):

            # Clean batch
            if type(model) == HookedTransformer:
                clean_logits = model(clean_batch)[:, -1, :]
            else:
                clean_logits = model(clean_batch)[0][:, -1, :]

            if "computation" in perturbation:
                valid_clean_preds, _ = logits_to_number_pred(
                    logits=clean_logits,
                    tokenizer=tokenizer,
                    batch=clean_batch,
                    tokenization_type=tokenization_type,
                )
            else:
                valid_clean_preds, _ = logits_to_valid_pred(
                    logits=clean_logits,
                    tokenizer=tokenizer,
                    valid_sol=["invalid", "wrong", "incorrect"],
                    invalid_sol=["valid", "right", "correct"],
                )
            clean_pred_accuracy.extend([1 if i else 0 for i in valid_clean_preds])

            # Corrupt batch
            if type(model) == HookedTransformer:
                corrupted_logits = model(corr_batch)[:, -1, :]
            else:
                corrupted_logits = model(corr_batch)[0][:, -1, :]

            if "computation" in perturbation:
                valid_corrupted_preds, _ = logits_to_number_pred(
                    logits=corrupted_logits,
                    tokenizer=tokenizer,
                    batch=corr_batch,
                    tokenization_type=tokenization_type,
                )
            else:
                valid_corrupted_preds, _ = logits_to_valid_pred(
                    logits=corrupted_logits,
                    tokenizer=tokenizer,
                    valid_sol=["valid", "right", "correct"],
                    invalid_sol=["invalid", "wrong", "incorrect"],
                )
            corrputed_pred_accuracy.extend(
                [1 if i else 0 for i in valid_corrupted_preds]
            )

            # Check if both are correct
            both_valid = t.logical_and(valid_clean_preds, valid_corrupted_preds)
            valid_clean_and_corrupted.extend(
                [1 if i else 0 for i in both_valid.tolist()]
            )

    # Compute the accuracies
    error_acc = sum(clean_pred_accuracy) / len(clean_pred_accuracy)
    no_error_acc = sum(corrputed_pred_accuracy) / len(corrputed_pred_accuracy)
    error_correct_acc = sum(valid_clean_and_corrupted) / len(valid_clean_and_corrupted)

    return error_acc, no_error_acc, error_correct_acc


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


def number_tokenization_type(model_name: str) -> str:
    """
    Get the type of tokenization used by the model to process numbers.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The type of tokenization used by the model.
    """
    if "llama" in model_name:
        return "two_digits"
    else:
        return "one_digit"


def list_to_string(lst: List[Any]) -> str:
    """
    Convert a list to a string.

    Args:
        lst (List): The list to convert.

    Returns:
        str: The converted string.
    """
    str_lst = f"{lst}".replace(", ", "; ")
    return str_lst


def prepare_data(
    model: HookedTransformer,
    template: str,
    perturbation: str,
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[t.Tensor, t.Tensor]:

    if perturbation != "both":
        file_path = (
            f"{args.data_dir}/template_{template}/math_prompts_{perturbation}.json"
        )
        math_data = load_math_data(file_path)
        math_data = subsample_math_data(math_data, args.samples_per_template)

        clean_prompts_strs, corrupt_prompts_strs, _, _ = get_prompt_strs(
            math_data=math_data, model_name=args.model, tokenizer=tokenizer
        )

        clean_batches, corrupted_batches = prep_math_data(
            error_prompts_strs=clean_prompts_strs,
            correct_prompts_strs=corrupt_prompts_strs,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        model_name = args.model.split("/")[-1].lower()
        promptloader_single, _, seq_labels = load_data(
            model=model,
            data_dir="data/math",
            save_model_name=model_name,
            template=template,
            perturbation="z1",
            num_train=args.samples_per_template,
            num_test=args.samples_per_template,
            batch_size=1,
            device=args.device,
        )

        if "llama-3.2" in args.model.lower():
            seq_labels = ["[bos]"] + seq_labels

        clean_batches: List[t.Tensor] = []
        corrupted_batches: List[t.Tensor] = []

        indexes_z1_z2 = [
            (seq_labels.index(z1), seq_labels.index(z2))
            for z1, z2 in Z1_AND_Z2_LABELS
            if z1 in seq_labels and z2 in seq_labels
        ]

        for batch in promptloader_single:
            clean, corrupt = batch.clean, batch.corrupt
            new_clean = deepcopy(clean)
            new_clean[:, indexes_z1_z2[0][1]] = clean[:, indexes_z1_z2[0][0]]
            clean_batches.append(new_clean)
            corrupted_batches.append(corrupt)

        clean_batches = t.cat(clean_batches)
        corrupted_batches = t.cat(corrupted_batches)

        clean_batches = t.split(clean_batches, args.batch_size)
        corrupted_batches = t.split(corrupted_batches, args.batch_size)

    return clean_batches, corrupted_batches


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

    # model & tokenizer
    model, tokenizer = load_baseline_model(
        model_name=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
    )
    tokenizer.add_bos_token = False  # Do not add BOS token
    tokenizer.padding_side = "left"

    num_tokenization = number_tokenization_type(args.model)

    for perturbation in ["z1", "z2", "both", f"computation_{num_tokenization}"]:

        perturbation_results = defaultdict(list)
        for template in ["0", "1", "2", "3", "4", "5", "6", "7"]:

            # load data
            clean_batches, corrupted_batches = prepare_data(
                model=model,
                template=template,
                perturbation=perturbation,
                args=args,
                tokenizer=tokenizer,
            )

            # run inference
            error_acc, correct_acc, error_correct_acc = correct_vs_error_evaluation(
                model_name=args.model,
                model=model,
                tokenizer=tokenizer,
                clean_batches=clean_batches,
                corrupted_batches=corrupted_batches,
                perturbation=perturbation,
            )

            logging.info(f"Perturbation: {perturbation}, Template: {template}")
            logging.info(f"Accuracy on prompts with an error: {error_acc}")
            logging.info(f"Accuracy on prompts without an error: {correct_acc}")
            logging.info(f"Accuracy on both error & no-error: {error_correct_acc}")

            for key, accuracy in zip(
                ["error", "correct", "error&correct"],
                [error_acc, correct_acc, error_correct_acc],
            ):
                perturbation_results[key].append(accuracy)

        save_model_name = args.model.split("/")[-1].lower()
        result_dict: dict[str, Any] = {
            "model_name": save_model_name,
            "perturbation": perturbation,
            "error_mean": np.mean(perturbation_results["error"]),
            "correct_mean": np.mean(perturbation_results["correct"]),
            "error&correct_mean": np.mean(perturbation_results["error&correct"]),
            "error_std": np.std(perturbation_results["error"]),
            "correct_std": np.std(perturbation_results["correct"]),
            "error&correct_std": np.std(perturbation_results["error&correct"]),
            "error_full": list_to_string(perturbation_results["error"]),
            "correct_full": list_to_string(perturbation_results["correct"]),
            "error&correct_full": list_to_string(perturbation_results["error&correct"]),
        }
        save_results(
            file_path="results/baseline_task_accuracy/accuracies.csv",
            result_dict=result_dict,
        )


if __name__ == "__main__":
    main()
