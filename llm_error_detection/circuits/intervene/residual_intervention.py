from functools import partial
from typing import List, Tuple

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from llm_error_detection.circuits.interpret.query_key_analysis import (
    has_one_digit_tokenizer,
)
from llm_error_detection.data_prep.math_filterer import logits_to_valid_pred

SRC_TO_DEST_LAYERS = {
    "meta-llama/Llama-3.2-3B-Instruct": (16, 2),
    "microsoft/Phi-3-mini-4k-instruct": (24, 1),
    "Qwen/Qwen2.5-Math-1.5B-Instruct": (22, 1),
    "Qwen/Qwen2.5-1.5B-Instruct": (22, 1),
}


def intervene_on_residual_stream(
    model: HookedTransformer,
    model_name: str,
    perturbation_prompts: List[t.Tensor],
    seq_labels: List[str],
    alpha: float = 2.0,
) -> Tuple[float, float]:
    """
    Intervene on the residual stream of a model adding the activations from a later layer
    to the one of an earlier layer.

    Args:
        model (HookedTransformer): The model to intervene on.
        model_name (str): The name of the model.
        perturbation_prompts (List[t.Tensor]): The perturbed prompts.
        seq_labels (List[str]): The labels for the sequence.
        alpha (float, optional): The scaling factor for the intervention. Defaults to 2.0.

    Returns:
        Tuple[float, float]: The accuracies for the original and intervened models on the
            perturbed prompts.
    """

    original_accuracy: List[int] = []
    intervened_accuracy: List[int] = []

    src_pos, dest_pos = get_src_dest_pos(model, seq_labels)

    with t.inference_mode():

        for prompts in tqdm(
            perturbation_prompts,
            total=len(perturbation_prompts),
            desc="Intervening on residual stream",
        ):

            original_logits = model(prompts, return_type="logits")

            valid_original, _ = logits_to_valid_pred(
                original_logits[:, -1, :],
                model.tokenizer,
                valid_sol=["incorrect", "invalid", "wrong"],
                invalid_sol=["correct", "valid", "right"],
            )
            original_accuracy.extend([1 if i else 0 for i in valid_original])

            _, cache = model.run_with_cache(prompts)

            src, dest = SRC_TO_DEST_LAYERS[model_name]
            src_act_name = f"blocks.{src}.hook_resid_post"
            dest_act_name = f"blocks.{dest}.hook_resid_post"
            stored_act = cache[src_act_name]
            fwd_hooks = [
                (
                    dest_act_name,
                    partial(
                        replace_residual_hook,
                        stored_act=stored_act,
                        src_pos=src_pos,
                        dest_pos=dest_pos,
                        alpha=alpha,
                    ),
                )
            ]

            intervened_logits = model.run_with_hooks(
                prompts, return_type="logits", fwd_hooks=fwd_hooks
            )

            valid_intervened, _ = logits_to_valid_pred(
                intervened_logits[:, -1, :],
                model.tokenizer,
                valid_sol=["incorrect", "invalid", "wrong"],
                invalid_sol=["correct", "valid", "right"],
            )
            intervened_accuracy.extend([1 if i else 0 for i in valid_intervened])

        original_acc = sum(original_accuracy) / len(original_accuracy)
        intervened_acc = sum(intervened_accuracy) / len(intervened_accuracy)

    return (original_acc, intervened_acc)


def get_src_dest_pos(
    model: HookedTransformer, seq_labels: List[str]
) -> Tuple[int, int]:
    """
    Get the source and destination positions for the intervention.

    Args:
        model (HookedTransformer): The model to intervene on.
        seq_labels (List[str]): The labels for the sequence.

    Returns:
        Tuple[int, int]: The source and destination positions.
    """
    src_label = (
        "[z1-first]_occ_1"
        if has_one_digit_tokenizer(model)
        else "[space_after_eq]_occ_1"
    )
    dest_label = "[z1-second]_occ_1" if has_one_digit_tokenizer(model) else "[z2]_occ_1"

    src_pos = seq_labels.index(src_label)
    dest_pos = seq_labels.index(dest_label)

    return (src_pos, dest_pos)


def replace_residual_hook(
    value: t.Tensor,
    hook: HookPoint,
    stored_act: t.Tensor,
    src_pos: int,
    dest_pos: int,
    alpha: float,
) -> t.Tensor:
    """
    Add the stored pattern at src_pos to the value of the residual activation
    at dest_pos scaled it by a factor of alpha.

    Args:
        value (t.Tensor): The value of the attention head.
        hook (HookPoint): The hook point.
        stored_act (t.Tensor): The stored residual activation.
        src_pos (int): The source position.
        dest_pos (int): The destination position.
        alpha (float): The scaling factor for the intervention.

    Returns:
        t.Tensor: The new value of the attention head.
    """
    value[:, dest_pos, :] = value[:, dest_pos, :] + alpha * stored_act[:, src_pos, :]
    return value
