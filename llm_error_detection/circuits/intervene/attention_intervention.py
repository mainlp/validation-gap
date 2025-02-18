from functools import partial
from typing import List, Tuple

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from llm_error_detection.data_prep.math_filterer import logits_to_valid_pred


def intervene_on_attention_heads(
    single_perturbation_prompts: List[t.Tensor],
    both_perturbation_prompts: List[t.Tensor],
    model: HookedTransformer,
    head_indices: List[Tuple[int, int]],  # List of (layer, head) tuples
    alpha: float = 2.0,
) -> Tuple[float, float]:
    """
    Intervene on the attention heads of a model patching the attention patterns from the single
    perturbation prompts into the both perturbation prompts.

    Args:
        single_perturbation_prompts (List[t.Tensor]): The prompts for the single perturbation.
        both_perturbation_prompts (List[t.Tensor]): The prompts for the both perturbation.
        model (HookedTransformer): The model to intervene on.
        head_indices (List[Tuple[int, int]]): The indices of the heads to intervene on.
        alpha (float, optional): The scaling factor for the intervention. Defaults to 2.0.

    Returns:
        Tuple[float, float]: The accuracies for the original and intervened models on the
            both perturbation prompts.
    """

    original_accuracy: List[int] = []
    intervened_accuracy: List[int] = []

    with t.inference_mode():

        for single, both in tqdm(
            zip(single_perturbation_prompts, both_perturbation_prompts),
            total=len(single_perturbation_prompts),
            desc="Intervening on attention heads",
        ):

            original_logits = model(both, return_type="logits")

            valid_original, _ = logits_to_valid_pred(
                original_logits[:, -1, :],
                model.tokenizer,
                valid_sol=["incorrect", "invalid", "wrong"],
                invalid_sol=["correct", "valid", "right"],
            )
            original_accuracy.extend([1 if i else 0 for i in valid_original])

            _, single_cache = model.run_with_cache(single)

            fwd_hooks = []
            for layer, head in head_indices:
                act_name = f"blocks.{layer}.attn.hook_pattern"
                stored_pattern = single_cache[act_name]
                fwd_hooks.append(
                    (
                        act_name,
                        partial(
                            replace_pattern_hook,
                            stored_pattern=stored_pattern,
                            head_idx=head,
                            alpha=alpha,
                        ),
                    )
                )

            intervened_logits = model.run_with_hooks(
                both, return_type="logits", fwd_hooks=fwd_hooks
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


def replace_pattern_hook(
    value: t.Tensor,
    hook: HookPoint,
    stored_pattern: t.Tensor,
    head_idx: int,
    alpha: float,
) -> t.Tensor:
    """
    Replace the pattern in the attention head with a new pattern scaled by
    a factor of alpha.

    Args:
        value (t.Tensor): The value of the attention head.
        hook (HookPoint): The hook point.
        stored_pattern (t.Tensor): The stored pattern.
        head_idx (int): The index of the head to intervene on.
        alpha (float): The scaling factor for the intervention.

    Returns:
        t.Tensor: The new value of the attention head.
    """
    value[:, head_idx, :, :] = alpha * stored_pattern[:, head_idx, :, :]
    return value
