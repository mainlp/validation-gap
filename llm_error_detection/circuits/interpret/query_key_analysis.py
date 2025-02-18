"""
Functions and modules analyzing the QK matrix patterns of a given model w.r.t. number tokens.
"""

import os
from typing import Dict, List, Literal, Tuple

import torch as t
from matplotlib import pyplot as plt
from transformer_lens import HookedTransformer

from llm_error_detection.circuits.circuit_analysis import Circuit
from llm_error_detection.circuits.interpret.attention_analysis import (
    get_attention_heads,
    get_union_heads,
    parse_attention_edge,
)
from llm_error_detection.utils import save_dict_to_json

RELEVANT_TOKENS = {
    "1_digit": [str(i) for i in range(10)],
    "multi_digit": [str(i) for i in range(20)],
}


def visualize_qk_patterns(
    model: HookedTransformer,
    circuits: Dict[str, Circuit],
    save_dir: str,
    uid: str,
) -> None:
    """
    Analyze the QK scores for attention heads in the circuit on specific numeric tokens
    looking for consistency or inconsistency patterns.

    Args:
        model (PatchableModel): The model to analyze.
        circuits (Dict[str, Circuit]): A dict with circuits associated to different types of perturbations.
        save_dir (str): The path to save the attention patterns.
        uid (str): Unique identifier for the analysis.
    """
    # Get shared attention head information
    attn_info_list = []
    for circuit in circuits.values():

        attn_heads = get_attention_heads(circuit)

        attn_info = [
            parse_attention_edge(head, edge_type, lbl)
            for head, edge_type, lbl in attn_heads
        ]

        attn_info_list.append(attn_info)

    attn_info, origins = get_union_heads(attn_info_list)

    tokenization_type = "1_digit" if has_one_digit_tokenizer(model) else "multi_digit"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{uid}"

    functional_heads: Dict[str, List[Tuple[int, int, float]]] = {
        "consistency": [],
        "inconsistency": [],
    }

    for (_, _, layer, head), origin in zip(attn_info, origins):
        QK = get_qk_matrix(model, layer, head)
        W_E = filter_embedding_matrix(model, model.W_E, "embedding")
        W_U = filter_embedding_matrix(model, model.W_U, "unembedding")
        QK_patterns = t.matmul(t.matmul(W_E, QK), W_U)

        phi_scores = []
        masks = []
        for consistency in [True, False]:
            if consistency:
                maps_phi_score, highlight_mask = maps_phi_consistency_score(
                    QK_patterns,
                    topk=1,
                    consistency=consistency,
                )
            else:
                maps_phi_score, highlight_mask = maps_phi_consistency_score(
                    QK_patterns,
                    topk=len(RELEVANT_TOKENS[tokenization_type]) - 1,
                    consistency=consistency,
                )
            phi_scores.append(maps_phi_score)
            masks.append(highlight_mask)

            if maps_phi_score >= 0.8:
                key = "consistency" if consistency else "inconsistency"
                head_tuple = (layer, head, maps_phi_score)
                if head_tuple not in functional_heads[key]:
                    functional_heads[key].append(head_tuple)

        plot_qk_patterns(
            masks,
            phi_scores,
            layer,
            head,
            save_path,
            RELEVANT_TOKENS[tokenization_type],
            origin,
        )

    save_dict_to_json(functional_heads, f"{save_dir}/functional_heads.json")


def has_one_digit_tokenizer(model: HookedTransformer) -> bool:
    """
    Check if the model has a one digit tokenizer.

    Args:
        model (HookedTransformer): The model to check.

    Returns:
        bool: True if the model has a one digit tokenizer.
    """
    tokens = model.to_tokens("10", prepend_bos=False)
    return tokens.size(1) != 1


def get_qk_matrix(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> t.Tensor:
    """
    Get the QK matrix for a specific attention head.

    Args:
        model (HookedTransformer): The model to analyze.
        layer (int): The layer index.
        head (int): The head index.

    Returns:
        t.Tensor: The QK matrix.
    """
    QK = model.QK[layer, head, :, :]
    QK = t.matmul(QK.A, QK.B)  # This is used to ensure QK is a tensor.
    return QK


def filter_embedding_matrix(
    model: HookedTransformer,
    matrix: t.Tensor,
    matrix_type: Literal["embedding", "unembedding"],
) -> t.Tensor:
    """
    Filter embedding or unembedding matrices to only include relevant tokens.

    Args:
        model (HookedTransformer): The model to analyze.
        matrix (t.Tensor): The matrix to filter.
        matrix_type (Literal["embedding", "unembedding"]): The type of matrix to filter.

    Returns:
        t.Tensor: The filtered matrix.
    """
    tokenization_type = "1_digit" if has_one_digit_tokenizer(model) else "multi_digit"
    tokens = [
        model.to_tokens(tok, prepend_bos=False).item()
        for tok in RELEVANT_TOKENS[tokenization_type]
    ]
    matrix = matrix[tokens, :] if matrix_type == "embedding" else matrix[:, tokens]
    return matrix


def maps_phi_consistency_score(
    QK_matrix: t.Tensor,
    topk: int = 1,
    consistency: bool = True,
) -> Tuple[float, t.Tensor]:
    """
    Calculate a score for how strong an attention head implements the "is equal"
    relation between number tokens.

    Since the QK matrix has the same numbers on the x and y axis, and since
    the relation we consider is "is equal", we compute the score simply as the
    ratio between:
        * number of elements that are on the diagonal of the QK matrix
        * the total number of pairs of tokens.

    Args:
        QK_matrix (t.Tensor): The QK matrix.
        topk (int): Number of top tokens to consider.
        consistency (bool): Whether to consider the consistency or inconsistency
            relation.

    Returns:
        Tuple[float, t.Tensor]: The MAPS phi relation score and the topk mask tensor.
    """
    topk_mask = t.zeros_like(QK_matrix)
    top_indices = t.topk(QK_matrix, topk, dim=1).indices
    for i, idxs in enumerate(top_indices):
        for idx in idxs:
            topk_mask[i, idx] = 1

    total_pairs = QK_matrix.shape[0]
    diagonal = topk_mask.diag().sum().item()

    if consistency:
        phi_maps = diagonal / total_pairs
    else:
        phi_maps = 1 - (diagonal / total_pairs)

    return phi_maps, topk_mask


def plot_qk_patterns(
    top_QK_matrices: List[t.Tensor],
    phi_scores: List[float],
    layer: int,
    head: int,
    save_path: str,
    token_labels: List[str],
    origin: str,
) -> None:
    """
    Plot the attention heatmap for the QK matrix using matplotlib.
    Only the topk tokens are highlighted in the heatmap.

    Args:
        top_QK_matrices (List[t.Tensor]): The top QK matrices.
        phi_scores (List[float]): The MAPS phi scores.
        layer (int): Layer number.
        head (int): Head number.
        save_path (str): The path to save the plot.
        token_labels (List[str]): Labels for the tokens.
        origin (str): The circuit to which the attention head belong.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(top_QK_matrices[0].cpu().float(), cmap="Blues")
    ax1.set_xticks(range(len(token_labels)))
    ax1.set_yticks(range(len(token_labels)))
    ax1.set_xticklabels(token_labels)
    ax1.set_yticklabels(token_labels)
    ax1.set_title(
        f"Consistency Pattern\nLayer {layer} Head {head}\nMAPS Score: {phi_scores[0]:.2f}"
    )
    ax1.set_xlabel("Tokens")
    ax1.set_ylabel("Tokens")

    ax2.imshow(top_QK_matrices[1].cpu().float(), cmap="Reds")
    ax2.set_xticks(range(len(token_labels)))
    ax2.set_yticks(range(len(token_labels)))
    ax2.set_xticklabels(token_labels)
    ax2.set_yticklabels(token_labels)
    ax2.set_title(
        f"Inconsistency Pattern\nLayer {layer} Head {head}\nMAPS Score: {phi_scores[1]:.2f}"
    )
    ax2.set_xlabel("Tokens")
    ax2.set_ylabel("Tokens")

    plt.tight_layout()
    plt.savefig(f"{save_path}_l{layer}h{head}_{origin}.png", dpi=300)
    plt.close()
