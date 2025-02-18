"""
Functions and modules for circuit attention analysis.
"""

import os
import re
from typing import Dict, List, Literal, Tuple, cast

import matplotlib.pyplot as plt
import torch as t
from auto_circuit.data import PromptDataLoader
from auto_circuit.types import Edge
from tqdm import tqdm
from transformer_lens import HookedTransformer

from llm_error_detection.circuits.circuit_analysis import Circuit

RELEVANT_TOKENS = [
    "[op1-in-eq]_occ_1",
    "[plus]_occ_1",
    "[space]_occ_7",
    "[op2-in-eq]_occ_1",
    "[equals]_occ_1",
    "[space_after_eq]_occ_1",
    "[z1-first]_occ_1",
    "[z1-second]_occ_1",
    "[z1]_occ_1",
    "[z2-first]_occ_1",
    "[z2-second]_occ_1",
    "[z2]_occ_1",
]


def visualize_attention_patterns(
    model: HookedTransformer,
    model_name: str,
    dataloaders: Dict[str, PromptDataLoader],
    circuits: Dict[str, Circuit],
    save_dir: str,
    uid: str,
) -> None:
    """
    Visualize the attention patterns.

    Args:
        model (HookedTransformer): The model to analyze.
        model_name (str): The name of the model.
        dataloaders (Dict[str, PromptDataLoader]): The dataloaders to use for the analysis.
        circuits (Dict[str, Circuit]): The circuits to analyze.
        save_dir (str): The directory to save the attention patterns.
        uid (str): The unique identifier for the attention patterns.
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

    attn_info, heads_origin = get_union_heads(attn_info_list)

    # Collect results for these attention heads
    result_dict = {}

    for perturbation, dataloader in dataloaders.items():
        perturbation_lit = cast(Literal["z1", "z2", "none", "both"], perturbation)
        token_labels, filtered_patterns = obtain_pattern_for_perturbation(
            model, model_name, dataloader, attn_info, perturbation_lit
        )
        result_dict[perturbation] = (
            token_labels,
            filtered_patterns,
            attn_info,
            heads_origin,
        )

    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{uid}"

    plot_attention_patterns(save_path, result_dict)


def obtain_pattern_for_perturbation(
    model: HookedTransformer,
    model_name: str,
    dataloader: PromptDataLoader,
    attn_info: List[Tuple[str, str, int, int]],
    perturbation: Literal["z1", "z2", "none", "both"],
) -> Tuple[List[List[str]], t.Tensor]:
    """
    Get the attention patterns for the attention heads in the circuit.

    Args:
        model (PatchableModel): The model to analyze.
        model_name (str): The name of the model.
        data_loader (PromptDataLoader): The data loader to use for the analysis.
        attn_info (List[Tuple[str, str, int, int]]): The attention head information.
        perturbation (Literal["z1", "z2", "none", "both"]): The perturbation used.

    Returns:
        Tuple[List[List[str], t.Tensor]: The token labels and the patterns.
    """

    patterns: List[List[t.Tensor]] = [[] for _ in range(len(attn_info))]

    for batch in tqdm(dataloader, desc=f"Obtaining patterns for {perturbation}"):
        batch = batch.corrupt if perturbation == "none" else batch.clean
        _, cache = model.run_with_cache(batch, return_type="logits", prepend_bos=False)

        for i, head in enumerate(attn_info):
            patterns[i].append(cache["pattern", head[2]].cpu()[:, head[3]])
        del cache

    concat_patterns = [t.cat(p).mean(dim=0) for p in patterns]
    stacked_patterns = t.stack(concat_patterns, dim=0)

    if "llama-3.2" in model_name.lower():
        seq_labels = ["[bos]"] + dataloader.seq_labels
    else:
        seq_labels = dataloader.seq_labels

    token_labels, filtered_patterns = filter_relevant_token_positions(
        seq_labels, stacked_patterns
    )

    return token_labels, filtered_patterns


def get_attention_heads(
    circuit: Circuit,
    attn_type: Literal["qkv", "out"] = "qkv",
) -> List[Tuple[Edge, Literal["src", "dest"], str]]:
    """
    Extract attention head information from circuit edges.

    Args:
        circuit (Circuit): circuit object with relevant circuit components.
        attn_type (Literal["qkv", "out"], optional): The type of attention to extract. Defaults to "qkv".

    Returns:
        List[Tuple[Edge, Literal["src", "dest"], str]]: List of tuples containing edges,
            their type (src or dest) and the seq_label for attention heads in the circuit.
    """
    attention_hook_types = (
        ["hook_k_input", "hook_v_input", "hook_q_input"]
        if attn_type == "qkv"
        else ["attn.hook_result"]
    )
    attn_heads: List[Tuple[Edge, Literal["src", "dest"], str]] = []

    for seq_label, comp in circuit.tok_pos_edges.items():
        for edge in comp:
            if attn_type == "qkv":
                is_in_circuit = any(
                    hook in edge.dest.module_name for hook in attention_hook_types
                )
            else:
                is_in_circuit = any(
                    hook in edge.src.module_name for hook in attention_hook_types
                )

            edge_type: Literal["src", "dest"] = "src"
            if is_in_circuit and attn_type == "out":
                attn_heads.append((edge, edge_type, seq_label))
            elif is_in_circuit:
                edge_type = "dest"
                attn_heads.append((edge, edge_type, seq_label))

    return attn_heads


def get_union_heads(
    attn_info_list: List[List[Tuple[str, str, int, int]]],
) -> Tuple[List[Tuple[str, str, int, int]], List[str]]:
    """
    Return the union of attention heads across two sets. Also return a list of origin labels:
    'z1' if a head is only in attn_info_list[0], 'z2' if it is only in attn_info_list[1],
    or 'all' if it appears in both.

    Args:
        attn_info_list (List[List[Tuple[str, str, int, int]]]): A list of two lists of attention head info.

    Returns:
        (union_heads, origins):
        union_heads: Sorted union of all heads.
        origins: Labels for each head in union_heads.
    """
    z1_heads = set(attn_info_list[0])
    z2_heads = set(attn_info_list[1])

    union_heads = z1_heads.union(z2_heads)
    union_heads_sorted = sorted(union_heads, key=lambda x: (x[2], x[3]))

    origins = []
    for head in union_heads_sorted:
        in_z1_heads = head in z1_heads
        in_z2_heads = head in z2_heads
        if in_z1_heads and in_z2_heads:
            origins.append("all")
        elif in_z1_heads:
            origins.append("z1")
        else:
            origins.append("z2")

    return union_heads_sorted, origins


def filter_relevant_token_positions(
    seq_labels: List[str],
    patterns: t.Tensor,
) -> Tuple[List[List[str]], t.Tensor]:
    """
    Filter the relevant token positions.

    Args:
        seq_labels (List[str]): The labels for the tokens.
        patterns (t.Tensor): The attention patterns.

    Returns:
        Tuple[List[List[str], t.Tensor]: The filtered labels and patterns.
    """
    relevant_indices = [
        i for i, label in enumerate(seq_labels) if label in RELEVANT_TOKENS
    ]
    token_labels = [
        [seq_labels[i] for i in relevant_indices] for _ in range(patterns.size(0))
    ]
    patterns = patterns[:, relevant_indices][:, :, relevant_indices]
    return token_labels, patterns


def parse_attention_edge(
    edge: Edge, edge_type: Literal["src", "dest"], seq_label: str
) -> Tuple[str, str, int, int]:
    """
    Parse the edge data to get attention head information.

    Args:
        edge (Edge): edge with a src attention head.
        edge_type (Literal["src", "dest"]): whether the head is source or destination.
        seq_label (str): the sequence label for the attention head.

    Returns:
        Tuple[str, str, int, int]: tuple with module name plus layer and head index.
    """
    if edge_type == "src":
        module_name = edge.src.module_name
        edge_name = edge.name.split("->")[0]
    else:
        module_name = edge.dest.module_name
        edge_name = edge.name.split("->")[1]

    search_res = re.search(r"A(\d+)\.(\d+)", edge_name)

    if search_res is not None:
        layer, head = search_res.groups()
    else:
        raise ValueError(f"No valid heads found for {edge.name}!")

    return seq_label, module_name, int(layer), int(head)


def plot_attention_patterns(
    save_path: str,
    result_dict: Dict[
        str,
        Tuple[
            List[List[str]], List[t.Tensor], List[Tuple[str, str, int, int]], List[str]
        ],
    ],
) -> None:
    """
    Plot the attention patterns.

    Args:
        save_path (str): The path to save the attention patterns.
        result_dict (Dict[str, Tuple[List[List[str]], List[t.Tensor], List[Tuple[str, str, int, int]], List[str]]]): The token_labels,
            filtered_patterns, attn_info, and heads_origin for each perturbation.
    """
    total_heads = len(result_dict["z1"][2])

    title_dict = {
        "z1": "Invalid Result",
        "z2": "Invalid Answer",
        "none": "Correct",
        "both": "Invalid Result & Answer",
    }

    for idx in range(total_heads):
        layer = result_dict["z1"][2][idx][2]
        head = result_dict["z1"][2][idx][3]

        # Increased figure width and adjusted spacing
        fig, axes = plt.subplots(1, 4, figsize=(29, 5))
        plt.subplots_adjust(wspace=0.5, hspace=0.3, top=0.85, bottom=0.2)

        for i, (pert, (lbls, pats, info, origin)) in enumerate(result_dict.items()):
            labels = [re.sub(r"_occ_\d", "", lbl) for lbl in lbls[idx]]
            labels = [
                label.replace("z1", "result") if "z1" in label else label
                for label in labels
            ]
            labels = [
                label.replace("z2", "answer") if "z2" in label else label
                for label in labels
            ]

            ax = axes[i]
            ax.imshow(pats[idx], cmap="plasma", vmin=0, vmax=1)
            ax.set_title(title_dict[pert], fontsize=16)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=12)

        cbar = fig.colorbar(ax.images[0], ax=axes.ravel().tolist())
        cbar.ax.tick_params(labelsize=14)
        plt.savefig(
            f"{save_path}_l{layer}h{head}_{origin[idx]}.png", bbox_inches="tight"
        )
        plt.close()
