"""
Functions and modules to apply vocabulary projections of intermediate activations.
"""

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import torch as t
from auto_circuit.data import PromptDataLoader
from auto_circuit.utils.custom_tqdm import tqdm
from transformer_lens import HookedTransformer

from llm_error_detection.circuits.circuit_analysis import Circuit

RELEVANT_TOKENS = [
    "[op1-in-eq]_occ_1",
    "[plus]_occ_1",
    "[op2-in-eq]_occ_1",
    "[equals]_occ_1",
    "[space_after_eq]_occ_1",
    "[mistake]_occ_1",
    "[mistake]_occ_2",
    "[mistake-first]_occ_1",
    "[mistake-first]_occ_2",
    "[mistake-second]_occ_1",
    "[mistake-second]_occ_2",
    "[correct]_occ_1",
    "[correct]_occ_2",
    "[correct-first]_occ_1",
    "[correct-first]_occ_2",
    "[correct-second]_occ_1",
    "[correct-second]_occ_2",
    "the",
    "above",
    "reasoning",
    "is",
]


z1_1digit = [
    ("[op1-in-eq]_occ_1", "[mistake-second]_occ_1"),
    ("[plus]_occ_1", "[mistake-second]_occ_1"),
    ("[op2-in-eq]_occ_1", "[mistake-second]_occ_1"),
    ("[equals]_occ_1", "[mistake-second]_occ_1"),
    ("[space_after_eq]_occ_1", "[mistake-first]_occ_1"),
    ("[mistake-first]_occ_1", "[mistake-second]_occ_1"),
]

z2_1digit = [
    ("[op1-in-eq]_occ_1", "[correct-second]_occ_1"),
    ("[plus]_occ_1", "[correct-second]_occ_1"),
    ("[op2-in-eq]_occ_1", "[correct-second]_occ_1"),
    ("[equals]_occ_1", "[correct-second]_occ_1"),
    ("[space_after_eq]_occ_1", "[correct-first]_occ_1"),
    ("[correct-first]_occ_1", "[correct-second]_occ_1"),
]

z1_multidigit = [
    ("[op1-in-eq]_occ_1", "[mistake]_occ_1"),
    ("[plus]_occ_1", "[mistake]_occ_1"),
    ("[op2-in-eq]_occ_1", "[mistake]_occ_1"),
    ("[equals]_occ_1", "[mistake]_occ_1"),
    ("[space_after_eq]_occ_1", "[mistake]_occ_1"),
]

z2_multidigit = [
    ("[op1-in-eq]_occ_1", "[correct]_occ_1"),
    ("[plus]_occ_1", "[correct]_occ_1"),
    ("[op2-in-eq]_occ_1", "[correct]_occ_1"),
    ("[equals]_occ_1", "[correct]_occ_1"),
    ("[space_after_eq]_occ_1", "[correct]_occ_1"),
]

COMPUTATION_TOKENS = {
    "1_digit": {
        "z1": z1_1digit,
        "z2": z2_1digit,
        "both": z1_1digit,  # same as z1
        "none": z1_1digit,  # same as z1 correct input
    },
    "multi_digit": {
        "z1": z1_multidigit,
        "z2": z2_multidigit,
        "both": z1_multidigit,  # same as z1
        "none": z1_multidigit,  # same as z1 correct input
    },
}


def visualize_mlp_projections(
    model: HookedTransformer,
    dataloader: PromptDataLoader,
    circuit: Circuit,
    save_dir: str,
    uid: str,
    perturbation: Literal["z1", "z2", "none", "both"],
    template: str,
) -> None:
    """
    Get the vocabulary projections for the MLPs (both input and output) in the circuit.

    Args:
        model (HookedTransformer): The model.
        dataloader (PromptDataLoader): The dataloader.
        circuit (Circuit): The circuit.
        save_dir (str): The directory to save the plot.
        uid (str): The unique identifier for the plot.
        perturbation (Literal["z1", "z2", "none", "both"]): The perturbation applied to the data.
        template (str): The template used.
    """

    important_positions = build_important_positions(
        dataloader.seq_labels, model.cfg.n_layers, perturbation, "mlp"
    )

    mlp_projections = defaultdict(list)

    # Collect projections across batches
    for batch in tqdm(dataloader):
        batch = batch.corrupt if perturbation == "none" else batch.clean
        _, cache = model.run_with_cache(batch, return_type="logits", prepend_bos=False)

        for mlp_in, mlp_out, seq_label in important_positions:
            for mlp in [mlp_in, mlp_out]:
                vocab_proj = t.einsum(
                    "bpd,dv->bpv", cache[mlp], model.unembed.W_U
                )  # (batch_size, seq_len, vocab_size)
                max_proj = vocab_proj.max(dim=-1).indices  # (batch_size, seq_len)
                mlp_projections[f"{mlp}_at_{seq_label}"].append(max_proj.cpu())

    # Process projections for each MLP pair
    final_projections: Dict[str, List[int]] = {}
    for mlp_in, mlp_out, seq_label in important_positions:
        for mlp in [mlp_in, mlp_out]:
            all_projs = t.cat(
                mlp_projections[f"{mlp}_at_{seq_label}"], dim=0
            )  # (batch_size * num_batches, seq_len)
            values, counts = t.unique(all_projs, dim=0, return_counts=True)
            max_count_idx = t.argmax(counts)
            final_projections[f"{mlp}_at_{seq_label}"] = (
                values[max_count_idx].squeeze().tolist()
            )

    os.makedirs(save_dir, exist_ok=True)
    plot_vocabulary_projections(
        model=model,
        seq_labels=dataloader.seq_labels,
        important_positions=important_positions,
        colored_indexes=colored_indexes(
            circuit,
            dataloader.seq_labels,
            perturbation,
            RELEVANT_TOKENS,
            computation_circuit=False,
        ),
        projections=final_projections,
        save_dir=save_dir,
        uid=uid,
        perturbation=perturbation,
        template=template,
        projection_type="mlp",
    )


def visualize_residual_projections(
    model: HookedTransformer,
    dataloader: PromptDataLoader,
    circuit: Circuit,
    save_dir: str,
    uid: str,
    perturbation: Literal["z1", "z2", "none", "both"],
    template: str,
) -> None:
    """
    Project the top 3 tokens projected from the residual stream from all layers.

    Args:
        model (HookedTransformer): The model.
        dataloader (PromptDataLoader): The dataloader.
        circuit (Circuit): The circuit.
        save_dir (str): The directory to save the plot.
        uid (str): The unique identifier for the plot.
        perturbation (Literal["z1", "z2", "none", "both"]): The perturbation applied to the data.
        template (str): The template used.
    """
    important_positions = build_important_positions(
        dataloader.seq_labels, model.cfg.n_layers, perturbation, "resid"
    )

    residual_projections = defaultdict(list)
    hooks = [
        (mid, post)
        for (mid, post, label) in important_positions
        if label == important_positions[0][2]
    ]

    for batch in tqdm(dataloader):
        batch = batch.corrupt if perturbation == "none" else batch.clean
        _, cache = model.run_with_cache(batch, return_type="logits", prepend_bos=False)

        # Project each layer's mid and post residual stream to vocabulary
        for resid_mid, resid_post in hooks:
            vocab_proj = t.einsum(
                "bpd,dv->bpv", cache[resid_post], model.unembed.W_U
            )  # (batch_size, seq_len, vocab_size)
            max_proj = vocab_proj.topk(k=3, dim=-1).indices  # (batch_size, seq_len, 3)
            residual_projections[resid_post].append(
                max_proj.reshape(
                    max_proj.size(-1) * max_proj.size(0), max_proj.size(1)
                ).cpu()  # (batch_size * 3, seq_len)
            )

    # Process projections for each layer and hook type
    final_projections: Dict[str, List[int]] = {}
    for resid_mid, resid_post in hooks:
        all_projs = t.cat(
            residual_projections[resid_post], dim=0
        )  # (batch_size * 3 * num_batches, seq_len)
        top_tokens = []
        for position in range(all_projs.size(1)):
            top_at_pos = all_projs[:, position]
            unique_vals, counts = t.unique(top_at_pos, return_counts=True)
            if len(unique_vals) < 3:  # WHY DOES THIS HAPPEN??
                top3 = unique_vals
            else:
                top3 = unique_vals[t.topk(counts, k=3).indices]
            top_tokens.append(top3.tolist())
        final_projections[resid_post] = top_tokens

    os.makedirs(save_dir, exist_ok=True)
    plot_vocabulary_projections(
        model=model,
        seq_labels=dataloader.seq_labels,
        important_positions=important_positions,
        colored_indexes=None,
        projections=final_projections,
        save_dir=save_dir,
        uid=uid,
        perturbation=perturbation,
        template=template,
        projection_type="residual",
    )


def visualize_computation_projections(
    model: HookedTransformer,
    dataloader: PromptDataLoader,
    circuit: Circuit,
    save_dir: str,
    uid: str,
    perturbation: Literal["z1", "z2", "none", "both"],
    template: str,
) -> None:
    """
    Similar to visualize_mlp_projections but focuses on COMPUTATION_TOKENS positions
    and uses the token "right" and "wrong" for projections that encode the correct or wrong
    result number, respectively.

    Args:
        model (HookedTransformer): The model.
        dataloader (PromptDataLoader): The dataloader.
        circuit (Circuit): The circuit.
        save_dir (str): The directory to save the plot.
        uid (str): The unique identifier for the plot.
        perturbation (Literal["z1", "z2", "none", "both"]): The perturbation applied to the data.
        template (str): The template used.
    """
    tokenization_type = (
        "1_digit" if has_one_digit_tokenizer(dataloader.seq_labels) else "multi_digit"
    )
    positions = COMPUTATION_TOKENS[tokenization_type][perturbation]
    right_token, wrong_token = special_number_tokens(model)

    comp_projections = defaultdict(list)

    for batch in tqdm(dataloader):
        no_error_batch = batch.corrupt
        batch = batch.corrupt if perturbation == "none" else batch.clean
        _, cache = model.run_with_cache(batch, return_type="logits", prepend_bos=False)
        # Iterate over all important positions, i.e. computation tokens
        for pos, target in positions:
            for layer in range(model.cfg.n_layers):
                mlp_in = f"blocks.{layer}.hook_mlp_in"
                mlp_out = f"blocks.{layer}.hook_mlp_out"
                for mlp in [mlp_in, mlp_out]:
                    vocab_proj = t.einsum(
                        "bpd,dv->bpv", cache[mlp], model.unembed.W_U
                    )  # (batch_size, seq_len, vocab_size)
                    max_proj = vocab_proj.max(dim=-1).indices  # (batch_size, seq_len)

                    pos_idx = dataloader.seq_labels.index(pos)
                    target_idx = dataloader.seq_labels.index(target)

                    computation_result = no_error_batch[:, target_idx]
                    # Replace the prediction of token at pos with "right" or "wrong"
                    # based on whether the representation encode the correct number in the result
                    max_proj[:, pos_idx] = t.where(
                        max_proj[:, pos_idx] == computation_result,
                        right_token * t.ones_like(computation_result),
                        wrong_token * t.ones_like(computation_result),
                    )
                    comp_projections[f"{mlp}_at_{pos}"].append(max_proj.cpu())

    final_projections: Dict[str, List[int]] = {}
    for pos, target in positions:
        for layer in range(model.cfg.n_layers):
            mlp_in = f"blocks.{layer}.hook_mlp_in"
            mlp_out = f"blocks.{layer}.hook_mlp_out"
            for mlp in [mlp_in, mlp_out]:
                all_projs = t.cat(comp_projections[f"{mlp}_at_{pos}"], dim=0).squeeze()
                values, counts = t.unique(all_projs, dim=0, return_counts=True)
                most_frequent = values[t.argmax(counts)]
                final_projections[f"{mlp}_at_{pos}"] = most_frequent.squeeze().tolist()

    important_positions = [
        (f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", lbl)
        for lbl, target in positions
        for layer in range(model.cfg.n_layers)
    ]
    plot_vocabulary_projections(
        model=model,
        seq_labels=dataloader.seq_labels,
        important_positions=important_positions,
        colored_indexes=colored_indexes(
            circuit,
            dataloader.seq_labels,
            perturbation,
            [pos for pos, _ in positions],
            computation_circuit=True,
        ),
        projections=final_projections,
        save_dir=save_dir,
        uid=uid,
        perturbation=perturbation,
        template=template,
        projection_type="computation",
    )


def build_important_positions(
    seq_labels: List[str],
    n_layers: int,
    perturbation: Literal["z1", "z2", "none", "both"],
    hook_type: Literal["mlp", "resid"],
) -> List[Tuple[str, str, str]]:
    """
    Build a list of (input_hook, output_hook, seq_label) for each relevant token.

    Args:
        seq_labels (List[str]): The sequence labels.
        n_layers (int): The number of layers in the model.
        perturbation (Literal["z1", "z2", "none", "both"]): The perturbation applied to the data.
        hook_type (Literal["mlp", "resid"]): The hook type.

    Returns:
        List[Tuple[str, str, str]]: The list of (input_hook, output_hook, seq_label).
    """
    positions: List[Tuple[str, str, str]] = []
    for seq_label in seq_labels:
        if seq_label in RELEVANT_TOKENS:
            if perturbation == "both":
                seq_label = seq_label.replace("correct", "mistake")
            for layer in range(n_layers):
                if hook_type == "mlp":
                    inpt = f"blocks.{layer}.hook_mlp_in"
                    out = f"blocks.{layer}.hook_mlp_out"
                else:
                    inpt = f"blocks.{layer}.hook_resid_mid"
                    out = f"blocks.{layer}.hook_resid_post"
                positions.append((inpt, out, seq_label))
    return positions


def colored_indexes(
    circuit: Circuit,
    seq_labels: List[str],
    perturbation: Literal["z1", "z2", "none", "both"],
    relevant_tokens: List[str],
    computation_circuit: bool = False,
) -> List[Tuple[int, int]]:
    """
    Get the indexes of elements that belongs to the circuit
    in the plotted table.

    Args:
        circuit (Circuit): The circuit.
        seq_labels (List[str]): The sequence labels.
        perturbation (Literal["z1", "z2", "none", "both"]): The perturbation applied to the data.
        relevant_tokens (List[str]): The relevant tokens.
        computation_circuit (bool): Whether the computation circuit is being used.

    Returns:
        List[Tuple[int, int]]: The important positions.
    """
    seq_labels = [seq_label for seq_label in seq_labels if seq_label in relevant_tokens]

    if computation_circuit:
        seq_labels = [
            seq_label.replace("mistake", "correct") for seq_label in seq_labels
        ]

    colored_idxes = []
    for seq_label, comp in circuit.tok_pos_edges.items():
        if perturbation == "both":
            seq_label = seq_label.replace("correct", "mistake")

        for edge in comp:
            is_mlp_source = "hook_mlp_out" in edge.src.module_name
            if is_mlp_source:
                seq_pos = seq_labels.index(seq_label)
                colored_idxes.append((int(edge.src.module_name.split(".")[1]), seq_pos))
    return colored_idxes


def plot_vocabulary_projections(
    model: HookedTransformer,
    seq_labels: List[str],
    important_positions: List[Tuple[str, str, str]],
    projections: Dict[str, t.Tensor],
    colored_indexes: Optional[List[Tuple[int, int]]],
    save_dir: str,
    uid: str,
    perturbation: Literal["z1", "z2", "none", "both"],
    template: str,
    projection_type: Literal["mlp", "resid", "computation"] = "mlp",
) -> None:
    """
    Plot the vocabulary projections for the circuit MLPs, residual stream, or computation tokens.

    Args:
        model (HookedTransformer): The model.
        seq_labels (List[str]): The sequence labels.
        important_positions (List[Tuple[str, str, str]]): The list of input, output, position to plot.
        colored_indexes (List[Tuple[int, int]]): List of (row,col) coordinates to highlight, or None
        projections (Dict[str, t.Tensor]): The vocabulary projections.
        save_dir (str): The directory to save the plot.
        uid (str): The unique identifier for the plot.
        perturbation (Literal["z1", "z2", "none", "both"]): The perturbation applied to the data.
        template (str): The template used.
        projection_type (Literal["mlp", "resid", "computation"], optional): The type of projection. Defaults to "mlp".
    """

    data: Dict[int, Any] = {}

    for inpt, out, seq_label in important_positions:

        seq_pos = seq_labels.index(seq_label)

        module_name = int(inpt.split(".")[1])

        if data.get(module_name) is None:
            data[module_name] = {}

        if projection_type == "residual":
            top_3 = [f"{model.to_string(i)}" for i in projections[out][seq_pos]]
            data[module_name][seq_labels[seq_pos]] = "  |  ".join(top_3)
        else:
            in_token = model.to_string(projections[f"{inpt}_at_{seq_label}"][seq_pos])
            out_token = model.to_string(projections[f"{out}_at_{seq_label}"][seq_pos])
            data[module_name][seq_labels[seq_pos]] = f"{in_token} → {out_token}"

    if projection_type == "residual":
        title = f"Residual Stream Projections - Template: {template}, Perturbation: {perturbation}"
        filename = f"residual_projections_{uid}.png"
    elif projection_type == "computation":
        title = f"Computation Projections (in→out) - Template: {template}, Perturbation: {perturbation}"
        filename = f"computation_projections_{uid}.png"
    else:
        title = f"MLP Projections (in→out) - Template: {template}, Perturbation: {perturbation}"
        filename = f"mlp_projections_{uid}.png"

    residual = projection_type == "residual"
    df = df_cleaning(pd.DataFrame(data).T, residual=residual, perturbation=perturbation)
    output_path = os.path.join(save_dir, filename)

    plot_table(
        df,
        colored_indexes,
        output_path,
        title,
    )


def has_one_digit_tokenizer(seq_labels: List[str]) -> bool:
    """
    Check if the dataloader has numbers tokenized digit by digit or not.

    Args:
        seq_labels (List[str]): the sequence labels.

    Returns:
        bool: Whether the dataloader has a one digit tokenization.
    """
    return "[mistake-first]_occ_1" in seq_labels


def special_number_tokens(model: HookedTransformer) -> Tuple[int, int]:
    """
    Get the special tokens for 'right' and 'wrong'.
    """
    right_token_id = model.to_tokens(" right", prepend_bos=False).squeeze().item()
    wrong_token_id = model.to_tokens(" wrong", prepend_bos=False).squeeze().item()
    return right_token_id, wrong_token_id


def df_cleaning(
    df: pd.DataFrame, perturbation: str, residual: bool = False
) -> pd.DataFrame:
    """
    Perform a series of processing steps to make the df data more readable.

    Args:
        df (pd.DataFrame): The dataframe to clean.
        perturbation (str): The perturbation applied to the data.
        residual (bool, optional): Whether the dataframe is for residual projections. Defaults to False.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # Fill NaN values with "-"
    df.fillna("-", inplace=True)
    # Rename columns to remove _occ_# suffixes
    df.columns = [re.sub(r"_occ_\d", "", col) for col in df.columns]
    # Replace "mistake" with "correct" if perturbation is "none"
    if perturbation == "none":
        df.columns = [col.replace("mistake", "correct") for col in df.columns]
    # Reorder rows based on layer number
    df = df.sort_index()
    if residual:
        df.index = "Layer " + df.index.astype(str)
    else:
        df.index = "MLP " + df.index.astype(str)
    return df


def plot_table(
    df: pd.DataFrame,
    colored_indexes: Optional[List[Tuple[int, int]]],
    output_path: str,
    title: str = "",
    show_index: bool = True,
) -> None:
    """
    Plot a table from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        colored_indexes (List[Tuple[int, int]]): List of (row,col) coordinates to highlight, or None
        output_path (str): The path to save the plot.
        title (str, optional): The title of the plot. Defaults to empty string "".
        show_index (bool, optional): Whether to show the index column. Defaults to True.
    """

    if show_index:
        df = df.reset_index()
        df.rename(columns={"index": ""}, inplace=True)

    # Create fill color matrix initialized to white
    fill_colors = [["white"] * len(df.columns) for _ in range(len(df))]

    # Update colors for highlighted cells if colored_indexes provided
    if colored_indexes:
        for row, col in colored_indexes:
            fill_colors[row][col + 1] = "lightblue"

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=df.columns,
                    fill_color=["white"] + ["lightgrey"] * (len(df.columns) - 1),
                    align="center",
                    line_color=["lightgrey"]
                    + ["darkslategray"] * (len(df.columns) - 1),
                ),
                cells=dict(
                    values=df.values.transpose(),
                    fill_color=list(
                        map(list, zip(*fill_colors))
                    ),  # Transpose fill_colors
                    align="left",
                    height=30,
                    line_color="lightgrey",
                ),
            )
        ]
    )

    if title != "":
        fig.update_layout(title_text=title, font=dict(family="Noto Sans CJK", size=12))
    else:
        fig.update_layout(font=dict(family="Noto Sans CJK"))

    fig.update_layout(
        autosize=False,
        width=200 * len(df.columns),
        height=40 * (len(df) + 1),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.write_image(output_path, engine="kaleido", scale=2)
