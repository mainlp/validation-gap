"""
Functions and modules for further circuit analyses
"""

import logging
from collections import defaultdict
from dataclasses import asdict, replace
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple

import torch as t
from auto_circuit.data import PromptDataLoader
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    DestNode,
    Edge,
    PatchType,
    PatchWrapper,
    PruneScores,
    SrcNode,
)
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import patch_mode
from auto_circuit.utils.misc import module_by_name
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.visualize import node_name

from llm_error_detection.utils import (
    faithfulness_score,
    get_save_dir_name,
    load_json_to_dict,
    save_dict_to_json,
)


class Circuit:
    def __init__(self, tok_pos_edges: Dict[str, List[Edge]] | None = None):
        """
        Initialize a Circuit with optional token position edges.

        Args:
            tok_pos_edges (Dict[str, List[Edge]] | None, optional): A dictionary mapping token positions labels to a list of edges.
                Defaults to None. If provided, it initializes the circuit with these edges.
        """
        self.tok_pos_edges: DefaultDict[str, List[Edge]] = defaultdict(list)
        self.num_edges: int = 0

        if tok_pos_edges:
            self.tok_pos_edges.update(tok_pos_edges)
            for _, edge_list in self.tok_pos_edges.items():
                self.num_edges += len(edge_list)

    def add_edge(self, token_pos: str, edge: Edge) -> None:
        """
        Add an edge to the circuit at the specified token position.

        Args:
            token_pos (str): The token position to associate with the edge.
            edge (Edge): The edge to be added.
        """
        edges = self.tok_pos_edges[token_pos]
        if edge not in edges:
            edges.append(edge)
            self.num_edges += 1
        else:
            logging.warning(
                f"Edge {edge.name} already exists at token position label: {token_pos}. Skipping..."
            )

    def _repr_dict(
        self, prune_scores: PruneScores, unembed: bool, use_source: bool, prefix: str
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """
        Internal method to create a representation of the circuit based on source or target nodes.

        Args:
            prune_scores (PruneScores): The prune scores for the edges.
            unembed (bool): Whether to unembed the node names.
            use_source (bool): Whether to generate the representation based on source nodes.

        Returns:
            Dict[str, Dict[str, List[Tuple[str, float]]]]: The circuit representation.
        """
        representation: DefaultDict[str, Dict[str, List[Tuple[str, float]]]] = (
            defaultdict(dict)
        )
        for token_pos, edge_list in self.tok_pos_edges.items():
            token_repr = representation[token_pos]
            for edge in edge_list:
                main_node = (
                    node_name(edge.src.name, unembed)
                    if use_source
                    else node_name(edge.dest.name, unembed)
                )
                other_node = (
                    node_name(edge.dest.name, unembed)
                    if use_source
                    else node_name(edge.src.name, unembed)
                )
                edge_score = edge.prune_score(prune_scores).item()

                main_node_name = f"{prefix}: {main_node}"
                if main_node_name not in token_repr:
                    token_repr[main_node_name] = []
                token_repr[main_node_name].append((other_node, edge_score))

        return representation

    def src_repr(
        self, prune_scores: PruneScores, unembed: bool = True
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """
        Return a representation of the circuit in terms of the source nodes.

        Args:
            prune_scores (PruneScores): The prune scores for the edges in the circuit.
            unembed (bool, optional): Whether to unembed the node names. Defaults to True.

        Returns:
            Dict[str, Dict[str, List[Tuple[str, float]]]]: The source-based representation of the circuit.
        """
        return self._repr_dict(
            prune_scores, unembed, use_source=True, prefix="source-node"
        )

    def target_repr(
        self, prune_scores: PruneScores, unembed: bool = True
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """
        Return a representation of the circuit in terms of the target nodes.

        Args:
            prune_scores (PruneScores): The prune scores for the edges in the circuit.
            unembed (bool, optional): Whether to unembed the node names. Defaults to True.

        Returns:
            Dict[str, Dict[str, List[Tuple[str, float]]]]: The target-based representation of the circuit.
        """
        return self._repr_dict(
            prune_scores, unembed, use_source=False, prefix="target-node"
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the circuit.

        Returns:
            str: The string representation of the circuit.
        """
        return f"Number of edges: {self.num_edges}\nRelevant token positions: {list(self.tok_pos_edges.keys())}\nEdges: \n{self.tok_pos_edges}"

    @property
    def size(self):
        """
        The size of the circuit, given by the number of edges in the circuit.

        Returns:
            int: The size of the circuit.
        """
        return self.num_edges

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Circuit object to a dictionary suitable for JSON serialization.

        Returns:
            Dict[str, Any]: The dictionary representation of the Circuit.
        """
        return {
            "tok_pos_edges": {
                key: [asdict(edge) for edge in edge_list]
                for key, edge_list in self.tok_pos_edges.items()
            },
            "num_edges": self.num_edges,
        }

    def save_to_file(self, file_path: str) -> None:
        """
        Save the Circuit object to a JSON file.

        Args:
            file_path (str): The file path where the Circuit will be saved.
        """
        save_dict_to_json(data=self.to_dict(), file_path=file_path)


def circuit_from_dict(data_dict: Dict[str, Any]) -> Circuit:
    """
    Create a Circuit object from a dictionary that was serialized from a Circuit object.

    Args:
        data_dict (Dict[str, Any]): The dictionary to create the Circuit from.

    Returns:
        Circuit: The Circuit object created from the dictionary.
    """

    tok_pos_edges = {
        key: [
            Edge(
                src=SrcNode(**edge["src"]),
                dest=DestNode(**edge["dest"]),
                seq_idx=edge["seq_idx"],
            )
            for edge in edge_list
        ]
        for key, edge_list in data_dict["tok_pos_edges"].items()
    }
    circuit = Circuit(tok_pos_edges)
    assert (
        circuit.num_edges == data_dict["num_edges"]
    ), f"Number of edges does not align with number reported in file: {circuit.num_edges} != {data_dict['num_edges']}!"

    return circuit


def load_circuit_from_file(file_path: str) -> Circuit:
    """
    Load a Circuit object from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the serialized Circuit data.

    Returns:
        Circuit: The Circuit object reconstructed from the JSON file.
    """
    data_dict = load_json_to_dict(file_path=file_path)
    circuit = circuit_from_dict(data_dict)
    return circuit


def load_circuit(
    save_model_name: str,
    perturbation: str,
    template_name: str,
    grad_function: str,
    answer_function: str,
    train_size: int,
    intersection_overlap: Optional[str] = None,
) -> Circuit:
    """
    Load a Circuit object using specified parameters.

    Args:
        save_model_name (str): The name of the model used to generate the circuit.
        perturbation (str): The perturbation used in the data.
        template_name (str): The template used in the data.
        grad_function (str): The gradient function used to generate the circuit.
        answer_function (str): The answer function used to generate the circuit.
        train_size (int): The number of samples used to compute edge scores.
        intersection_overlap (Optional[str]): The number of circuits to intersect. If None, no intersection is performed.
    """
    if template_name == "intersection":
        assert (
            intersection_overlap is not None
        ), "Intersection overlap must be provided!"

    result_path = get_save_dir_name(
        prefix="results/discovered-circuits", template=template_name
    )
    result_path += f"/{save_model_name}"
    uid = f"{perturbation}_template_{template_name}_gradfunc_{grad_function}_ansfunc_{answer_function}_train_size_{train_size}"
    if intersection_overlap is not None:
        uid += f"_overlap_{intersection_overlap}"
    file_path = f"{result_path}/circuit_{uid}.json"

    return load_circuit_from_file(file_path=file_path)


def circuit_intersection(circuits: List[Circuit]) -> Circuit:
    """
    Compute the intersection of a list of circuits.

    This function takes a list of Circuit objects and returns a new Circuit object
    that contains only the edges that are present in all circuits.

    Args:
        circuits (List[Circuit]): The list of circuits to intersect.

    Returns:
        Circuit: The circuit that is the intersection of all circuits.
    """
    if not circuits:
        return Circuit()

    intersection_edges = circuits[0].tok_pos_edges.copy()

    for circuit in circuits[1:]:
        new_intersection_edges: DefaultDict[str, List[Edge]] = defaultdict(list)

        shared_token_positions = (
            intersection_edges.keys() & circuit.tok_pos_edges.keys()
        )

        for token_position in shared_token_positions:
            edges_a = [str(i) for i in intersection_edges[token_position]]
            edges_b = [str(i) for i in circuit.tok_pos_edges[token_position]]

            intersection = list(set(edges_a) & set(edges_b))
            intersection = [
                edge
                for edge in circuit.tok_pos_edges[token_position]
                if str(edge) in intersection
            ]

            if intersection:
                new_intersection_edges[token_position] = intersection

        intersection_edges = new_intersection_edges

        assert intersection_edges, "The intersection between circuits is empty!"

    return Circuit(tok_pos_edges=intersection_edges)


def update_edge_patch_idx(circuit: Circuit, target_seq_labels: List[str]) -> Circuit:
    """
    Update the sequence index (seq_idx) of all edges in the circuit based on the target sequence labels.

    Args:
        target_seq_labels (List[str]): The target sequence labels to map the edges onto.

    Returns:
        Circuit: a novel circuit with updated sequence labelns
    """
    new_circuit_edges: Dict[str, List[Edge]] = defaultdict(list)

    for tok_pos, edge_list in circuit.tok_pos_edges.items():
        try:
            tok_idx = target_seq_labels.index(tok_pos)
        except ValueError:
            logging.warning(
                f"Token: {tok_pos} is not in target sequence: \n {target_seq_labels}\nSkipping modules at position!"
            )
            continue

        for edge in edge_list:
            new_circuit_edges[tok_pos].append(replace(edge, seq_idx=tok_idx))

    return Circuit(tok_pos_edges=new_circuit_edges)


def circuit_soft_intersection(circuits: List[Circuit], overlap: float = 1.0) -> Circuit:
    """
    Compute the intersection of a list of circuits.

    This function takes a list of Circuit objects and returns a new Circuit object
    that contains only the edges that are present in all circuits.

    Args:
        circuits (List[Circuit]): The list of circuits to intersect.
        overlap (float, Optional): A value in [0, 1] specifying the fraction of circuits in which an
                         edge must appear to be included in the intersection.

    Returns:
        Circuit: The circuit that is the intersection of all circuits.
    """
    assert (
        overlap >= 0.0 and overlap <= 1.0
    ), f"Overlap must be a float in the range [0, 1], not: {overlap}."

    if not circuits:
        return Circuit()

    # minimum number of circuits an edge must appear in to be included
    min_count = max(1, int(overlap * len(circuits)))

    # initialize the edge counts and mappings
    edge_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    edge_mappings: Dict[str, Dict[str, Edge]] = defaultdict(lambda: defaultdict(Edge))

    # Aggregate counts of each edge
    for circuit in circuits:
        for token_pos, edges in circuit.tok_pos_edges.items():
            for edge in edges:
                edge_counts[token_pos][edge.name] += 1
                edge_mappings[token_pos][edge.name] = edge

    # filter edges based on overlap criteria
    intersection_edges: Dict[str, List[Edge]] = DefaultDict(List)
    for token_pos, counts in edge_counts.items():
        filtered_edges = [
            edge_mappings[token_pos][edge_name]
            for edge_name, count in counts.items()
            if count >= min_count
        ]
        if filtered_edges:
            intersection_edges[token_pos] = filtered_edges

    assert (
        intersection_edges
    ), f"The intersection between circuits is empty: {intersection_edges}!"

    return Circuit(tok_pos_edges=intersection_edges)


def edges_from_soft_intersection_no_token_pos(
    circuits: List[Circuit], overlap: float = 1.0
) -> List[Edge]:
    """
    Compute the edges resulting from the soft intersection of circuits when ignoring token positions.


    Args:
        circuits (List[Circuit]): The list of circuits to compute the intersection for.
        overlap (float, optional): A value in [0, 1] specifying the fraction of circuits
                                   in which an edge must appear to be included. Defaults to 1.0.

    Returns:
        Circuit: A new Circuit object containing the intersected edges.
    """
    if not (0.0 <= overlap <= 1.0):
        raise ValueError(
            f"Overlap must be a float in the range [0, 1], not: {overlap}."
        )

    if not circuits:
        return []

    # minimum number of circuits an edge must appear in to be included
    min_count = max(1, int(overlap * len(circuits)))

    # Aggregate edge counts across circuits
    edge_counts: Dict[str, int] = defaultdict(int)
    edge_mappings: Dict[str, Edge] = {}

    for circuit in circuits:
        seen_edges: Set[str] = set()
        for edges in circuit.tok_pos_edges.values():
            for edge in edges:
                if edge.name not in seen_edges:
                    edge_counts[edge.name] += 1
                    edge_mappings[edge.name] = edge
                    seen_edges.add(edge.name)

    # filter edges based on the overlap criteria
    filtered_edges: List[Edge] = [
        edge_mappings[edge_name]
        for edge_name, count in edge_counts.items()
        if count >= min_count
    ]

    if not filtered_edges:
        raise ValueError("The intersection between circuits is empty.")

    return filtered_edges


def intersection_over_union(circuit_list: List[Circuit], token_pos: bool) -> float:
    """
    Calculate the Intersection over Union (IoU) for a list of circuits.

    Args:
        circuit_list (List[Circuit]): The list of circuits to evaluate.
        token_pos (bool): If True, considers token positions in the intersection and union calculation.

    Returns:
        float: The IoU value, representing the ratio of the intersection size to the union size.

    Raises:
        ValueError: If the union size is zero, indicating no overlap between circuits.
    """
    if token_pos:
        try:
            circuit_intersection = circuit_soft_intersection(
                circuits=circuit_list, overlap=1.0
            )
            intersection_size = circuit_intersection.size
        except AssertionError:
            intersection_size = 0

        circuit_union = circuit_soft_intersection(circuits=circuit_list, overlap=0.0)
        union_size = circuit_union.size
    else:
        edges_intersection = edges_from_soft_intersection_no_token_pos(
            circuits=circuit_list, overlap=1.0
        )
        intersection_size = len(edges_intersection)

        edges_union = edges_from_soft_intersection_no_token_pos(
            circuits=circuit_list, overlap=0.0
        )
        union_size = len(edges_union)

    if union_size > 0:
        return intersection_size / union_size

    raise ValueError("Size of union is 0!")


def intersection_over_minimum(circuit_list: List[Circuit], token_pos: bool) -> float:
    """
    Calculate the Intersection over Minimum (IoU) for a list of circuits.

    Args:
        circuit_list (List[Circuit]): The list of circuits to evaluate.
        token_pos (bool): If True, considers token positions in the intersection and union calculation.

    Returns:
        float: The IoU value, representing the ratio of the intersection size to the union size.

    Raises:
        ValueError: If the union size is zero, indicating no overlap between circuits.
    """
    if token_pos:
        try:
            circuit_intersection = circuit_soft_intersection(
                circuits=circuit_list, overlap=1.0
            )
            intersection_size = circuit_intersection.size
        except AssertionError:
            intersection_size = 0

        minimum_size = min([circuit.size for circuit in circuit_list])
    else:
        edges_intersection = edges_from_soft_intersection_no_token_pos(
            circuits=circuit_list, overlap=1.0
        )
        intersection_size = len(edges_intersection)

        minimum_size = min(
            [
                len(
                    edges_from_soft_intersection_no_token_pos(
                        [circuit, circuit], overlap=1.0
                    )
                )
                for circuit in circuit_list
            ]
        )

    if minimum_size > 0:
        return intersection_size / minimum_size

    raise ValueError("Size of the minimum is 0!")


def prepare_input_and_patches(
    model: PatchableModel,
    batch: PromptDataLoader,
    patch_type: PatchType = PatchType.EDGE_PATCH,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> Tuple[t.Tensor, t.Tensor]:
    """Prepare the input and patches for the model evaluation.

    Args:
        model (PatchableModel): The model to run
        batch (PromptDataLoader): The batch to use for input and patches
        patch_type (PatchType): Whether to patch the circuit or the complement.
        ablation_type (AblationType): The type of ablation to use.

    Returns:
        A tuple containing the input tensor and the source ablations tensor.
    """
    if patch_type == PatchType.TREE_PATCH:
        batch_input = batch.clean
        if not ablation_type.mean_over_dataset:
            patch_src_outs = src_ablations(model, batch.corrupt, ablation_type)
    elif patch_type == PatchType.EDGE_PATCH:
        batch_input = batch.corrupt
        if not ablation_type.mean_over_dataset:
            patch_src_outs = src_ablations(model, batch.clean, ablation_type)
    else:
        raise NotImplementedError

    return batch_input, patch_src_outs


def eval_circuit(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    circuit: Circuit,
    metric: Literal["correct", "diff", "kl"],
    patch_type: PatchType = PatchType.EDGE_PATCH,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> CircuitOutputs:
    """Eval the model using a specified circuit and ablation type.

    Args:
        model (PatchableModel): The model to run.
        dataloader (PromptDataLoader): The dataloader to use for input and patches.
        circuit (Circuit): The circuit to use for evaluation.
        metric (Literal["correct", "diff", "kl"]): Metric to consider.
        patch_type (PatchType): Whether to patch the circuit or the complement.
        ablation_type (AblationType): The type of ablation to use.

    Returns:
        A dictionary mapping from the number of pruned edges to a
            [`BatchOutputs`][auto_circuit.types.BatchOutputs] object, which is a
            dictionary mapping from [`BatchKey`s][auto_circuit.types.BatchKey] to output
            tensors.
    """
    circ_outs: CircuitOutputs = defaultdict(dict)

    patch_src_outs: Optional[t.Tensor] = None
    if ablation_type.mean_over_dataset:
        patch_src_outs = src_ablations(model, dataloader, ablation_type)

    for batch_idx, batch in enumerate(batch_pbar := tqdm(dataloader)):
        batch_pbar.set_description_str(f"Pruning Batch {batch_idx}", refresh=True)

        batch_input, patch_src_outs = prepare_input_and_patches(
            model, batch, patch_type, ablation_type
        )

        assert patch_src_outs is not None

        patch_edge_count = 0

        modules_to_edges = defaultdict(list)
        for edge_list in circuit.tok_pos_edges.values():
            for edge in edge_list:
                modules_to_edges[edge.dest.module_name].append(edge)

        with patch_mode(model, patch_src_outs):

            for mod_name, edge_list in modules_to_edges.items():
                dest = module_by_name(model, mod_name)
                assert isinstance(dest, PatchWrapper)
                assert dest.is_dest and dest.patch_mask is not None
                if patch_type == PatchType.EDGE_PATCH:
                    for edge in edge_list:
                        dest.patch_mask.data[edge.patch_idx] = 1.0
                    patch_edge_count += dest.patch_mask.int().sum().item()
                else:
                    assert patch_type == PatchType.TREE_PATCH
                    dest.patch_mask.data = t.ones_like(dest.patch_mask.data).float()
                    for edge in edge_list:
                        dest.patch_mask.data[edge.patch_idx] = 0.0
                    patch_edge_count += (1 - dest.patch_mask.int()).sum().item()

            with t.inference_mode():
                model_output = model(batch_input)[model.out_slice]
            circ_outs[patch_edge_count][batch.key] = model_output.detach().clone()

    del patch_src_outs

    n_edge, metric_value = faithfulness_score(model, dataloader, circ_outs, metric)

    return n_edge, metric_value
