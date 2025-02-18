"""
Tests for intersection code.
"""

from typing import List

import pytest
from auto_circuit.types import DestNode, Edge, SrcNode

from llm_error_detection.circuits.circuit_analysis import (
    Circuit,
    circuit_intersection,
    circuit_soft_intersection,
)


@pytest.fixture
def edges() -> List[Edge]:
    """
    Provide a set of reusable Edge objects for testing.
    """
    src1 = SrcNode(name="nodeA", module_name="MLP 0", layer=0)
    dest1 = DestNode(name="nodeB", module_name="A15.7.K", layer=30)
    dest2 = DestNode(name="nodeC", module_name="A13.2.Q", layer=26)
    dest3 = DestNode(name="nodeD", module_name="MLP 13", layer=26)

    return [
        Edge(src=src1, dest=dest1),
        Edge(src=src1, dest=dest2),
        Edge(src=src1, dest=dest3),
    ]


@pytest.fixture
def empty_circuits() -> List[Circuit]:
    """
    Provide an empty list of Circuit objects for testing.
    """
    return []


@pytest.fixture
def single_circuit(edges: List[Edge]) -> Circuit:
    """
    Provide a single Circuit object for testing.
    """
    return Circuit(tok_pos_edges={"pos1": [edges[0], edges[1]], "pos2": [edges[2]]})


@pytest.fixture
def sample_circuits(edges: List[Edge]) -> List[Circuit]:
    """
    Provide a list of sample Circuit objects for testing.
    """
    return [
        Circuit(tok_pos_edges={"pos1": [edges[0], edges[1]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos1": [edges[0]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos1": [edges[0]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos1": [edges[0]], "pos2": [edges[2]]}),
    ]


def test_empty_circuit_list(empty_circuits: List[Circuit]):
    """
    Test that the function returns an empty Circuit when the input list is empty.
    """
    result = circuit_soft_intersection(empty_circuits, overlap=0.5)
    assert result.tok_pos_edges == {}, "Expected empty circuit for empty input list"


def test_single_circuit(single_circuit: Circuit):
    """
    Test that the function returns the same circuit when there's only one circuit in the input.
    """
    result = circuit_soft_intersection([single_circuit], overlap=0.5)
    assert (
        result.tok_pos_edges == single_circuit.tok_pos_edges
    ), "Expected the result to match the single input circuit"


def test_single_circuit_full_overlap(single_circuit: Circuit):
    """
    Test that the function returns the same circuit when there's only one circuit in the input.
    """
    result = circuit_soft_intersection([single_circuit], overlap=1.0)
    assert (
        result.tok_pos_edges == single_circuit.tok_pos_edges
    ), "Expected the result to match the single input circuit with overlap=1.0"


def test_full_overlap(sample_circuits: List[Circuit], edges: List[Edge]):
    """
    Test with overlap=1.0 to ensure the function behaves as a strict intersection.
    """
    result = circuit_soft_intersection(sample_circuits, overlap=1.0)
    assert result.tok_pos_edges == {
        "pos1": [edges[0]],
        "pos2": [edges[2]],
    }, "Expected strict intersection with overlap=1.0"


def test_partial_overlap(sample_circuits: List[Circuit], edges: List[Edge]):
    """
    Test with overlap=0.5 to ensure edges appearing in at least half the circuits are included.
    """
    result = circuit_soft_intersection(sample_circuits, overlap=0.5)
    assert result.tok_pos_edges == {
        "pos1": [edges[0]],
        "pos2": [edges[2]],
    }, "Expected edges appearing in at least 50% of circuits"


def test_low_overlap(sample_circuits: List[Circuit], edges: List[Edge]):
    """
    Test with overlap=0.33 (approximately 1/3) to ensure most edges are included.
    """
    result = circuit_soft_intersection(sample_circuits, overlap=0.33)
    assert result.tok_pos_edges == {
        "pos1": [edges[0], edges[1]],
        "pos2": [edges[2]],
    }, "Expected edges appearing in at least one circuit with overlap=0.33"


def test_invalid_overlap(sample_circuits: List[Circuit]):
    """
    Test that the function raises an AssertionError for invalid overlap values.
    """
    with pytest.raises(
        AssertionError, match="Overlap must be a float in the range \\[0, 1\\]."
    ):
        circuit_soft_intersection(sample_circuits, overlap=-0.1)

    with pytest.raises(
        AssertionError, match="Overlap must be a float in the range \\[0, 1\\]."
    ):
        circuit_soft_intersection(sample_circuits, overlap=1.1)


def test_no_common_edges(edges: List[Edge]):
    """
    Test when circuits have no common edges to ensure the result is empty.
    """
    circuits = [
        Circuit(tok_pos_edges={"pos1": [edges[0]]}),
        Circuit(tok_pos_edges={"pos1": [edges[1]]}),
        Circuit(tok_pos_edges={"pos1": [edges[2]]}),
        Circuit(tok_pos_edges={"pos2": [edges[2]]}),
    ]
    with pytest.raises(
        AssertionError, match="The intersection between circuits is empty!"
    ):
        circuit_soft_intersection(circuits, overlap=0.5)


def test_missing_token_positions(edges: List[Edge]):
    """
    Test when circuits have mismatched token positions to ensure proper handling.
    """
    circuits = [
        Circuit(tok_pos_edges={"pos1": [edges[0]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos2": [edges[0]], "pos3": [edges[2]]}),
        Circuit(tok_pos_edges={"pos2": [edges[1]], "pos3": [edges[2]]}),
        Circuit(tok_pos_edges={"pos2": [edges[1]], "pos4": [edges[2]]}),
    ]
    with pytest.raises(
        AssertionError, match="The intersection between circuits is empty!"
    ):
        circuit_soft_intersection(circuits, overlap=0.75)


def test_soft_intersection_vs_strict_intersection_all_shared(edges: List[Edge]):
    """
    Test that the soft intersection function with overlap=1.0 produces the same result
    as the strict intersection function when all circuits have common edges.
    """
    # Create circuits with fully shared edges
    circuits = [
        Circuit(tok_pos_edges={"pos1": [edges[0], edges[1]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos1": [edges[0], edges[1]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos1": [edges[0], edges[1]], "pos2": [edges[2]]}),
    ]

    # Perform strict intersection
    strict_result = circuit_intersection(circuits)

    # Perform soft intersection with overlap=1.0
    soft_result = circuit_soft_intersection(circuits, overlap=1.0)

    # Compare results
    assert (
        strict_result.tok_pos_edges == soft_result.tok_pos_edges
    ), "Soft intersection with overlap=1.0 did not match strict intersection."


def test_soft_intersection_vs_strict_intersection_partial_shared(edges: List[Edge]):
    """
    Test that the soft intersection function with overlap=1.0 produces the same result
    as the strict intersection function when only some edges are shared across all circuits.
    """
    # Create circuits with partially shared edges
    circuits = [
        Circuit(tok_pos_edges={"pos1": [edges[0], edges[1]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos1": [edges[0]], "pos2": [edges[2]]}),
        Circuit(tok_pos_edges={"pos1": [edges[0]], "pos2": [edges[2]]}),
    ]

    # Perform strict intersection
    strict_result = circuit_intersection(circuits)

    # Perform soft intersection with overlap=1.0
    soft_result = circuit_soft_intersection(circuits, overlap=1.0)

    # Compare results
    assert (
        strict_result.tok_pos_edges == soft_result.tok_pos_edges
    ), "Soft intersection with overlap=1.0 did not match strict intersection when edges were partially shared."
