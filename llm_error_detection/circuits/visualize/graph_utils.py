"""
Utility functions for plotting graphs.
"""

import colorsys
import os
from io import BytesIO
from typing import Dict, List, Optional

import pygraphviz as pgv
from auto_circuit.types import Edge, Node
from PIL import Image, ImageDraw, ImageFont

from llm_error_detection.circuits.circuit_analysis import Circuit

# Define colors for node types
NODE_TYPE_COLORS = {
    "Resid Start": "#e6f7ff",
    "MLP_in": "#8cd98c",
    "MLP_out": "#77b300",
    "Head.Q": "#e6b800",
    "Head.K": "#ff5c33",
    "Head.V": "#ffb3d9",
    "Head.O": "#e6ccff",
    "Resid End": "#66ccff",
    "Unknown": "#ffffff",
}


def get_node_name(node: Node) -> str:
    """
    Get name for a node in the circuit.

    Args:
        node (Node): The node to generate a name for.

    Returns:
        str: The generated name.
    """
    node_name = node.name

    if "mlp" in node.module_name:
        if "mlp_in" in node.module_name:
            node_name = node_name.replace("MLP", "MLP_in")
        elif "mlp_out" in node.module_name:
            node_name = node_name.replace("MLP", "MLP_out")
    elif "attn" in node.module_name:
        if "W_O" in node.weight:
            node_name += ".O"

    return node_name


def get_node_type(node_name: str) -> str:
    """
    Determine the type of a node based on its name.

    Args:
        node_name (str): The name of the node.

    Returns:
        str: The type of the node (e.g., "MLP", "Head.Q").
    """
    if node_name.startswith("MLP_in"):
        return "MLP_in"
    elif node_name.startswith("MLP_out"):
        return "MLP_out"
    elif ".Q" in node_name:
        return "Head.Q"
    elif ".K" in node_name:
        return "Head.K"
    elif ".V" in node_name:
        return "Head.V"
    elif ".O" in node_name:
        return "Head.O"
    elif "Resid Start" in node_name:
        return "Resid Start"
    elif "Resid End" in node_name:
        return "Resid End"
    return "Unknown"


def get_node_color(node_name: str) -> str:
    """
    Get node color based on the node's name.

    Args:
        node_name (str): The name of the node

    Returns:
        str: The respective color.
    """
    node_type = get_node_type(node_name)
    return NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS["Unknown"])


def get_edge_color(layer: int, total_layers: int = 10) -> str:
    """
    Generate a consistent color for edges based on the layer index using HSL.

    Args:
        layer (int): The layer number of the source node.
        total_layers (int): The total number of layers, used to determine the color spacing.

    Returns:
        str: A HEX color string.
    """
    # Normalize layer to a value between 0 and 1
    hue = (
        layer % total_layers
    ) / total_layers  # Cycle colors if layers exceed total_layers
    saturation = 0.7  # Fixed saturation for vibrant colors
    lightness = 0.5  # Fixed lightness for balanced colors

    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    # Scale RGB values to 0-255 and format as HEX
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def create_graph_structure(
    edges: List[Edge],
    minimum_penwidth: float = 0.3,
    layout: str = "dot",
    remove_self_loops: bool = True,
    node_pos: Optional[Dict[str, str]] = None,
    num_layers: int = 28,
) -> pgv.AGraph:
    """
    Constructs a PyGraphviz AGraph structure for visualization from a list of edges.

    Args:
        edges (List[Edge]): A list of Edge objects defining the graph.
        colors (Dict[str, str]): A dictionary mapping node names to their respective colors.
        minimum_penwidth (float): Minimum width for edges.
        layout (str): Layout algorithm for the graph (e.g., 'dot', 'neato').
        remove_self_loops (bool): Whether to remove edges that loop back to the same node.
        node_pos (Optional[Dict[str, str]]): Precomputed positions for nodes (if available).
        num_layers (int): Number of layers in the graph.

    Returns:
        pgv.AGraph: The constructed graph object.
    """
    # Initialize the graph
    g = pgv.AGraph(
        directed=True,
        bgcolor="transparent",
        overlap="false",
        splines="true",
        layout=layout,
    )

    layer_nodes: Dict[int, List[str]] = {}  # Group nodes by layer

    for edge in edges:
        # Extract source and destination node names
        src_name = edge.src.name
        dest_name = edge.dest.name

        # Skip self-loops if required
        if remove_self_loops and src_name == dest_name:
            continue

        src_node = edge.src
        dest_node = edge.dest

        # Add nodes to the graph and group by layer
        for i, node in enumerate([src_node, dest_node]):
            node_name = get_node_name(node)
            node_color = get_node_color(node_name)
            layer = node.layer

            if i == 0:
                src_name = node_name
            else:
                dest_name = node_name

            if layer not in layer_nodes:
                layer_nodes[layer] = []
            if node_name not in layer_nodes[layer]:
                layer_nodes[layer].append(node_name)

            g.add_node(
                node_name,
                fillcolor=node_color,
                color="black",
                style="filled, rounded",
                shape="box",
                fontname="Helvetica",
                pos=node_pos.get(node_name) if node_pos else None,
            )

        # Add edge to the graph
        edge_color = get_edge_color(src_node.layer, 2 * num_layers + 2)
        g.add_edge(
            src_name,
            dest_name,
            penwidth=str(max(minimum_penwidth, 1.0)),
            color=edge_color,
        )

    # Ensure nodes are ranked by layer from bottom to top
    sorted_layers = list(range(2 * num_layers + 2))

    # Add dummy nodes for consistent global alignment
    prev_dummy_node = None

    for layer in sorted_layers:
        dummy_node = f"dummy_{layer}"
        g.add_node(dummy_node, shape="none", width=0, height=0, label="")
        nodes_to_add = [dummy_node]

        if layer in layer_nodes:
            nodes_to_add += layer_nodes[layer]

        g.add_subgraph(
            nodes_to_add,
            rank="same",
        )

        # Connect dummy nodes to enforce alignment
        if prev_dummy_node:
            g.add_edge(
                prev_dummy_node,
                dummy_node,
                style="invis",  # Invisible edge
            )
        prev_dummy_node = dummy_node

    # Set rank direction for proper bottom-to-top alignment
    g.graph_attr.update(rankdir="BT")

    return g


def plot_circuit(
    circuit: Circuit,
    file_path: str,
    tok_pos_name: str,
    minimum_penwidth: float = 0.6,
    layout: str = "dot",
    remove_self_loops: bool = True,
    node_pos: Optional[Dict[str, str]] = None,
    num_layers: int = 28,
) -> None:
    """
    Plot a circuit using a PyGraphviz AGraph structure.

    Args:
        circuit (Circuit): The Circuit object to be plotted.
        file_path (str): The path where the image should be stored.
        minimum_penwidth (float): Minimum width for edges.
        layout (str): Layout algorithm for the graph (e.g., 'dot', 'neato').
        remove_self_loops (bool): Whether to remove edges that loop back to the same node.
        node_pos (Optional[Dict[str, str]]): Precomputed positions for nodes (if available).
        num_layers (int): Number of layers in the graph.
    """
    graph = create_graph_structure(
        edges=circuit.tok_pos_edges[tok_pos_name],
        minimum_penwidth=minimum_penwidth,
        layout=layout,
        remove_self_loops=remove_self_loops,
        node_pos=node_pos,
        num_layers=num_layers,
    )
    graph.layout(prog=layout)

    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    graph.draw(file_path)


def plot_circuits_for_all_positions(
    circuit: Circuit,
    file_path: str,
    minimum_penwidth: float = 0.6,
    layout: str = "dot",
    remove_self_loops: bool = True,
    node_pos: Optional[Dict[str, str]] = None,
    num_layers: int = 28,
) -> None:
    """
    Plot circuits for all token positions and arrange them horizontally with labels.

    Args:
        circuit (Circuit): The Circuit object to be plotted.
        file_path (str): The path where the combined image should be stored.
        minimum_penwidth (float): Minimum width for edges.
        layout (str): Layout algorithm for the graph (e.g., 'dot', 'neato').
        remove_self_loops (bool): Whether to remove edges that loop back to the same node.
        node_pos (Optional[Dict[str, str]]): Precomputed positions for nodes (if available).
        num_layers (int): Number of layers in the graph.
    """
    images = []
    labels = []

    # Render each token position as a graph and collect images and labels
    for token_pos, edges in circuit.tok_pos_edges.items():
        graph = create_graph_structure(
            edges=edges,
            minimum_penwidth=minimum_penwidth,
            layout=layout,
            remove_self_loops=remove_self_loops,
            node_pos=node_pos,
            num_layers=num_layers,
        )
        # Create an in-memory image of the graph
        graph.layout(prog=layout)
        graph_buffer = BytesIO()
        graph.draw(graph_buffer, format="png")
        graph_buffer.seek(0)
        images.append(Image.open(graph_buffer))
        labels.append(token_pos)

    # Determine the maximum dimensions
    max_width = sum(img.width for img in images)
    max_image_height = max(img.height for img in images)
    label_height = 80  # Height allocated for labels
    combined_image_height = max_image_height + label_height

    # Create a blank combined image with a white background
    combined_image = Image.new("RGB", (max_width, combined_image_height), (0, 0, 0))
    x_offset = 0

    # Load font for labels
    font = ImageFont.load_default(size=32)

    # Draw each graph and its label
    draw = ImageDraw.Draw(combined_image)
    for img, label in zip(images, labels):
        # Align the image at the bottom
        y_offset = max_image_height - img.height
        combined_image.paste(img, (x_offset, y_offset))

        # Add label below the image
        label_x = x_offset + (img.width - len(label) * 15) // 2
        label_y = max_image_height + (label_height // 2)
        draw.text((label_x, label_y), label, fill="white", font=font)

        x_offset += img.width

    # Extract the directory from the file path
    directory = os.path.dirname(file_path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the combined image
    combined_image.save(file_path)
