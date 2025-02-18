import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer


RELEVANT_TOKENS = [
    "[equals]_occ_1",
    "[space_after_eq]_occ_1",
    "[z1]_occ_1",
    "[z1-first]_occ_1",
    "[z1-second]_occ_1",
]


CLASSES = range(10)


class ActivationProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dtype: t.dtype = t.bfloat16):
        super().__init__()
        self.classes = output_dim
        self.dtype = dtype
        self.linear = nn.Linear(input_dim, output_dim).to(dtype=self.dtype)

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        logits = self.linear(x)
        return nn.functional.log_softmax(logits, dim=-1)

    def train_probe(self, data_loader: DataLoader, epochs=1, lr=1e-3):
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for _ in range(epochs):
            for inputs, labels_batch in data_loader:
                inputs = inputs.to(device).to(dtype=self.dtype)
                labels_batch = labels_batch.to(device)
                optimizer.zero_grad()
                logits = self(inputs)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

    def test_probe(self, data_loader: DataLoader):
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.to(device)
        predictions = []
        all_labels = []
        with t.no_grad():
            for inputs, labels_batch in data_loader:
                inputs = inputs.to(device).to(dtype=self.dtype)
                labels_batch = labels_batch.to(device)
                outputs = self(inputs)
                preds = t.argmax(outputs, dim=-1)
                predictions.append(preds.cpu())
                all_labels.append(labels_batch.cpu())
        predictions = t.cat(predictions, dim=0)
        all_labels = t.cat(all_labels, dim=0)
        return predictions, all_labels


def probe_residual_components(
    model_name: str,
    model: HookedTransformer,
    train_batches_with_seq_labels: List[Tuple[t.Tensor, List[str]]],
    test_batches_with_seq_labels: List[Tuple[t.Tensor, List[str]]],
    circuit_tokens: List[str],
    save_dir: str,
) -> None:
    """
    Probes the residual stream of a transformer model to predict the result of
    the computation contained in the math input prompt.
    The probing accuracy of the model layers at different token positions is
    then plotted using a heatmap.

    Args:
        model_name (str): The name of the model.
        model (HookedTransformer): The transformer model to probe.
        train_loader (List[Tuple[t.Tensor, List[str]]]): The training batches paired with seq_labels for each template.
        test_loader (List[Tuple[t.Tensor, List[str]]]): The testing batches paired with seq_labels for each template.
        circuit_tokens (List[str]): The tokens that are part of the circuit.
        save_dir (str): The directory to save the results.
        template (str): The template to use for the probing.
    """

    token_labels = get_relevant_tokens_labels(
        train_batches_with_seq_labels[0][1], circuit_tokens
    )

    os.makedirs(f"{save_dir}/activations", exist_ok=True)
    if not os.path.exists(f"{save_dir}/activations/resid_acts_{model_name}.pt"):
        residual_activations = collect_residual_activations_on_dataset(
            model_name=model_name,
            model=model,
            circuit_tokens=circuit_tokens,
            train_batches_with_seq_labels=train_batches_with_seq_labels,
            test_batches_with_seq_labels=test_batches_with_seq_labels,
            save_dir=f"{save_dir}/activations",
        )
    else:
        residual_activations = t.load(
            f"{save_dir}/activations/resid_acts_{model_name}.pt"
        )

    train_acts, train_labels = residual_activations["train"]
    test_acts, test_labels = residual_activations["test"]
    classes = CLASSES

    results = {}
    for layer_name, train_layer_acts in train_acts.items():
        test_layer_acts = test_acts[layer_name]
        _, pos_dim, h_dim = train_layer_acts.shape
        accs = []
        for pos in range(pos_dim):
            X_train = train_layer_acts[:, pos, :]
            X_test = test_layer_acts[:, pos, :]
            train_dl = prepare_dataloader(X_train, train_labels)
            test_dl = prepare_dataloader(X_test, test_labels)
            probe = ActivationProbe(h_dim, len(classes), dtype=train_layer_acts.dtype)
            probe.train_probe(train_dl, epochs=1, lr=1e-3)
            preds, labs = probe.test_probe(test_dl)
            accs.append((preds == labs).float().mean().item())

        results[layer_name] = accs

    plot_dir = f"{save_dir}/accuracies/"
    os.makedirs(plot_dir, exist_ok=True)
    plot_heatmap(
        results, token_labels, f"{plot_dir}/probing_accuracies_{model_name}.png"
    )


def collect_residual_activations_on_dataset(
    model_name: str,
    model: HookedTransformer,
    circuit_tokens: List[str],
    train_batches_with_seq_labels: List[Tuple[t.Tensor, List[str]]],
    test_batches_with_seq_labels: List[Tuple[t.Tensor, List[str]]],
    save_dir: str,
) -> Dict[str, Tuple[Dict[str, t.Tensor], t.Tensor]]:
    """
    Collects the residual activations of the transformer model on the training and testing data.

    Args:
        model_name (str): The name of the model.
        model (HookedTransformer): The transformer model to probe.
        circuit_tokens (List[str]): The tokens that are part of the circuit.
        train_batches_with_seq_labels (List[Tuple[t.Tensor, List[str]]]): The training batches paired with seq_labels for each template.
        test_batches_with_seq_labels (List[Tuple[t.Tensor, List[str]]]): The testing batches paired with seq_labels for each template.
        save_dir (str): The directory to save the results.

    Returns:
        Dict[str, Tuple[Dict[str, t.Tensor], t.Tensor]]: The residual activations and labels for the training and testing data.
    """

    model.eval()
    results: Dict[str, Tuple[Dict[str, t.Tensor], t.Tensor]] = {}

    for split, batches_with_labels in [
        ("train", train_batches_with_seq_labels),
        ("test", test_batches_with_seq_labels),
    ]:

        all_acts: Dict[str, List[t.Tensor]] = {
            f"blocks.{layer}.hook_resid_post": [] for layer in range(model.cfg.n_layers)
        }
        all_labels = []
        for template, (batches, seq_labels) in enumerate(batches_with_labels):

            relevant_tokens_idxes = get_relevant_tokens_indexes(
                seq_labels, circuit_tokens
            )

            for inpt in batches:
                target = extract_computation_result(model, inpt)
                with t.no_grad():
                    _, cache = model.run_with_cache(inpt)
                for layer in range(model.cfg.n_layers):
                    key = f"blocks.{layer}.hook_resid_post"
                    resid_activations = cache[key].cpu()
                    all_acts[key].append(resid_activations[:, relevant_tokens_idxes])
                all_labels.append(target.cpu())

        final_acts = {k: t.cat(v, dim=0) for k, v in all_acts.items()}
        final_labels = t.cat(all_labels, dim=0)

        results[split] = (final_acts, final_labels)

    t.save(results, f"{save_dir}/resid_acts_{model_name}.pt")

    return results


def prepare_dataloader(
    x_data: t.Tensor, y_data: t.Tensor, batch_size=256
) -> DataLoader:
    """
    Prepares a DataLoader for the given data.

    Args:
        x_data (t.Tensor): The input data.
        y_data (t.Tensor): The target data.
        batch_size (int): The batch size.

    Returns:
        DataLoader: The DataLoader for the given data.
    """
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def extract_computation_result(model: HookedTransformer, inpt: t.Tensor) -> t.Tensor:
    """
    Extracts the result of the computation from the input prompt.

    Args:
        model (HookedTransformer): The transformer model.
        inpt (t.Tensor): The input prompt.

    Returns:
        t.Tensor: The result of the computation.
    """
    pattern = r"(\d+)[^\d+\-*/]*(?:[+\-*/])[^\d+\-*/]*(\d+)"
    results = []
    full_input = [model.to_string(x) for x in inpt]

    for input_str in full_input:
        match = re.search(pattern, input_str)
        assert (
            match is not None
        ), f"Pattern '[num1] + [num2] =' not found in input: {input_str}"

        computed = str(int(match.group(1)) + int(match.group(2)))
        computed = computed[-1]

        results.append(int(computed))

    return t.tensor(results)


def get_relevant_tokens_indexes(
    seq_labels: List[str], circuit_tokens: List[str]
) -> List[int]:
    """
    Get the indexes of the relevant tokens in the sequence labels.

    Args:
        seq_labels (List[str]): The sequence labels.
        circuit_tokens (List[str]): The tokens that are part of the circuit.

    Returns:
        List[int]: The indexes of the relevant tokens.
    """
    relevant_tokens_idxes = [
        idx
        for idx, token in enumerate(seq_labels)
        if token in RELEVANT_TOKENS or token in circuit_tokens
    ]
    return relevant_tokens_idxes


def get_relevant_tokens_labels(
    seq_labels: List[str], circuit_tokens: List[str]
) -> List[str]:
    """
    Get the labels of the relevant tokens in the sequence labels.

    Args:
        seq_labels (List[str]): The sequence labels.
        circuit_tokens (List[str]): The tokens that are part of the circuit.

    Returns:
        List[str]: The labels of the relevant tokens.
    """
    relevant_tokens_labels = [
        re.sub(r"_occ_\d", "", token)
        for token in seq_labels
        if token in RELEVANT_TOKENS or token in circuit_tokens
    ]
    return relevant_tokens_labels


def plot_heatmap(
    result: Dict[str, List[float]], token_labels: List[str], save_path: str
) -> None:
    """
    Plots a heatmap of the probing accuracies.

    Args:
        result (Dict[str, List[float]]): The probing accuracies.
        token_labels (List[str]): The token labels.
        save_path (str): The path to save the plot.
    """
    layers = list(result.keys())[::-1]
    layer_labels = [f"{layer.split('.')[1]}" for layer in layers]
    accuracies = [result[layer] for layer in layers]

    plt.figure(figsize=(10, 8))
    plt.imshow(accuracies, aspect="auto", cmap="plasma", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar()
    cbar.set_label("Linear Probe Accuracy", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.xticks(ticks=range(len(token_labels)), labels=token_labels, fontsize=14, rotation=45)
    plt.yticks(ticks=range(len(layers)), labels=layer_labels, fontsize=14)
    plt.xlabel("Token Position", fontsize=16)
    plt.ylabel("Layer", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
