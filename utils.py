import torch
from pathlib import Path

import torch






def build_structural_features(triples, num_entities, num_relations):
    """
    Build non-random node features from graph structure.

    Features:
    - in-degree
    - out-degree
    - total degree
    - outgoing relation histogram
    - incoming relation histogram

    Output:
        x shape = [num_entities, 3 + 2 * num_relations]
    """

    device = triples.device

    heads = triples[:, 0]
    relations = triples[:, 1]
    tails = triples[:, 2]

    out_degree = torch.zeros(num_entities, 1, device=device)
    in_degree = torch.zeros(num_entities, 1, device=device)

    out_degree.index_add_(
        0,
        heads,
        torch.ones_like(heads, dtype=torch.float).view(-1, 1)
    )

    in_degree.index_add_(
        0,
        tails,
        torch.ones_like(tails, dtype=torch.float).view(-1, 1)
    )

    total_degree = in_degree + out_degree

    out_rel_hist = torch.zeros(num_entities, num_relations, device=device)
    in_rel_hist = torch.zeros(num_entities, num_relations, device=device)

    out_rel_hist[heads, relations] += 1
    in_rel_hist[tails, relations] += 1

    x = torch.cat(
        [
            in_degree,
            out_degree,
            total_degree,
            out_rel_hist,
            in_rel_hist
        ],
        dim=1
    )

    # Log transform to reduce scale differences
    x = torch.log1p(x)

    # Normalize each feature column
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-12
    x = (x - mean) / std

    return x

    
def save_model(model, path, optimizer=None, epoch=None, metadata=None):
    """
    Save a PyTorch or PyTorch Geometric model.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance to save.

    path : str or Path
        File path where the checkpoint will be saved.

    optimizer : torch.optim.Optimizer, optional
        Optimizer to save if you want to resume training.

    epoch : int, optional
        Current training epoch.

    metadata : dict, optional
        Extra information to store.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {}
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    torch.save(checkpoint, path)

    print(f"Model saved to: {path}")


def load_model(model, path, optimizer=None, map_location=None, eval_mode=True):
    """
    Load weights directly into an existing PyTorch or PyTorch Geometric model.

    Parameters
    ----------
    model : torch.nn.Module
        Existing model instance.

    path : str or Path
        Path to the saved checkpoint.

    optimizer : torch.optim.Optimizer, optional
        Existing optimizer instance. If provided, its state will also be loaded.

    map_location : str or torch.device, optional
        Device mapping, for example "cpu" or "cuda".

    eval_mode : bool
        If True, set model to evaluation mode after loading.

    Returns
    -------
    model : torch.nn.Module
        Model with loaded weights.

    optimizer : torch.optim.Optimizer or None
        Optimizer with loaded state, if provided.

    checkpoint : dict
        Full checkpoint dictionary.
    """

    checkpoint = torch.load(
        path,
        map_location=map_location
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if map_location is not None:
        model = model.to(map_location)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if eval_mode:
        model.eval()

    return model, optimizer, checkpoint