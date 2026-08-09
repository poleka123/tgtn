# methods/forward.py
import torch


def model_forward(model, x, data_set, device):
    """
    统一 forward 入口：
    - requires_graph=False（TuckerGODE）: model(x)
    - requires_graph=True（Step 4 GCN/GAT）: model(x, adj)
    """
    if getattr(model, "requires_graph", False):
        adj = data_set.get("adj")
        if adj is None:
            raise ValueError(
                f"Model '{model.__class__.__name__}' requires graph, "
                f"but data_set['adj'] is None. Set --graph_type correlation/identity/distance."
            )
        adj = adj.to(device)
        return model(x, adj)
    return model(x)