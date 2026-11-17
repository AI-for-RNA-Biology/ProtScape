import torch

import torch.nn.functional as F


def calc_link_pred_loss(mg_pred, mg_y, ppi_preds, ppi_y, loss_type="BCE", version="standard"):

    # Calculate link prediction loss on metagraph
    """Return mean per-cell PPI BCE and metagraph BCE."""
    mg_loss = 0
    if not mg_pred is None:
        if len(mg_pred) > 0:
            mg_loss = F.binary_cross_entropy(mg_pred, mg_y["y"].to(mg_pred.device))

    # Calculate link prediction loss on PPI networks
    assert ppi_preds is not None
    ppi_loss = 0
    n_cells = 0
    for celltype, edge_preds in ppi_preds.items():
        if version=='standard':
            ppi_loss += F.binary_cross_entropy(edge_preds, ppi_y[celltype]["y"].to(edge_preds.device), reduction='mean')
        elif version == 'factored':
            ppi_loss += F.binary_cross_entropy(edge_preds, ppi_y[celltype], reduction='mean')

        n_cells += 1
    ppi_loss /= n_cells
    return ppi_loss, mg_loss


def calc_center_loss(center_loss, embed, centers, y, mask):
    """Apply center loss to proteins selected by the supplied node mask."""
    loss = center_loss(embed[mask, :], centers, y[mask].to(embed.device))
    return loss


# Rethinking Graph Masked Autoencoders through Alignment and Uniformity (AUG-MAE).
# https://arxiv.org/abs/2402.07225

def calc_uniformity_loss(embed, t=2.0):
    """Compute log mean pairwise similarity after normalizing protein embeddings."""
    z = F.normalize(embed, p=2, dim=-1)
    return torch.log(torch.exp(2.0 * float(t) * ((z @ z.T) - 1.0)).mean())


def el_dot(embed, edges, relation):
    """Return sigmoid edge scores from node products, optionally weighted by a relation."""
    source = embed[edges[0, :]]
    target = embed[edges[1, :]]
    if len(relation) != 0:
        dots = torch.sum(source * relation * target, dim = 1)
    else:
        dots = torch.sum(source * target, dim = 1)
    return torch.sigmoid(dots)
