import torch

import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_link_pred_loss(mg_pred, mg_y, ppi_preds, ppi_y, loss_type="BCE", version="standard"):

    # Calculate link prediction loss on metagraph
    mg_loss = 0
    if not mg_pred is None:
        if len(mg_pred) > 0:
            mg_loss = F.binary_cross_entropy(mg_pred, mg_y["y"].to(device))

    # Calculate link prediction loss on PPI networks
    assert ppi_preds is not None
    ppi_loss = 0
    n_cells = 0
    for celltype, edge_preds in ppi_preds.items():
        if version=='standard':
            ppi_loss += F.binary_cross_entropy(edge_preds, ppi_y[celltype]["y"].to(device), reduction='mean')
        elif version == 'factored':
            ppi_loss += F.binary_cross_entropy(edge_preds, ppi_y[celltype], reduction='mean')

        n_cells += 1
    ppi_loss /= n_cells
    return ppi_loss, mg_loss


def calc_center_loss(center_loss, embed, centers, y, mask):
    loss = center_loss(embed[mask, :], centers, y[mask].to(device))
    return loss


def calc_uniformity_loss(embed, t=2.0):
    z = F.normalize(embed, p=2, dim=-1)
    return torch.log(torch.exp(2.0 * float(t) * ((z @ z.T) - 1.0)).mean())


def max_margin_loss(pred, y):
    loss = (1 - (pred[y == 1] - pred[y != 1])).clamp(min=0)
    return loss


def el_dot(embed, edges, relation): 
    """
    Calculate the dot product for edge prediction.
    Args:
        embed (torch.Tensor): Node embeddings.
        edges (torch.Tensor): Edge indices.
        relation (torch.Tensor): Relation embeddings.
    Returns:
        torch.Tensor: Sigmoid of the dot product.
    """
    source = embed[edges[0, :]]
    target = embed[edges[1, :]]
    if len(relation) != 0:
        #print('Using relation embeddings for edge prediction')
        #print('relation shape:', relation.shape)
        dots = torch.sum(source * relation * target, dim = 1)
    else:
        dots = torch.sum(source * target, dim = 1)
    return torch.sigmoid(dots)
