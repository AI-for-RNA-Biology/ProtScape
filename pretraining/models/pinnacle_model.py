"""Adapted PINNACLE protein/cell/tissue model.

Contextual AI models for single-cell protein biology, Nature Methods (2024):
https://doi.org/10.1038/s41592-024-02341-3
Original code: https://github.com/mims-harvard/PINNACLE

Fixes: local protein-feature ordering, bidirectional PPI edges, protein-pair
splits shared across cell types, and negatives excluding supplied positive
edges, their reverses and self-loops. ESM-2 inputs, ACM/GIN layers and shared
PPI weights are optional extensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm

from .pinnacle_conv import PCTConv, PPIConv


class Pinnacle(nn.Module):
    """Two protein–cell–tissue message-passing rounds with relation-wise decoding."""
    def __init__(
        self,
        gnn_method,
        nfeat,
        hidden,
        output,
        num_ppi_relations,
        num_mg_relations,
        ppi_data,
        n_heads,
        pc_att_channels,
        dropout = 0.2,
        shared_ppi_gnn = False,
        device='cuda:0'):
        """Create two graph layers; GAT concatenates heads, while GIN/ACM use output dimensions directly."""
        super(Pinnacle, self).__init__()

        self.gnn_method = gnn_method
        self.dropout = dropout

        # Layer dimensions
        self.layer1_in = nfeat
        self.layer1_out = hidden
        self.layer2_in = hidden if n_heads is None else self.layer1_out * n_heads 
        self.layer2_out = output
        self.output = self.layer2_out if n_heads is None else self.layer2_out * n_heads
        self.shared_ppi_gnn = shared_ppi_gnn
        
        # Complete layer #1
        self.conv1_up = PCTConv(
            self.layer1_in,
            num_ppi_relations,
            num_mg_relations,
            ppi_data,
            self.gnn_method,
            self.layer1_out,
            sem_att_channels=8,
            pc_att_channels=pc_att_channels,
            node_heads=n_heads,
            shared_ppi_gnn=shared_ppi_gnn,
            device=device
            )
        self.conv1_down = PPIConv(
            hidden if n_heads is None else self.layer1_out * n_heads ,
            num_ppi_relations,
            self.layer1_out,
            ppi_data,
            self.gnn_method,
            sem_att_channels=8,
            node_heads=n_heads,
            shared_ppi_gnn=shared_ppi_gnn,
            device=device
            )

        # Normalization
        self.layer_norm1 = LayerNorm(self.layer2_in)
        self.batch_norm1 = BatchNorm(self.layer2_in)

        # Complete layer #2
        self.conv2_up = PCTConv(
            self.layer2_in,
            num_ppi_relations,
            num_mg_relations,
            ppi_data,
            self.gnn_method,
            self.layer2_out,
            sem_att_channels=8,
            pc_att_channels=pc_att_channels,
            node_heads=n_heads,
            shared_ppi_gnn=shared_ppi_gnn,
            device=device
            )
        self.conv2_down = PPIConv(
            self.output,
            num_ppi_relations,
            self.layer2_out,
            ppi_data,
            self.gnn_method,
            sem_att_channels=8,
            node_heads=n_heads,
            shared_ppi_gnn=shared_ppi_gnn,
            device=device
            )

        # Metagraph decoder
        self.mg_relw = nn.Parameter(torch.Tensor(num_mg_relations, int(self.output)))
        nn.init.xavier_uniform_(self.mg_relw, gain = nn.init.calculate_gain('leaky_relu'))


    def forward(
        self, 
        ppi_x,
        mg_x,
        ppi_metapaths, 
        mg_metapaths, 
        ppi_edge_index, 
        mg_edge_index, 
        tissue_neighbors):
        """Apply two rounds of protein-to-cell pooling and cell-to-protein updates."""
        # Layer 1

        # Update Protein-Celltype-Tissue
        ppi_x, mg_x = self.conv1_up(
            ppi_x, mg_x, ppi_metapaths, mg_metapaths,
            ppi_edge_index, mg_edge_index, tissue_neighbors, init_cci=True)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv1_down(
            ppi_x, ppi_metapaths, mg_x, self.conv1_up.ppi_attn)

        # Apply Leaky ReLU, dropout, and normalization.
        for celltype, x in ppi_x.items():
            ppi_x[celltype] = self.layer_norm1(x)
            ppi_x[celltype] = F.leaky_relu(ppi_x[celltype])
            ppi_x[celltype] = self.batch_norm1(ppi_x[celltype])
            ppi_x[celltype] = F.dropout(ppi_x[celltype], p = self.dropout, training = self.training)
            
        mg_x = self.layer_norm1(mg_x)
        mg_x = F.leaky_relu(mg_x)
        mg_x = self.batch_norm1(mg_x)
        mg_x = F.dropout(mg_x, p = self.dropout, training = self.training)

        # Layer 2

        # Update Protein-Celltype-Tissue
        ppi_x, mg_x = self.conv2_up(
            ppi_x, mg_x, ppi_metapaths, mg_metapaths,
            ppi_edge_index, mg_edge_index, tissue_neighbors)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv2_down(
            ppi_x, ppi_metapaths, mg_x, self.conv2_up.ppi_attn)

        return ppi_x, mg_x
