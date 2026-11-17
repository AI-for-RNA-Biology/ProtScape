import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.nn.inits import glorot, zeros
from .custom_gnn import ACM_RandomWalk

def build_convolution(method, input_dim, output_dim, heads, device):
    """Build one PINNACLE graph layer; GAT concatenates its attention heads."""
    if method == "GATv2":
        return GATv2Conv(input_dim, output_dim, heads)
    if method == "GIN":
        return GINConv(nn.Linear(input_dim, output_dim))
    if method == "ACM_RandomWalk":
        return ACM_RandomWalk(
            input_dim, output_dim, gnn_activation="relu", use_batchnorm=True,
            T=3., device=device,
        )
    raise ValueError(f"Unknown PINNACLE graph layer: {method}")

def attend_relations(x, relation_edges, convolution, projection, bias, query):
    """Convolve each nonempty relation, then attention-pool relations per node.

    Each edge tensor has shape (2, edges). The result has one feature vector
    per node; attention is normalized across relations, not across nodes.
    """
    features = [convolution(x, edges) for edges in relation_edges if edges.shape[1] > 0]
    features = F.leaky_relu(torch.stack(features, dim=1).to(x.device))
    scores = torch.sum(projection * features.unsqueeze(-1), dim=-2) + bias
    scores = torch.sum(query * torch.tanh(scores), dim=-1)
    weights = torch.softmax(scores, dim=1)
    return torch.sum(features * weights.unsqueeze(-1), dim=1)

class PCTConv(nn.Module):
    """Update proteins, pool them into cell nodes, and propagate through tissues."""
    def __init__(
        self,
        in_channels,
        num_ppi_relations,
        num_mg_relations,
        ppi_data,
        gnn_method,
        out_channels,
        sem_att_channels,
        pc_att_channels,
        node_heads=3,
        tissue_update = 100,
        shared_ppi_gnn=False,
        device='cuda:0'
        ):
        """Build protein and metagraph convolutions with shared semantic attention."""
        super().__init__()

        self.ppi_data = ppi_data
        self.in_channels = in_channels
        self.num_ppi_relations = num_ppi_relations
        self.num_mg_relations = num_mg_relations
        self.out_channels = out_channels
        self.gnn_method = gnn_method
        self.sem_att_channels = sem_att_channels
        self.node_heads = node_heads
        self.tissue_update = tissue_update
        self.shared_ppi_gnn = shared_ppi_gnn

        # Cell-type specific PPI weights
        self.ppi_attn = dict()

        output_dimension = out_channels * node_heads if gnn_method == "GATv2" else out_channels
        # Use one layer per cell type, or one shared layer for the shared-GNN variant.
        self.ppi_w = nn.ModuleList([
            build_convolution(gnn_method, in_channels, out_channels, node_heads, device)
            for _ in range(1 if shared_ppi_gnn else len(ppi_data))
        ])
        self.mg_conv_in = build_convolution(gnn_method, in_channels, out_channels, node_heads, device)
        self.mg_conv_out = build_convolution(gnn_method, output_dimension, out_channels, node_heads, device)

        # Semantic attention (shared across networks)
        self.W = nn.Parameter(torch.Tensor(1, 1, output_dimension, sem_att_channels))
        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        nn.init.xavier_uniform_(self.W, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.b, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.q, gain = nn.init.calculate_gain('leaky_relu'))

        # Protein - Cell type attention (shared across networks)
        self.pc_W = nn.Parameter(torch.Tensor(1, 1, output_dimension, pc_att_channels))
        self.pc_b = nn.Parameter(torch.Tensor(1, 1, pc_att_channels))
        self.pc_q = nn.Parameter(torch.Tensor(1, 1, pc_att_channels))
        nn.init.xavier_uniform_(self.pc_W, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.pc_b, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.pc_q, gain = nn.init.calculate_gain('leaky_relu'))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the shared relation-attention projection and query."""
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)

    def forward(
        self,
        ppi_x,
        mg_x,
        ppi_metapaths,
        mg_metapaths,
        ppi_edge_index,
        mg_edge_index,
        tissue_neighbors,
        init_cci=False):
        """Pool proteins into cell nodes, update tissues, then convolve the metagraph."""

        if init_cci:
            mg_x_list = []

        else: # Project metagraph embeddings to the same dimension as PPI
            mg_x = attend_relations(mg_x, mg_metapaths, self.mg_conv_in, self.W, self.b, self.q)

        for celltype, x in ppi_x.items():
            convolution = self.ppi_w[0 if self.shared_ppi_gnn else celltype]
            ppi_x[celltype] = attend_relations(
                x, ppi_metapaths[celltype], convolution, self.W, self.b, self.q,
            )

            # Attention on PPI nodes per cell type
            w = torch.sum(self.pc_W * ppi_x[celltype].unsqueeze(-1), dim=-2) + self.pc_b
            w = torch.tanh(w)
            gamma = torch.sum(self.pc_q * w, dim=-1)
            gamma = torch.softmax(gamma, dim=1)
            self.ppi_attn[celltype] = gamma.squeeze(0)

            if init_cci: # Initialize CCI embeddings using PPI embeddings
                weighted_x = torch.sum(ppi_x[celltype] * self.ppi_attn[celltype].unsqueeze(-1), dim=0)
                mg_x_list.append(weighted_x)
            else: # Update CCI embeddings
                mg_x[celltype, :] += torch.sum(ppi_x[celltype] * self.ppi_attn[celltype].unsqueeze(-1), dim=0)

        if init_cci: # Concatenate initialized metagraph embeddings
            mg_x = torch.stack(mg_x_list)
            bto = torch.zeros(
                len(tissue_neighbors),
                mg_x.shape[1],
                device=mg_x.device,
                dtype=mg_x.dtype,
            )
            mg_x = torch.cat((mg_x, torch.normal(bto, std=1)))

        # Smooth tissue embeddings over their neighbors before the learned update.
        for _ in range(self.tissue_update):
            for t in sorted(tissue_neighbors):
                assert len(tissue_neighbors[t]) != 0
                mg_x[t, :] = torch.mean(mg_x[tissue_neighbors[t]], 0)

        mg_x = attend_relations(mg_x, mg_metapaths, self.mg_conv_out, self.W, self.b, self.q)

        return ppi_x, mg_x

class PPIConv(nn.Module):
    """Update each PPI graph and add its attention-weighted cell embedding."""
    def __init__(
        self,
        in_channels,
        num_ppi_relations,
        out_channels,
        ppi_data,
        gnn_method,
        sem_att_channels,
        node_heads=3,
        shared_ppi_gnn=False,
        device='cuda:0'):
        """Create per-cell or shared protein layers and their relation-attention parameters."""
        super().__init__()
        self.in_channels = in_channels
        self.num_ppi_relations = num_ppi_relations
        self.out_channels = out_channels
        self.gnn_method = gnn_method
        self.sem_att_channels = sem_att_channels
        self.node_heads = node_heads
        self.shared_ppi_gnn = shared_ppi_gnn

        output_dimension = out_channels * node_heads if gnn_method == "GATv2" else out_channels
        self.ppi_w = nn.ModuleList([
            build_convolution(gnn_method, in_channels, out_channels, node_heads, device)
            for _ in range(1 if shared_ppi_gnn else len(ppi_data))
        ])

        # Define semantic-level attention parameters
        self.W = nn.Parameter(torch.Tensor(1, 1, output_dimension, sem_att_channels))
        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        nn.init.xavier_uniform_(self.W, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.b, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.q, gain = nn.init.calculate_gain('leaky_relu'))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the shared relation-attention projection and query."""
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)

    def forward(self, ppi_x, ppi_metapaths, mg_x, ppi_attn):
        """Return protein features by cell ID, using the pooling weights from PCTConv."""

        for celltype, x in ppi_x.items():
            convolution = self.ppi_w[0 if self.shared_ppi_gnn else celltype]
            ppi_x[celltype] = attend_relations(
                x, ppi_metapaths[celltype], convolution, self.W, self.b, self.q,
            )

            # Downpool using cell-type embedding
            gamma = ppi_attn[celltype] # (n,) where n = number of proteins in the cell type (in the batch)
            ppi_x[celltype] += (mg_x[celltype, :].repeat(len(gamma), 1) * gamma.unsqueeze(-1))

        return ppi_x
