import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from .custom_gnn import ACM_RandomWalk


class prot_module(nn.Module):
    """Encode cell-specific PPI graphs with shared GAT, GIN or ACM layers.

    Layer outputs can be concatenated for protein embeddings and S2GAE decoding."""

    def __init__(
        self,
        protein_config,
        device,
        graph_saint_norm=False):
        """Build shared protein GNN layers and the selected Jumping Knowledge aggregation."""
        super().__init__()

        # Instantiate protein module
        print('protein_config:', protein_config)
        self.protein_config = protein_config
        self.input_dim = protein_config['input_dim']
        self.device = device
        self.graph_saint_norm = graph_saint_norm

        self.gnn_method =  protein_config['gnn_method']
        assert self.gnn_method in ['GATv2', 'GIN', 'ACM_RandomWalk']
        if self.graph_saint_norm:
            assert self.gnn_method == 'ACM_RandomWalk'

        self.gnn_hidden_dim = protein_config['hidden_dim']
        self.gnn_n_layers = protein_config['n_layers']
        self.gnn_dropout = protein_config['gnn_dropout']
        self.gnn_batchnorm = protein_config['gnn_batchnorm']

        self.gnn_activation = protein_config['gnn_activation']
        assert self.gnn_activation in ['relu', 'leaky_relu']

        self.gnn_jumping_knowledge = protein_config['jumping_knowledge']

        # Virtual node option (default False)
        self.n_cells = protein_config.get('n_cells', None)  # Number of cell types for VN

        # Initialize GNN layers based on the specified method
        if self.gnn_method == 'GATv2':
            self.gnn_gat_heads = protein_config['gat_heads']
            local_gnn_hidden_dim = self.gnn_hidden_dim * self.gnn_gat_heads
            base_layers = [
                GATv2Conv(self.input_dim if i == 0 else local_gnn_hidden_dim,
                          self.gnn_hidden_dim,
                          heads=self.gnn_gat_heads,
                          dropout=0.)
                for i in range(self.gnn_n_layers)
            ]
        elif self.gnn_method == 'GIN':
            local_gnn_hidden_dim = self.gnn_hidden_dim
            base_layers = [
                GINConv(nn.Linear(self.input_dim if i == 0 else local_gnn_hidden_dim,
                                  self.gnn_hidden_dim))
                for i in range(self.gnn_n_layers)
            ]
        elif self.gnn_method == 'ACM_RandomWalk':
            assert self.gnn_activation in ['relu', 'leaky_relu']

            local_gnn_hidden_dim = self.gnn_hidden_dim
            base_layers = [
                ACM_RandomWalk(
                    self.input_dim if i==0 else local_gnn_hidden_dim,
                    self.gnn_hidden_dim,
                    gnn_activation=self.gnn_activation,
                    use_batchnorm=self.gnn_batchnorm,
                    T=3.,
                    graph_saint_norm=graph_saint_norm,
                    device=device)
                for i in range(self.gnn_n_layers)
            ]

        self.local_gnn_hidden_dim = local_gnn_hidden_dim

        # Initialize batch normalization layers if requested
        if self.gnn_batchnorm and (self.gnn_method != 'ACM_RandomWalk'):
            self.gnn_batchnorm_layers = nn.ModuleList([
                nn.BatchNorm1d(local_gnn_hidden_dim) for _ in range(self.gnn_n_layers)
            ])
        else:
            self.gnn_batchnorm_layers = None

        # Use base layers directly
        self.gnn_prot_layers = nn.ModuleList(base_layers)

        if self.gnn_jumping_knowledge == 'concat':
            self.gnn_output_dim = local_gnn_hidden_dim * self.gnn_n_layers
        elif self.gnn_jumping_knowledge == 'concat_full':
            self.gnn_output_dim = local_gnn_hidden_dim * self.gnn_n_layers + self.input_dim
        else:
            self.gnn_output_dim = local_gnn_hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the protein convolutions and Jumping Knowledge module."""
        for m in self.gnn_prot_layers:
            m.reset_parameters()
        if self.gnn_batchnorm_layers is not None:
            for m in self.gnn_batchnorm_layers:
                m.reset_parameters()


    def encode_graph(self, graph, return_layer_outputs=False, use_edge_weights=True):
        """Run the shared GNN on a single graph or a PyG batch."""
        x = graph.x
        edge_index = getattr(graph, "edge_index_mp", None)
        if edge_index is None:
            edge_index = graph.edge_index
        edge_weight = getattr(graph, "edge_weight", None) if use_edge_weights else None

        # Labels filter sampled negatives. Virtual-node edges have no matching labels.
        labels = getattr(graph, "y", None)
        if labels is not None:
            positive = labels.view(-1).to(dtype=torch.bool)
            if positive.numel() == edge_index.size(1):
                edge_index = edge_index[:, positive]
            if edge_weight is not None and positive.numel() == edge_weight.shape[0]:
                edge_weight = edge_weight[positive]

        # Initialize output lists for jumping knowledge
        outs = [x] if self.gnn_jumping_knowledge == 'concat_full' else []
        layer_outputs = [] if return_layer_outputs else None

        # GNN forward pass
        for i, layer in enumerate(self.gnn_prot_layers):
            if self.gnn_method == "ACM_RandomWalk":
                x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            else:
                x = layer(x, edge_index=edge_index)

            # Batchnorm + activation (for non-ACM)
            if self.gnn_method != 'ACM_RandomWalk':
                if self.gnn_batchnorm:
                    x = self.gnn_batchnorm_layers[i](x)
                if self.gnn_activation == 'relu':
                    x = F.relu(x)
                elif self.gnn_activation == 'leaky_relu':
                    x = F.leaky_relu(x, negative_slope=0.2)

            if self.gnn_dropout > 0:
                x = F.dropout(x, p=self.gnn_dropout, training=self.training)

            # Store outputs for jumping knowledge
            outs.append(x)
            if return_layer_outputs:
                layer_outputs.append(x)

        # Apply jumping knowledge (proteins only)
        if self.gnn_jumping_knowledge in ['concat', 'concat_full']:
            new_x = torch.cat(outs, dim=-1)
        elif self.gnn_jumping_knowledge == 'max':
            new_x = torch.max(torch.stack(outs, dim=-1), dim=-1)[0]
        else:
            new_x = outs[-1]


        if return_layer_outputs:
            return new_x, layer_outputs
        return new_x


    def forward(self, ppi_x, batching=False, return_layer_outputs=False):
        """Encode a PyG batch or a dictionary of cell-specific graphs."""
        if batching:
            return self.encode_graph(ppi_x, return_layer_outputs=return_layer_outputs)

        layer_outputs = {}
        for celltype, graph in ppi_x.items():
            output = self.encode_graph(
                graph,
                return_layer_outputs=return_layer_outputs,
                use_edge_weights=self.graph_saint_norm,
            )
            if return_layer_outputs:
                ppi_x[celltype], layer_outputs[celltype] = output
            else:
                ppi_x[celltype] = output
        return (ppi_x, layer_outputs) if return_layer_outputs else ppi_x
