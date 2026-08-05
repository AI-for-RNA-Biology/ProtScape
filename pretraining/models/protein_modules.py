import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_mean
from .custom_gnn import ACM_RandomWalk


def add_virtual_nodes_to_batch(x, edge_index, batch, vn_embeddings, num_graphs, vn_ids=None):
    """
    Add virtual nodes to a batched graph. Each graph gets its own VN connected to all its nodes.
    
    Args:
        x: [N, d] node features
        edge_index: [2, E] edges
        batch: [N] graph assignment for each node
        vn_embeddings: [n_celltypes, d] learnable VN embeddings (one per cell type)
        num_graphs: number of graphs in batch (must equal n_celltypes)
    
    Returns:
        x_with_vn: [N + num_graphs, d] node features with VN appended
        edge_index_with_vn: [2, E + 2*N] edges including VN connections
        vn_indices: [num_graphs] indices of VN nodes in the new graph
        num_real_nodes: N (to separate proteins from VNs later)
    """
    N = x.size(0)
    device = x.device
    
    # One VN embedding per graph
    if vn_ids is None:
        vn_features = vn_embeddings[:num_graphs]  # [num_graphs, d]
    else:
        vn_ids = vn_ids.view(-1).to(dtype=torch.long, device=vn_embeddings.device)
        vn_features = vn_embeddings[vn_ids]  # [num_graphs, d]
    
    #Add VN nodes to feature matrix
    x_with_vn = torch.cat([x, vn_features], dim=0)  # [N + num_graphs, d]
    
    # VN indices: N, N+1, ..., N+num_graphs-1
    vn_indices = torch.arange(N, N + num_graphs, device=device)
    
    # Create edges connecting each VN to all nodes in its graph
    # For each node i in graph g, add edges (vn_g, i) and (i, vn_g)
    vn_for_each_node = vn_indices[batch]  # [N] - which VN each node connects to
    node_indices = torch.arange(N, device=device)
    
    # Edges from VN to nodes and nodes to VN
    vn_to_nodes = torch.stack([vn_for_each_node, node_indices], dim=0)
    nodes_to_vn = torch.stack([node_indices, vn_for_each_node], dim=0)  
    
    # concat with original edges
    edge_index_with_vn = torch.cat([edge_index, vn_to_nodes, nodes_to_vn], dim=1)
    
    return x_with_vn, edge_index_with_vn, vn_indices, N


class prot_module(nn.Module):
    """
    Protein Graph Neural Network (GNN) module supporting multiple GNN architectures.

    This module provides a flexible interface for applying various GNN layers (GATv2, GIN, ACM_RandomWalk)
    to protein-protein interaction (PPI) graphs, with configurable depth, hidden dimensions, normalization,
    activation, and jumping knowledge strategies.

    Args:
        protein_config (dict): Configuration dictionary with the following keys:
            - 'input_dim' (int): Input feature dimension.
            - 'hidden_dim' (int): Hidden layer dimension for GNN layers.
            - 'n_layers' (int): Number of GNN layers.
            - 'gnn_method' (str): GNN type ('GATv2', 'GIN', or 'ACM_RandomWalk').
            - 'gnn_dropout' (float): Dropout rate for GATv2 layers.
            - 'gnn_batchnorm' (bool): Whether to use batch normalization.
            - 'gnn_activation' (str): Activation function ('relu' or 'leaky_relu').
            - 'jumping_knowledge' (str): Jumping knowledge strategy ('concat', 'concat_full', 'max', or None).
            - 'gat_heads' (int, optional): Number of attention heads for GATv2 (required if 'GATv2').
        device (str): Device to use ('cpu' or 'cuda').

    Attributes:
        gnn_prot_layers (nn.ModuleList): List of GNN layers.
        gnn_batchnorm_layers (nn.ModuleList, optional): List of batch normalization layers.
        gnn_output_dim (int): Output feature dimension after all GNN layers and jumping knowledge.
        protein_config (dict): Configuration dictionary.

    Methods:
        reset_parameters(): Resets all learnable parameters.
        forward(ppi_x, ppi_edge_index, batching=False): Applies the GNN to input features and edge indices.
            - If batching=False: expects dictionaries for cell-type specific PPIs.
            - If batching=True: expects a single batched PPI object.

    Example:
        >>> config = {
        ...     'input_dim': 64,
        ...     'hidden_dim': 128,
        ...     'n_layers': 3,
        ...     'gnn_method': 'GATv2',
        ...     'gnn_dropout': 0.1,
        ...     'gnn_batchnorm': True,
        ...     'gnn_activation': 'relu',
        ...     'jumping_knowledge': 'concat',
        ...     'gat_heads': 4
        ... }
        >>> model = prot_module(config, device='cuda')
        >>> out = model(ppi_x, ppi_edge_index)
    """
    
    def __init__(
        self,
        protein_config,
        device,
        graph_saint_norm=False):
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
        for m in self.gnn_prot_layers:
            m.reset_parameters()
        if self.gnn_batchnorm_layers is not None:
            for m in self.gnn_batchnorm_layers:
                m.reset_parameters()
        
        
    def _mp_edge_index(self, edge_index, y):
        # If y is missing / mismatched, fall back to using all edges.
        if y is None:
            return edge_index
        y = y.view(-1)
        if y.numel() != edge_index.size(1):
            return edge_index
        return edge_index[:, y.to(dtype=torch.bool)]


    def _mp_edge_weight(self, edge_weight, y):
        # If edge_weight is None or y is missing / mismatched, fall back to using all edges.
        if edge_weight is None or y is None:
            return edge_weight
        y = y.view(-1)
        if y.numel() != edge_weight.shape[0]:
            return edge_weight
        return edge_weight[y.to(dtype=torch.bool)]

    def forward_batch_graph(self, batch_graph, return_layer_outputs=False):
        """
        Forward pass through the protein GNN for a single graph.
        """
        x = batch_graph.x

        edge_index = getattr(batch_graph, "edge_index_mp", None)
        edge_weight = getattr(batch_graph, "edge_weight", None)
        
        if edge_index is None:
            edge_index = batch_graph.edge_index
        if edge_weight is None:
            edge_weight = batch_graph.edge_weight
            
        y = getattr(batch_graph, "y", None)
        edge_index = self._mp_edge_index(edge_index, y)
        edge_weight = self._mp_edge_weight(edge_weight, y)
        
        # Initialize output lists for jumping knowledge
        outs = [x] if self.gnn_jumping_knowledge == 'concat_full' else []
        layer_outputs = [] if return_layer_outputs else None
        
        # GNN forward pass
        for i, layer in enumerate(self.gnn_prot_layers):
            x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            
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
    
    
    def forward_original(self, ppi_x, ppi_edge_index, batching=False, return_layer_outputs=False):
        """
        Forward pass through the protein GNN.
        
        Returns:
            - If use_virtual_node=False: protein embeddings (dict or tensor)
            - If use_virtual_node=True: tuple of (protein_embeddings, vn_embeddings)
              where vn_embeddings is the cell embedding from the virtual node
        """
        # i'm not found of this ppi_metapaths terminology in our case
        # where essentially it coincides with the ppi_edge_index
        # maybe things could be coded more efficiently by factoring every
        # cell-type specific PPI as a batch
        raise 'to update with handling vn, s2gae and so on'
        if not batching:
            # Non-batched mode: iterate through cell types
            # Note: VN mode requires batching=True for proper batch tensor
            if self.use_virtual_node:
                raise ValueError("Virtual node mode requires batching=True. "
                                 "Set n_cells in cell_config to enable batching.")

            layer_outputs_dict = {} if return_layer_outputs else None
            
            for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers
                edge_index_full = ppi_edge_index[celltype]['total_edge_index']
                y_full = ppi_edge_index[celltype].get('y', None)
                edge_index_mp = self._mp_edge_index(edge_index_full, y_full)

                if self.gnn_jumping_knowledge == 'concat_full':
                    outs = [x]
                else:
                    outs = []
                layer_outs = [] if return_layer_outputs else None
                # iterate through GNN layers
                for i, layer in enumerate(self.gnn_prot_layers):
                    if i == 0:
                        out = layer(x, edge_index=edge_index_mp)
                    else:
                        out = layer(out, edge_index=edge_index_mp)

                    if self.gnn_method != 'ACM_RandomWalk':
                        if self.gnn_batchnorm:
                            out = self.gnn_batchnorm_layers[i](out)
                        if self.gnn_activation == 'relu':
                            out = F.relu(out)
                        elif self.gnn_activation == 'leaky_relu':
                            out = F.leaky_relu(out, negative_slope=0.2)
                    if self.gnn_dropout > 0:
                        out = F.dropout(out, p=self.gnn_dropout, training=self.training)
                        
                    outs.append(out)
                    if return_layer_outputs:
                        layer_outs.append(out)
                # Jumping knowledge
                if self.gnn_jumping_knowledge in ['concat', 'concat_full']:
                    ppi_x[celltype] = torch.cat(outs, dim=-1)
                elif self.gnn_jumping_knowledge == 'max':
                    ppi_x[celltype] = torch.max(torch.stack(outs, dim=-1), dim=-1)[0]
                else:
                    ppi_x[celltype] = outs[-1] 

                if return_layer_outputs:
                    layer_outputs_dict[celltype] = layer_outs

            if return_layer_outputs:
                return ppi_x, layer_outputs_dict
            return ppi_x
        
        else:  # batching=True: ppi_x is a Batch object
            return self.forward_batch_graph(ppi_x, return_layer_outputs=return_layer_outputs)

    
    def forward_factored(self, ppi_data, batching=False, return_layer_outputs=False):
        """
        Forward pass through the protein GNN.
        """
        
        if not batching:
            assert isinstance(ppi_data, dict), \
                "Factored input mode requires ppi_data to be a dict of pytorch geometric Data objects."
           
            layer_outputs_dict = {} if return_layer_outputs else None
            
            for celltype, graph in ppi_data.items(): # Iterate through cell-type specific PPI layers
                edge_index_raw = getattr(graph, "edge_index_mp", None)
                edge_weight_raw = None
                if edge_index_raw is None:
                    edge_index_raw = graph.edge_index
                
                # Check for graph_saint_norm attribute safely
                gn_norm = getattr(self, 'graph_saint_norm', False)
                if gn_norm:
                    edge_weight_raw = graph.edge_weight
                    
                edge_index_mp = self._mp_edge_index(edge_index_raw, getattr(graph, "y", None))
                edge_weight_mp = self._mp_edge_weight(edge_weight_raw, getattr(graph, "y", None))
                if self.gnn_jumping_knowledge == 'concat_full':
                    outs = [graph.x]
                else:
                    outs = []
                layer_outs = [] if return_layer_outputs else None
                # iterate through GNN layers
                for i, layer in enumerate(self.gnn_prot_layers):
                    if i == 0:
                        out = layer(graph.x, edge_index=edge_index_mp, edge_weight=edge_weight_mp)
                    else:
                        out = layer(out, edge_index=edge_index_mp, edge_weight=edge_weight_mp)

                    if self.gnn_method != 'ACM_RandomWalk':
                        if self.gnn_batchnorm:
                            out = self.gnn_batchnorm_layers[i](out)
                        if self.gnn_activation == 'relu':
                            out = F.relu(out)
                        elif self.gnn_activation == 'leaky_relu':
                            out = F.leaky_relu(out, negative_slope=0.2)
                    
                    if self.gnn_dropout > 0:
                        out = F.dropout(out, p=self.gnn_dropout, training=self.training)
                        
                    outs.append(out)
                    if return_layer_outputs:
                        layer_outs.append(out)
                # Jumping knowledge
                if self.gnn_jumping_knowledge in ['concat', 'concat_full']:
                    ppi_data[celltype] = torch.cat(outs, dim=-1)
                elif self.gnn_jumping_knowledge == 'max':
                    ppi_data[celltype] = torch.max(torch.stack(outs, dim=-1), dim=-1)[0]
                else:
                    ppi_data[celltype] = outs[-1] 

                if return_layer_outputs:
                    layer_outputs_dict[celltype] = layer_outs

            if return_layer_outputs:
                return ppi_data, layer_outputs_dict
            else:
                return ppi_data 
        
        else:  # batching=True: ppi_x is a Batch object
            return self.forward_batch_graph(ppi_data, return_layer_outputs=return_layer_outputs)


    def forward(self, ppi_x, ppi_edge_index=None, batching=False, return_layer_outputs=False):
        """
        Forward pass through the protein GNN.
        
        Returns:
            - If use_virtual_node=False: protein embeddings (dict or tensor)
            - If use_virtual_node=True: tuple of (protein_embeddings, vn_embeddings)
              where vn_embeddings is the cell embedding from the virtual node
        """
        # Backwards compatible calling patterns:
        # - forward(ppi_x_dict, ppi_edge_index_dict, ...)  [original]
        # - forward((ppi_x_dict, ppi_edge_index_dict), ...) [original]
        # - forward(ppi_dict_of_Data, ...) [factored]
        # - forward(batch_graph, ...) [batched]
        if isinstance(ppi_x, tuple):
            ppi_x, ppi_edge_index = ppi_x

        if ppi_edge_index is not None:
            return self.forward_original(ppi_x, ppi_edge_index, batching=batching, return_layer_outputs=return_layer_outputs)
        
        else:
            return self.forward_factored(ppi_x, batching=batching, return_layer_outputs=return_layer_outputs)
        
