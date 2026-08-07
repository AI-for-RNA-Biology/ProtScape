import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.data import Batch, Data
from typing import Optional

from .protein_modules import prot_module
from .cell_modules import cell_module
from .custom_gnn import ACM_RandomWalk
from ..s2gae_utils import S2GAEDecoder, S2GAEDecoderSimple


def add_virtual_node(graph: Data, vn_embedding: torch.Tensor, *, clone: bool = False) -> Data:
    """
    Add a virtual node (VN) as the last node of the graph.

    - Keeps `graph.edge_index` unchanged (used for link prediction).
    - Stores message-passing edges (original + VN connections) in `edge_index_mp`.
    """
    data = graph.clone() if clone else graph

    # avoid appending multiple VNs on the same Data object
    has_vn = getattr(data, "has_vn", None)
    if has_vn is not None:
        try:
            if bool(has_vn):
                return data
        except Exception:
            pass

    if data.x is None:
        raise ValueError("Cannot add a virtual node to a graph with no `x`.")

    device = data.x.device
    vn_embedding = vn_embedding.to(device=device, dtype=data.x.dtype).view(1, -1)

    num_nodes_before = int(data.x.size(0))
    data.x = torch.cat([data.x, vn_embedding], dim=0)  # [N+1, d]
    # fix num_nodes bug
    data.num_nodes = num_nodes_before + 1

    base_edge_index_mp = getattr(data, "edge_index_mp", None)
    if base_edge_index_mp is None:
        base_edge_index_mp = data.edge_index

    node_indices = torch.arange(num_nodes_before, device=device, dtype=torch.long)
    vn_index = torch.full((num_nodes_before,), num_nodes_before, device=device, dtype=torch.long)
    vn_to_nodes = torch.stack([vn_index, node_indices], dim=0)
    nodes_to_vn = torch.stack([node_indices, vn_index], dim=0)
    data.edge_index_mp = torch.cat([base_edge_index_mp, vn_to_nodes, nodes_to_vn], dim=1)
    data.has_vn = torch.tensor(True, device=device)

    return data


def _build_cci_gnn(
    gnn_method: str,
    input_dim: int,
    output_dim: int,
    gat_heads: int,
    gnn_activation: str,
    gnn_batchnorm: bool,
    device: str,
) -> nn.Module:
    if gnn_method == "GATv2":
        heads = gat_heads if gat_heads is not None else 1
        if output_dim % heads != 0:
            heads = 1
        out_channels = output_dim // heads
        return GATv2Conv(input_dim, out_channels, heads=heads, dropout=0.0)
    if gnn_method == "GIN":
        return GINConv(nn.Linear(input_dim, output_dim))
    if gnn_method == "ACM_RandomWalk":
        return ACM_RandomWalk(
            input_dim,
            output_dim,
            gnn_activation=gnn_activation,
            use_batchnorm=gnn_batchnorm,
            device=device,
        )
    raise ValueError(f"Unknown gnn_method={gnn_method!r} for CCI head.")


class CCIGraphHead(nn.Module):
    def __init__(
        self,
        gnn_method: str,
        input_dim: int,
        output_dim: int,
        gat_heads: int,
        gnn_activation: str,
        gnn_batchnorm: bool,
        dropout: float,
        device: str,
    ):
        super().__init__()
        self.gnn = _build_cci_gnn(
            gnn_method=gnn_method,
            input_dim=input_dim,
            output_dim=output_dim,
            gat_heads=gat_heads,
            gnn_activation=gnn_activation,
            gnn_batchnorm=gnn_batchnorm,
            device=device,
        )
        self.dropout = float(dropout)

    def reset_parameters(self):
        if hasattr(self.gnn, "reset_parameters"):
            self.gnn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gnn(x, edge_index)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class hierarchical_model(nn.Module):
    """
    Hierarchical model for contextualized protein-protein interaction (PPI) learning.

    This model implements a multi-level architecture for learning representations from protein,
    cell, and tissue levels. It integrates a protein encoder, a cell encoder, and a tissue classifier,
    supporting hierarchical biological data such as protein-protein interaction graphs across cell types and tissues.

    Args:
        hierarchical_mode (str): Mode for hierarchical modeling (currently supports 'CTassignment').
        protein_config (dict): Configuration dictionary for the protein encoder (prot_module).
        cell_config (dict): Configuration dictionary for the cell encoder (cell_module).
        tissue_config (object): Configuration for the tissue classifier (must have attribute 'n_tissues').
        ppi_data (dict): Dictionary of PPI data for each cell type.
        device (str): Device to use ('cpu' or 'cuda').

    Attributes:
        prot_encoder (prot_module): Protein-level encoder module.
        cell_encoder (cell_module): Cell-level encoder module.
        tissue_encoder (nn.Module): Tissue-level classifier (linear layer).
        hierarchical_mode (str): Selected hierarchical modeling mode.
        protein_config (dict): Protein encoder configuration.
        cell_config (dict): Cell encoder configuration.
        tissue_config (object): Tissue classifier configuration.

    Methods:
        reset_parameters(): Resets all learnable parameters in the model.
        forward(ppi_x, ppi_edge_index, batching=False): 
            Computes protein, cell, and tissue representations.
            - If batching=False: expects dictionaries for cell-type specific PPIs.
            - If batching=True: expects batched PPI data.

    Returns (from forward):
        ppi_x (dict): Updated protein representations per cell type.
        cells_x (torch.Tensor): Cell embeddings.
        cells_pred (torch.Tensor): Tissue-level predictions for multi-label classification.
    """
    def __init__(
        self,
        hierarchical_mode,
        protein_config,
        cell_config,
        tissue_config,
        ppi_data,
        device,
        s2gae_config=None,
        use_metagraph: bool = False,
        graph_saint_norm: bool = False,
        uniformity_dim: int = 0,
        ):
        super(hierarchical_model, self).__init__()
        """
        Hierarchical model for contextualized protein-protein interaction learning
        
        Args:
            s2gae_config (dict, optional): Configuration for S2GAE training mode.
                Keys:
                - 'enabled': bool, whether to use S2GAE training
                - 'mask_ratio': float, ratio of edges to mask (default 0.5)
                - 'decoder_type': str, 'cross_layer' or 'simple' (default 'cross_layer')
                - 'decode_channels': int, hidden dim in decoder MLP (default 256)
                - 'decoder_layers': int, number of MLP layers (default 2)
                - 'decoder_dropout': float, dropout in decoder (default 0.5)
        """

        assert hierarchical_mode in ['CTassignment']
        self.hierarchical_mode = hierarchical_mode
        self.protein_config = protein_config
        self.cell_config = cell_config
        self.tissue_config = tissue_config
        self.use_metagraph = bool(use_metagraph)

        self.input_dim = int(protein_config["input_dim"])
        self.celltype_to_vn_id = {c: i for i, c in enumerate(ppi_data.keys())}
        
        if self.cell_config.get('n_cells') is not None:
            assert len(self.celltype_to_vn_id) == self.cell_config['n_cells'], "cell_config['n_cells'] must match the number of cell types in ppi_data"
            self.n_cells = self.cell_config.get('n_cells')
        else:
            self.n_cells = len(self.celltype_to_vn_id)
        
        # S2GAE configuration
        self.s2gae_config = s2gae_config or {}
        self.s2gae_enabled = self.s2gae_config.get('enabled', False)
        
        # enable graph saint propagation
        self.graph_saint_norm = graph_saint_norm
        
        # Update protein config for S2GAE mode
        if self.s2gae_enabled:
            protein_config['s2gae_mode'] = True
        
        if self.cell_config.get("pooling") == 'vn' or self.protein_config.get("add_virtual_node"):
            self.register_buffer(
                "virtual_node_features",
                torch.zeros(self.input_dim, device=device, requires_grad=False),
            )
        elif self.cell_config.get("pooling") == 'learnedvn':
            if self.n_cells is None:
                raise ValueError("cell_config['n_cells'] must be set when pooling='learnedvn'.")
            self.virtual_node_features = nn.Parameter(torch.zeros((self.n_cells, self.input_dim), device=device))
        else:
            self.virtual_node_features = None

        # instantiate PPI module
        self.prot_encoder = prot_module(protein_config, device, graph_saint_norm)
        
        # instantiate cell module
        self.cell_config['embedding_dim'] = self.prot_encoder.gnn_output_dim
        self.cell_encoder = cell_module(
            cell_config,
            ppi_data,
            device)
        
        # instantiate tissue module
        if hierarchical_mode == 'CTassignment':
            self.tissue_encoder = torch.nn.Linear(
                self.prot_encoder.gnn_output_dim,
                self.tissue_config.n_tissues         
            )

        self.cci_encoder = None
        self.cci_edge_index = None
        self.cci_num_nodes = None
        if self.use_metagraph:
            self.cci_encoder = CCIGraphHead(
                gnn_method=self.protein_config["gnn_method"],
                input_dim=self.prot_encoder.gnn_output_dim,
                output_dim=self.prot_encoder.gnn_output_dim,
                gat_heads=self.protein_config.get("gat_heads"),
                gnn_activation=self.protein_config.get("gnn_activation"),
                gnn_batchnorm=self.protein_config.get("gnn_batchnorm"),
                dropout=self.protein_config.get("gnn_dropout"),
                device=device,
            )

        # relation-specific weights for metagraph decoding (edge types 0-3)
        self.mg_relw = torch.nn.Parameter(
            torch.empty(4, self.prot_encoder.gnn_output_dim, device=device)
        )
        torch.nn.init.xavier_uniform_(
            self.mg_relw, gain=torch.nn.init.calculate_gain('leaky_relu')
        )
        
        # S2GAE decoder for link prediction using cross-layer embeddings
        self.s2gae_decoder = None
        if self.s2gae_enabled:
            decoder_type = self.s2gae_config.get('decoder_type', 'cross_layer')
            n_layers = protein_config['n_layers']
            raw_input_dim = int(protein_config['input_dim']) if protein_config.get('jumping_knowledge') == 'concat_full' else 0
            # Use the per-layer hidden dim (not the concatenated output dim)
            if protein_config['gnn_method'] == 'GATv2':
                layer_dim = protein_config['hidden_dim'] * protein_config.get('gat_heads')
            else:
                layer_dim = protein_config['hidden_dim']
            
            if decoder_type == 'cross_layer':
                self.s2gae_decoder = S2GAEDecoder(
                    hidden_channels=layer_dim,
                    decode_channels=self.s2gae_config.get('decode_channels'),
                    num_encoder_layers=n_layers,
                    num_decoder_layers=self.s2gae_config.get('decoder_layers'),
                    dropout=self.s2gae_config.get('decoder_dropout'),
                    raw_input_dim=raw_input_dim,
                )
            else:  # simple decoder
                self.s2gae_decoder = S2GAEDecoderSimple(
                    hidden_channels=layer_dim,
                    num_encoder_layers=n_layers,
                    dropout=self.s2gae_config.get('decoder_dropout')
                )

        # Uniformity projection head (follows AUG-MAE: linear + ReLU before uniformity loss)
        self.uniformity_layer = None
        if uniformity_dim > 0:
            self.uniformity_layer = nn.Linear(self.prot_encoder.gnn_output_dim, uniformity_dim, bias=False)

    def reset_parameters(self):
        self.prot_encoder.reset_parameters()
        self.cell_encoder.reset_parameters()
        self.tissue_encoder.reset_parameters()
        torch.nn.init.xavier_uniform_(
            self.mg_relw, gain=torch.nn.init.calculate_gain('leaky_relu')
        )
        if self.s2gae_decoder is not None:
            self.s2gae_decoder.reset_parameters()
        if self.uniformity_layer is not None:
            self.uniformity_layer.reset_parameters()
        
        if self.cell_config['pooling'] == 'learnedvn':
            nn.init.zeros_(self.virtual_node_features)
        if self.cci_encoder is not None:
            self.cci_encoder.reset_parameters()

    def set_cci_graph(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        self.cci_edge_index = edge_index
        self.cci_num_nodes = int(num_nodes)

    def _encode_cci(self, cells_x: torch.Tensor, cell_ids) -> Optional[torch.Tensor]:
        if self.cci_encoder is None:
            return None
        if self.cci_edge_index is None or self.cci_num_nodes is None:
            raise ValueError("CCI graph not set. Call set_cci_graph before using metagraph mode.")
        cell_ids_t = torch.as_tensor(cell_ids, device=cells_x.device, dtype=torch.long)
        mg_x = cells_x.new_zeros((self.cci_num_nodes, cells_x.size(1)))
        if cell_ids_t.numel() > 0:
            mg_x = mg_x.index_copy(0, cell_ids_t, cells_x)
        edge_index = self.cci_edge_index.to(device=cells_x.device)
        return self.cci_encoder(mg_x, edge_index)

    def apply_cci(self, cells_x: torch.Tensor, cell_ids):
        if not self.use_metagraph:
            return cells_x
        cci_x = self._encode_cci(cells_x, cell_ids)
        if cci_x is None:
            return cells_x
        cell_ids_t = torch.as_tensor(cell_ids, device=cci_x.device, dtype=torch.long)
        return cci_x[cell_ids_t]

    def predict_cci_edges(
        self,
        cells_x: torch.Tensor,
        cell_ids,
        edge_index: torch.Tensor,
        *,
        assume_cci_encoded: bool = True,
    ) -> Optional[torch.Tensor]:
        if not self.use_metagraph:
            return None
        if edge_index is None:
            return None
        device = cells_x.device
        edge_index = edge_index.to(device=device)
        if assume_cci_encoded:
            cell_ids_t = torch.as_tensor(cell_ids, device=device, dtype=torch.long)
            if cell_ids_t.numel() == 0:
                return None
            if self.cci_num_nodes is None:
                self.cci_num_nodes = int(cell_ids_t.max().item() + 1)
            idx_map = torch.full((self.cci_num_nodes,), -1, device=device, dtype=torch.long)
            idx_map[cell_ids_t] = torch.arange(cell_ids_t.numel(), device=device)
            src_idx = idx_map[edge_index[0]]
            dst_idx = idx_map[edge_index[1]]
            valid = (src_idx >= 0) & (dst_idx >= 0)
            if valid.sum() == 0:
                return None
            src = cells_x[src_idx[valid]]
            dst = cells_x[dst_idx[valid]]
        else:
            cci_x = self._encode_cci(cells_x, cell_ids)
            if cci_x is None:
                return None
            src = cci_x[edge_index[0]]
            dst = cci_x[edge_index[1]]
        return (src * dst).sum(dim=-1)


    def build_metagraph_embeddings(self, cells_x, cell_ids, mg_data):
        """
        Build metagraph node embeddings by injecting learned cell embeddings and
        tissue classifier weights as tissue embeddings.
        """
        mg_x = cells_x.new_zeros((mg_data.num_nodes, cells_x.shape[1]))

        if len(cell_ids) > 0:
            mg_x[cell_ids, :] = cells_x[:len(cell_ids)]

        node_type = mg_data.node_type.to(cells_x.device)
        tissue_ids = torch.nonzero(node_type == 0, as_tuple=False).view(-1)
        if tissue_ids.numel() > 0:
            # Align tissues directly to metagraph ordering, capping if fewer classifier rows
            max_rows = min(len(tissue_ids), self.tissue_encoder.weight.shape[0])
            mg_x[tissue_ids[:max_rows]] = self.tissue_encoder.weight[:max_rows].to(cells_x.device)

        return mg_x
    
    def remove_virtual_nodes(self, ppi_emb, batch_ppi_x: Optional[Batch] = None):
        """
        helper to drop VN rows from returned embeddings.

        - If `ppi_emb` is a dict: drops the last row of each tensor.
        - If `ppi_emb` is a tensor: requires `batch_ppi_x` and removes one VN per graph.
          Returns `(ppi_emb_no_vn, protein_mask, new_ptr)`.
        """
        pooling = self.cell_config.get("pooling", None)
        if pooling == "virtual_node":
            pooling = "vn"
        if pooling not in {"vn", "learnedvn"}:
            return ppi_emb

        if isinstance(ppi_emb, dict):
            return {k: (v[:-1] if v is not None and v.size(0) > 0 else v) for k, v in ppi_emb.items()}

        if batch_ppi_x is None:
            raise ValueError("batch_ppi_x must be provided when removing VNs from a batched embedding tensor.")

        vn_indices = batch_ppi_x.ptr[1:] - 1
        mask = torch.ones(int(ppi_emb.size(0)), device=ppi_emb.device, dtype=torch.bool)
        mask[vn_indices.to(device=ppi_emb.device)] = False
        ppi_emb_no_vn = ppi_emb[mask]

        num_graphs = int(batch_ppi_x.ptr.numel() - 1)
        ptr_shift = torch.arange(num_graphs + 1, device=batch_ppi_x.ptr.device, dtype=batch_ppi_x.ptr.dtype)
        new_ptr = batch_ppi_x.ptr - ptr_shift
        return ppi_emb_no_vn, mask, new_ptr

    def remove_virtual_nodes_from_batch(
        self,
        batch_ppi_x: Batch,
        node_embeddings: torch.Tensor,
        layer_outputs: Optional[list] = None,
        *,
        drop_message_passing_edges: bool = True,
    ):
        """
        Remove VN nodes from a batched representation and remap indices so decoding uses
        the original protein-only indexing.

        cleanup after a forward pass when pooling is done with VN

        Returns:
            node_embeddings_no_vn, batch_ppi_x_clean, layer_outputs_no_vn
        """
        pooling = self.cell_config.get("pooling", None)
        if pooling == "virtual_node":
            pooling = "vn"
        if pooling not in {"vn", "learnedvn"}:
            return node_embeddings, batch_ppi_x, layer_outputs

        if batch_ppi_x.ptr is None:
            raise ValueError("batch_ppi_x must have a valid ptr to remove virtual nodes.")

        vn_indices = batch_ppi_x.ptr[1:] - 1  # one VN per graph (last node)
        total_nodes = int(batch_ppi_x.batch.numel())
        if total_nodes != int(node_embeddings.size(0)):
            raise ValueError(
                f"node_embeddings has {int(node_embeddings.size(0))} rows but batch has {total_nodes} nodes."
            )

        mask = torch.ones(total_nodes, device=node_embeddings.device, dtype=torch.bool)
        mask[vn_indices.to(device=mask.device)] = False

        # Remap decoding edges from the old indexing (with 1 VN per graph) to the
        # protein only indexing by subtracting the graph id
        if getattr(batch_ppi_x, "edge_index", None) is not None:
            edge_index = batch_ppi_x.edge_index
            batch_vec = batch_ppi_x.batch.to(device=edge_index.device)
            batch_ppi_x.edge_index = edge_index - batch_vec[edge_index]

        # Update node-level tensors
        if getattr(batch_ppi_x, "x", None) is not None and int(batch_ppi_x.x.size(0)) == total_nodes:
            batch_ppi_x.x = batch_ppi_x.x[mask.to(device=batch_ppi_x.x.device)]
        batch_ppi_x.batch = batch_ppi_x.batch[mask.to(device=batch_ppi_x.batch.device)]

        # Some samplers keep `n_id` as a node-level attribute.
        if getattr(batch_ppi_x, "n_id", None) is not None and int(batch_ppi_x.n_id.size(0)) == total_nodes:
            batch_ppi_x.n_id = batch_ppi_x.n_id[mask.to(device=batch_ppi_x.n_id.device)]

        # Update ptr (one removed node per graph)
        num_graphs = int(batch_ppi_x.ptr.numel() - 1)
        ptr_shift = torch.arange(num_graphs + 1, device=batch_ppi_x.ptr.device, dtype=batch_ppi_x.ptr.dtype)
        batch_ppi_x.ptr = batch_ppi_x.ptr - ptr_shift

        if drop_message_passing_edges and hasattr(batch_ppi_x, "edge_index_mp"):
            delattr(batch_ppi_x, "edge_index_mp")

        node_embeddings = node_embeddings[mask]
        if layer_outputs is not None:
            layer_outputs = [h[mask] for h in layer_outputs]

        return node_embeddings, batch_ppi_x, layer_outputs

    def forward_original(
        self, 
        ppi_x,
        ppi_edge_index,
        batching=False,
        return_layer_outputs=None
        ):
        """

        Args:
            ppi_x (dict): Dictionary of protein-celltype-tissue representations.
            ppi_edge_index (dict): Edge index, edge type, edge labels - for PPI graph.
            batching (bool): Whether to use batched mode.
            return_layer_outputs (bool, optional): Whether to return per-layer embeddings for S2GAE.
                                                   If None, uses s2gae_enabled setting.
        Returns:
            ppi_x (dict): Updated dictionary of protein representations per cell type.
            batch_ppi_x: Batched PPI data (if batching=True)
            cells_x (torch.Tensor): cell embeddings
            cells_pred (torch.Tensor): cell predictions for the multi-label classification of tissues
            layer_outputs (list, optional): Per-layer embeddings if return_layer_outputs=True
        """
        
        # Determine if we should return layer outputs for S2GAE
        if return_layer_outputs is None:
            return_layer_outputs = self.s2gae_enabled
        
        if not batching:
            # compute protein representations
            prot_output = self.prot_encoder(ppi_x, ppi_edge_index, 
                                            return_layer_outputs=return_layer_outputs)
            
            if return_layer_outputs:
                ppi_x, layer_outputs_dict = prot_output
            else:
                ppi_x = prot_output
                layer_outputs_dict = None
            
            # compute cell representations
            cells_x = self.cell_encoder(ppi_x)
            
            # compute tissue representations
            if self.use_metagraph:
                cells_x = self.apply_cci(cells_x, list(ppi_x.keys()))
            cells_pred = self.tissue_encoder(cells_x)
            # BCEWithLogitsLoss applies the sigmoid internally.
            
            if return_layer_outputs:
                return ppi_x, None, cells_x, cells_pred, layer_outputs_dict
            return ppi_x, None, cells_x, cells_pred
        else:
            data_list = []
            for i, celltype in enumerate(ppi_x.keys()):
                vn_id = i
                if ppi_edge_index is not None and (celltype in ppi_edge_index):
                    vn_id = ppi_edge_index[celltype].get('vn_id', i)
                data_list.append(
                    Data(
                        x=ppi_x[celltype],
                        edge_index=ppi_edge_index[celltype]['total_edge_index'],
                        y=ppi_edge_index[celltype]['y'],
                        vn_id=torch.tensor([vn_id], device=ppi_x[celltype].device, dtype=torch.long),
                    )
                )

            batch_ppi_x = Batch().from_data_list(data_list)
            
            # Get protein embeddings (and VN embeddings if using virtual node)
            prot_output = self.prot_encoder(batch_ppi_x, None, batching=batching,
                                            return_layer_outputs=return_layer_outputs)
            
            # Handle different return patterns based on VN and S2GAE modes
            layer_outputs = None
            if self.use_virtual_node:
                if return_layer_outputs:
                    emb_ppi_x, cells_vn, layer_outputs = prot_output
                else:
                    emb_ppi_x, cells_vn = prot_output
            else:
                if return_layer_outputs:
                    emb_ppi_x, layer_outputs = prot_output
                else:
                    emb_ppi_x = prot_output
                cells_vn = None
            
            if self.cell_config['n_cells'] is None:
                for i, celltype in enumerate(ppi_x.keys()):
                    start_idx = batch_ppi_x.ptr[i]
                    end_idx = batch_ppi_x.ptr[i + 1] 
                    ppi_x[celltype] = emb_ppi_x[start_idx:end_idx]
                # compute cell representations
                cells_x = self.cell_encoder(ppi_x, cells_vn=cells_vn)
                
                # compute tissue representations
                if self.use_metagraph:
                    cells_x = self.apply_cci(cells_x, list(ppi_x.keys()))
                cells_pred = self.tissue_encoder(cells_x)
                # BCEWithLogitsLoss applies the sigmoid internally.
                
                if return_layer_outputs:
                    return ppi_x, batch_ppi_x, cells_x, cells_pred, layer_outputs
                return ppi_x, batch_ppi_x, cells_x, cells_pred
            else:
                # compute cell representations (pass VN embeddings if available)
                cells_x = self.cell_encoder(emb_ppi_x, batch_ppi_x, cells_vn=cells_vn)
                
                # compute tissue representations
                if self.use_metagraph:
                    cells_x = self.apply_cci(cells_x, list(ppi_x.keys()))
                cells_pred = self.tissue_encoder(cells_x)
                
                if return_layer_outputs:
                    return emb_ppi_x, batch_ppi_x, cells_x, cells_pred, layer_outputs
                return emb_ppi_x, batch_ppi_x, cells_x, cells_pred
            
    
    def forward_factored(
        self, 
        ppi_dict,
        batching=False,
        return_layer_outputs=None
        ):
        """
        Args:
            ppi_x (dict): Dictionary of protein-celltype-tissue representations.
            ppi_edge_index (dict): Edge index, edge type, edge labels - for PPI graph.
        Returns:
            ppi_x (dict): Updated dictionary of protein representations per cell type.
            cells_x (torch.Tensor): cell embeddings
            cells_pred (torch.Tensor): cell predictions for the multi-label classification of tissues
        """
        
        # Determine if we should return layer outputs for S2GAE
        if return_layer_outputs is None:
            return_layer_outputs = self.s2gae_enabled
        
        # Keep the sampler's input graphs protein-only; it generates negatives after
        # this forward pass and must never see the temporary virtual nodes.
        ppi_dict_local = dict(ppi_dict)
        if self.cell_config.get("pooling") == 'vn' or self.protein_config.get("add_virtual_node"):
            for i, celltype in enumerate(ppi_dict_local.keys()):
                ppi_dict_local[celltype] = add_virtual_node(
                    ppi_dict_local[celltype], self.virtual_node_features, clone=True)
        
        elif self.cell_config.get("pooling") == 'learnedvn':
            for i, celltype in enumerate(ppi_dict_local.keys()):
                vn_id = self.celltype_to_vn_id.get(celltype, i)
                vn_emb = self.virtual_node_features[vn_id]
                ppi_dict_local[celltype] = add_virtual_node(ppi_dict_local[celltype], vn_emb, clone=True)

        if not batching:

            # compute protein representations - keep dictionary format for embeddings
            prot_output = self.prot_encoder(ppi_dict_local, batching=batching, return_layer_outputs=return_layer_outputs)

            if return_layer_outputs:
                ppi_emb_dict, layer_outputs_dict = prot_output
            else:
                ppi_emb_dict = prot_output
                layer_outputs_dict = None
            
            # Remove virtual nodes before pooling when used only for message passing.
            if self.cell_config.get("pooling") in ['mean','attention'] and self.protein_config.get("add_virtual_node"):
                ppi_emb_dict = self.remove_virtual_nodes(ppi_emb_dict)
                if return_layer_outputs and layer_outputs_dict is not None:
                    layer_outputs_dict = {
                        k: [
                            (h[:-1] if h is not None and h.size(0) > 0 else h)
                            for h in hs
                        ]
                        for k, hs in layer_outputs_dict.items()
                    }
        
            # compute cell representations
            cells_x = self.cell_encoder(ppi_emb_dict)
            
            # compute tissue representations
            if self.use_metagraph:
                cells_x = self.apply_cci(cells_x, list(ppi_dict_local.keys()))
            cells_pred = self.tissue_encoder(cells_x)
            # BCEWithLogitsLoss applies the sigmoid internally.

            # Drop virtual node rows from returned protein embeddings (keep VN only for pooling)
            if self.cell_config.get("pooling") in ['vn','learnedvn'] :
                
                ppi_emb_dict = self.remove_virtual_nodes(ppi_emb_dict)
                if return_layer_outputs and layer_outputs_dict is not None:
                    layer_outputs_dict = {
                        k: [
                            (h[:-1] if h is not None and h.size(0) > 0 else h)
                            for h in hs
                        ]
                        for k, hs in layer_outputs_dict.items()
                    }

            if return_layer_outputs:
                return ppi_emb_dict, None, cells_x, cells_pred, layer_outputs_dict
            return ppi_emb_dict, None, cells_x, cells_pred
        
        else:
            batch_ppi_x = Batch().from_data_list([ppi_dict_local[celltype] for celltype in ppi_dict_local.keys()])
            
            # Get protein embeddings (and VN embeddings if using virtual node)
            # keep batch structure
            prot_output = self.prot_encoder(batch_ppi_x, batching=batching, return_layer_outputs=return_layer_outputs)

            layer_outputs = None
            
            if return_layer_outputs:
                emb_ppi_x, layer_outputs = prot_output
            else:
                emb_ppi_x = prot_output

            if self.cell_config['n_cells'] is None:
                if self.cell_config.get("pooling") in {"vn", "learnedvn"} or self.protein_config.get("add_virtual_node"):
                    raise NotImplementedError('Virtual-node mode requires n_cells to be specified.')
                
                for i, celltype in enumerate(ppi_dict_local.keys()):
                    start_idx = batch_ppi_x.ptr[i]
                    end_idx = batch_ppi_x.ptr[i + 1] 
                    ppi_dict_local[celltype] = emb_ppi_x[start_idx:end_idx]
                # compute cell representations
                cells_x = self.cell_encoder(ppi_dict_local)
                
                # compute tissue representations
                if self.use_metagraph:
                    cells_x = self.apply_cci(cells_x, list(ppi_dict_local.keys()))
                cells_pred = self.tissue_encoder(cells_x)
                # BCEWithLogitsLoss applies the sigmoid internally.
                if return_layer_outputs:
                    return ppi_dict_local, batch_ppi_x, cells_x, cells_pred, layer_outputs
                return ppi_dict_local, batch_ppi_x, cells_x, cells_pred
            else:
                # Remove virtual nodes before pooling when used only for message passing.
                if self.cell_config.get("pooling") in ['mean','attention'] and self.protein_config.get("add_virtual_node"):
                    emb_ppi_x, batch_ppi_x, layer_outputs = self.remove_virtual_nodes_from_batch(
                        batch_ppi_x, emb_ppi_x, layer_outputs
                    )
                # compute cell representations
                cells_x = self.cell_encoder(emb_ppi_x, batch_ppi_x)

                # compute tissue representations
                if self.use_metagraph:
                    cells_x = self.apply_cci(cells_x, list(ppi_dict_local.keys()))
                cells_pred = self.tissue_encoder(cells_x)

                # Drop virtual nodes from returned batch/embeddings (keep VN only for pooling)
                if self.cell_config.get("pooling") in {"vn", "learnedvn"}:
                    emb_ppi_x, batch_ppi_x, layer_outputs = self.remove_virtual_nodes_from_batch(
                        batch_ppi_x, emb_ppi_x, layer_outputs
                    )
                if return_layer_outputs:
                    return emb_ppi_x, batch_ppi_x, cells_x, cells_pred, layer_outputs
                return emb_ppi_x, batch_ppi_x, cells_x, cells_pred


    def forward(self, ppi_data, batching=False, return_layer_outputs=None):
        if isinstance(ppi_data, tuple):
            # data must be passed as (ppi_x_dict, ppi_edge_index_dict)
            ppi_x, ppi_edge_index = ppi_data
            return self.forward_original(
                ppi_x,
                ppi_edge_index,
                batching=batching,
                return_layer_outputs=return_layer_outputs,
            )
        
        elif isinstance(ppi_data, dict):
            # data must be passed as ppi_dict
            return self.forward_factored(ppi_data, batching=batching, return_layer_outputs=return_layer_outputs)
        
        else:
            raise ValueError('ppi_data must be either a tuple (ppi_x, ppi_edge_index) or a dict ({cell_type}: PyGeo Data})')
