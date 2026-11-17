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
    if bool(getattr(data, "has_vn", False)):
        return data

    if data.x is None:
        raise ValueError("Cannot add a virtual node to a graph with no `x`.")

    device = data.x.device
    vn_embedding = vn_embedding.to(device=device, dtype=data.x.dtype).view(1, -1)

    num_nodes_before = int(data.x.size(0))
    data.x = torch.cat([data.x, vn_embedding], dim=0)  # [N+1, d]
    # PyG uses num_nodes when concatenating graphs into a batch.
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


class CCIGraphHead(nn.Module):
    """Apply one cell-cell GNN layer and optional dropout."""
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
        """Build the selected GAT, GIN or ACM cell-cell layer and record dropout."""
        super().__init__()
        if gnn_method == "GATv2":
            heads = gat_heads if gat_heads is not None else 1
            if output_dim % heads != 0:
                heads = 1
            self.gnn = GATv2Conv(input_dim, output_dim // heads, heads=heads, dropout=0.0)
        elif gnn_method == "GIN":
            self.gnn = GINConv(nn.Linear(input_dim, output_dim))
        elif gnn_method == "ACM_RandomWalk":
            self.gnn = ACM_RandomWalk(
                input_dim, output_dim, gnn_activation=gnn_activation,
                use_batchnorm=gnn_batchnorm, device=device,
            )
        else:
            raise ValueError(f"Unknown CCI GNN: {gnn_method!r}")
        self.dropout = float(dropout)

    def reset_parameters(self):
        """Reinitialize the selected cell-cell convolution."""
        if hasattr(self.gnn, "reset_parameters"):
            self.gnn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Update cell features along CCI edges and apply training dropout."""
        x = self.gnn(x, edge_index)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class hierarchical_model(nn.Module):
    """ProtScape: protein GNN, protein-to-cell pooling, CCI GNN and tissue classifier.

    Optional S2GAE reconstructs masked PPI edges from the protein layer outputs."""
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
        """Build protein encoding, cell pooling, tissue classification and optional edge decoders."""
        super(hierarchical_model, self).__init__()

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

        # Classify pooled cell embeddings by tissue.
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

        # AUG-MAE uniformity projection; paper and loss are in losses/loss.py.
        self.uniformity_layer = None
        if uniformity_dim > 0:
            self.uniformity_layer = nn.Linear(self.prot_encoder.gnn_output_dim, uniformity_dim, bias=False)

    def reset_parameters(self):
        """Reinitialize the protein, cell, tissue and optional decoder/projection parameters."""
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
        """Store the CCI message-passing graph and its total node count."""
        self.cci_edge_index = edge_index
        self.cci_num_nodes = int(num_nodes)

    def encode_cci(self, cells_x: torch.Tensor, cell_ids) -> Optional[torch.Tensor]:
        """Place cell features at metagraph node IDs and run the CCI encoder."""
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
        """Return CCI-updated features in the supplied cell order."""
        if not self.use_metagraph:
            return cells_x
        cci_x = self.encode_cci(cells_x, cell_ids)
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
        """Score cell-cell edges from the CCI features, applying the encoder if needed."""
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
            cci_x = self.encode_cci(cells_x, cell_ids)
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
        """Drop each cell's final virtual-node row.
        For batched tensors, also return the protein mask and updated graph pointers."""
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
        """Remove virtual nodes and remap batch edges to protein indices.
        Return protein embeddings, the updated batch and optional layer outputs."""
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

    def forward(self, ppi_data, batching=False, return_layer_outputs=None):
        """Encode cell-specific graphs and pool proteins into cell/tissue predictions.

        Returns protein embeddings, the PyG batch, cell embeddings and tissue logits.
        S2GAE also returns the protein embeddings from each GNN layer.
        """
        # S2GAE decodes the representation from every protein layer.
        if return_layer_outputs is None:
            return_layer_outputs = self.s2gae_enabled

        # Keep the sampler's input graphs protein-only; it generates negatives after
        # this forward pass and must never see the temporary virtual nodes.
        ppi_dict_local = dict(ppi_data)
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
