import torch
import torch.nn as nn
from .attention import GatedAttention
from torch_geometric.nn import global_mean_pool


class moving_average(nn.Module):
    """Average each cell embedding over recent batches. Evaluation leaves the memory unchanged."""
    def __init__(
        self,
        window_size,
        embedding_dim,
        n_cells=None,
        device=None,
        dtype=torch.float32):
        """Allocate a fixed window of previous embeddings for one cell or all cells."""
        super().__init__()

        self.n_cells = n_cells
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        if self.n_cells is None:
            # considered as only one cell register
            window_embeddings = torch.zeros(
                window_size, embedding_dim,
                device=device, dtype=dtype,
                requires_grad=False)
        else:
            assert self.n_cells > 1
            # consider n_cells registers - one channel per cell
            window_embeddings = torch.zeros(
                (self.n_cells, window_size, embedding_dim),
                device=device, dtype=dtype,
                requires_grad=False)
        self.step = 0

        self.register_buffer('window_embeddings', window_embeddings)

    def reset_parameters(self):
        """Clear stored embeddings and restart the window counter."""
        self.step = 0
        self.window_embeddings.fill_(0)

    def forward(self, batch_cell_x):
        """Average current and stored cell features; update the window only during training."""
        if self.n_cells is None:

            if self.training:
                if self.step < self.window_size:
                    # Fill the window with the first embeddings
                    agg_cell_x = (batch_cell_x + self.window_embeddings[:self.step].sum(dim=0)) / (self.step + 1)
                    self.window_embeddings[self.step] = batch_cell_x.detach().clone()
                else:
                    # Update the window by removing the oldest embedding and adding the new one
                    agg_cell_x = (batch_cell_x + self.window_embeddings[1:].sum(dim=0)) / self.window_size
                    self.window_embeddings[: self.window_size - 1] = self.window_embeddings[1:].clone()
                    self.window_embeddings[self.window_size - 1] = batch_cell_x.detach().clone()

                self.step += 1
            else:
                if self.step < self.window_size:
                    agg_cell_x = (batch_cell_x + self.window_embeddings[:self.step].sum(dim=0)) / (self.step + 1)
                else:
                    agg_cell_x = (batch_cell_x + self.window_embeddings.sum(dim=0)) / (self.window_size + 1)

        else:
            if self.training:
                if self.step < self.window_size:
                    # Fill the window with the first embeddings
                    agg_cell_x = (batch_cell_x + self.window_embeddings[:, :self.step, :].sum(dim=1)) / (self.step + 1)
                    self.window_embeddings[:, self.step, :] = batch_cell_x.detach().clone()
                else:
                    # Update the window by removing the oldest embedding and adding the new one
                    agg_cell_x = (batch_cell_x + self.window_embeddings[:, 1:, :].sum(dim=1)) / self.window_size
                    self.window_embeddings[:, : self.window_size - 1, :] = self.window_embeddings[:, 1:, :].clone()
                    self.window_embeddings[:, self.window_size - 1, :] = batch_cell_x.detach().clone()

                self.step += 1
            else:
                if self.step < self.window_size:
                    agg_cell_x = (batch_cell_x + self.window_embeddings[:, :self.step, :].sum(dim=1)) / (self.step + 1)
                else:
                    agg_cell_x = (batch_cell_x + self.window_embeddings.sum(dim=1)) / (self.window_size + 1)

        return agg_cell_x


class cell_module(nn.Module):
    """Pool protein embeddings by mean, attention or a virtual node, then apply cell memory."""
    def __init__(
        self,
        cell_config,
        ppi_data,
        device):
        """Configure cell pooling and optional moving-average memory for the PPI contexts."""
        super().__init__()

        # Instantiate protein module

        self.device = device
        self.cell_config = cell_config
        self.cell_pooling = cell_config['pooling']
        # Pooling method for cell-type specific PPI

        assert self.cell_pooling in ['mean', 'attention', 'vn', 'learnedvn']

        if self.cell_pooling == 'attention':
            self.attention = GatedAttention(
                input_dim=cell_config['embedding_dim'],
                emb_dim=cell_config['att_embedding_dim']
            )
        # the VN embeddings come from prot_module and are passed directly

        self.cell_memory = cell_config['memory']
        assert self.cell_memory in ['average', 'lstm', 'none'] # Memory mechanism for cell-type specific PPI sampled by batches


        if self.cell_memory == 'average':
            if self.cell_config['n_cells'] is None:
                self.cell_memory_layers = nn.ModuleList([
                    moving_average(
                        window_size=cell_config['window_size'],
                        embedding_dim=cell_config['embedding_dim'],
                        device=device)
                    for celltype, _ in ppi_data.items()])
            else:
                assert self.cell_config['n_cells'] == len(list(ppi_data.keys()))
                self.cell_memory_layers = moving_average(
                        window_size=cell_config['window_size'],
                        embedding_dim=cell_config['embedding_dim'],
                        n_cells= cell_config['n_cells'],
                        device=device)

        elif self.cell_memory == 'lstm':
            raise NotImplementedError("LSTM memory is not implemented yet.")

        else:
            self.cell_memory_layers = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize attention parameters and clear cell memory when present."""
        if self.cell_memory_layers is not None:
            if self.cell_config['n_cells'] is None:
                for layer in self.cell_memory_layers:
                    layer.reset_parameters()
            else:
                self.cell_memory_layers.reset_parameters()


    def get_cell_embedding_per_celltype(self, x):
        """Pool a single cell's protein embeddings using the configured pooling method."""
        if self.cell_pooling == 'mean':
            # Mean pooling to get cell embeddings
            cell_x = x.mean(dim=0)

        elif self.cell_pooling == 'attention':
            Att, cell_x = self.attention(x, None)
            if cell_x.shape[0] == 1:
                cell_x = cell_x[0, :]

        elif self.cell_pooling in ['vn', 'learnedvn']:
            # stored vn as the last element of the graph
            cell_x = x[-1, :]

        return cell_x


    def forward(self, ppi_x, batch_ppi_x=None, cells_vn=None):
        """Pool a dictionary or PyG batch of proteins into one embedding per cell."""
        if isinstance(ppi_x, dict):
            assert batch_ppi_x is None

        if (self.cell_config['n_cells'] is None):
            assert isinstance(ppi_x, dict)

        # Standard pooling modes (mean/attention)
        if self.cell_config['n_cells'] is None:
            cells_x = []
            for i, (celltype, x) in enumerate(ppi_x.items()): # Iterate through cell-type specific PPI layers
                cell_x = self.get_cell_embedding_per_celltype(x)

                if self.cell_memory == 'average':
                    # Apply moving average memory mechanism
                    cell_x = self.cell_memory_layers[i](cell_x)

                elif self.cell_memory == 'lstm':
                    raise NotImplementedError("LSTM memory is not implemented yet.")

                elif self.cell_memory == 'none':
                    cell_x = cell_x
                cells_x.append(cell_x[None, :])

            cells_x = torch.concat(cells_x, dim=0)

        else: # in this setup a unique layer containing parameters of the form (ppix)
            if batch_ppi_x is not None: # we can parallelize operations on GPUs in batch form
                if self.cell_pooling == 'mean':
                    # Mean pooling to get cell embeddings
                    cells_x = global_mean_pool(ppi_x, batch_ppi_x.batch)
                elif self.cell_pooling == 'attention':
                    list_Att, cells_x = self.attention(ppi_x, batch_ppi_x)
                elif self.cell_pooling in ['vn', 'learnedvn']:
                    # Virtual node is stored as the last node of each graph.
                    vn_indices = batch_ppi_x.ptr[1:] - 1  # [num_graphs]
                    cells_x = ppi_x[vn_indices, :]

                if self.cell_memory == 'average':
                    # Apply moving average memory mechanism
                    cells_x = self.cell_memory_layers(cells_x)

                elif self.cell_memory == 'lstm':
                    raise NotImplementedError("LSTM memory is not implemented yet.")

            else:
                cells_x = []
                for i, (celltype, x) in enumerate(ppi_x.items()): # Iterate through cell-type specific PPI layers
                    cell_x = self.get_cell_embedding_per_celltype(x)

                    cells_x.append(cell_x[None, :])

                cells_x = torch.concat(cells_x, dim=0)
                if self.cell_memory == 'average':
                    # Apply moving average memory mechanism
                    cells_x = self.cell_memory_layers(cells_x)

                elif self.cell_memory == 'lstm':
                    raise NotImplementedError("LSTM memory is not implemented yet.")

        return cells_x
