import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.nn.inits import glorot, zeros
from .attention import GatedAttention
from torch_geometric.nn import global_mean_pool


class moving_average(nn.Module):
    """
    Moving average memory module for aggregating cell embeddings over a sliding window.

    This module maintains a fixed-size window of recent cell embeddings and computes the moving average,
    which can be used as a memory mechanism for cell-level representations across training batches.

    Args:
        window_size (int): Number of recent embeddings to keep in the moving window.
        embedding_dim (int): Dimension of each cell embedding.
        device (str or torch.device, optional): Device to store the window buffer.
        dtype (torch.dtype, optional): Data type for the embeddings (default: torch.float32).

    Attributes:
        window_size (int): Size of the moving window.
        embedding_dim (int): Dimension of each embedding.
        window_embeddings (torch.Tensor): Buffer storing the recent embeddings.
        step (int): Counter for the number of updates.

    Methods:
        reset_parameters(): Resets the window buffer and step counter.
        forward(batch_cell_x): Updates the window with a new embedding and returns the current moving average.

    Notes:
        - During training, the window is updated with each new embedding.
        - During evaluation, the moving average is computed without updating the window.
        - For the first few steps (less than window_size), the average is computed over available embeddings.
    """
    def __init__(
        self,
        window_size,
        embedding_dim,
        n_cells=None,
        device=None,
        dtype=torch.float32):
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
        self.step = 0
        self.window_embeddings.fill_(0)
        
    def forward(self, batch_cell_x):
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
    """
    Cell processing module for cell-type specific protein-protein interaction (PPI) data.

    This module performs pooling of protein embeddings to obtain cell embeddings for each cell type,
    followed by an optional memory mechanism to aggregate cell embeddings across batches.

    Args:
        cell_config (dict): Configuration dictionary with the following keys:
            - 'pooling' (str): Pooling method ('mean' or 'attention') for aggregating protein embeddings.
            - 'memory' (str): Memory mechanism ('average', 'lstm', or 'none') for aggregating cell embeddings.
            - 'window_size' (int, optional): Window size for moving average (required if 'average' memory).
            - 'embedding_dim' (int): Dimension of cell embeddings.
        ppi_data (dict): Dictionary mapping cell types to their PPI data (used to initialize memory layers).
        device (str): Device to use ('cpu' or 'cuda').

    Attributes:
        cell_pooling (str): Pooling method used.
        cell_memory (str): Memory mechanism used.
        cell_memory_layers (nn.ModuleList, optional): List of memory modules for each cell type.
        device (str): Device used for computation.
        cell_config (dict): Configuration dictionary.

    Methods:
        reset_parameters(): Resets all learnable parameters and memory buffers.
        forward(ppi_x): Applies pooling and memory mechanism to input protein embeddings for each cell type.

    Notes:
        - Only 'mean' pooling is currently implemented; 'attention' pooling raises NotImplementedError.
        - Only 'average' memory is currently implemented; 'lstm' memory raises NotImplementedError.
        - If 'none' memory is selected, no memory mechanism is applied to cell embeddings.
    """
    def __init__(
        self,
        cell_config,
        ppi_data,
        device):
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
        
        # Skip memory instantiation for virtual_node pooling (VN is learned via SGD)
        
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
        if self.cell_memory_layers is not None:
            if self.cell_config['n_cells'] is None:
                for layer in self.cell_memory_layers:
                    layer.reset_parameters()
            else:
                self.cell_memory_layers.reset_parameters()
    
    
    def get_cell_embedding_per_celltype(self, x):
        if self.cell_pooling == 'mean':
            # Mean pooling to get cell embeddings
            cell_x = x.mean(dim=0)
            
        elif self.cell_pooling == 'attention':
            # Attention pooling: for now only implemented with forward on a single input
            Att, cell_x = self.attention(x, None)
            if cell_x.shape[0] == 1:
                cell_x = cell_x[0, :]
        
        elif self.cell_pooling in ['vn', 'learnedvn']:
            # stored vn as the last element of the graph
            cell_x = x[-1, :] 
        
        return cell_x
        
        
    def forward(self, ppi_x, batch_ppi_x=None, cells_vn=None):
        """
        Compute cell embeddings from protein embeddings.
        Args:
            ppi_x : can be either "emb_ppi_x" i.e embeddings associated to batch_ppi_x, batch concatenation of ppi subgraphs
                or a dictionary with all keys corresponding to the cell types, in which case batch_ppi_x must be None. 
            batch_ppi_x: Batch object with .batch attribute (for batched mode)
            cells_vn: Pre-computed virtual node embeddings [B, d] (for virtual_node pooling)
        
        Returns:
            cells_x: Cell embeddings [n_cells, d]
        """
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
                    # Attention pooling: for now only implemented with forward on a single input
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
