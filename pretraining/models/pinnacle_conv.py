import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.nn.inits import glorot, zeros
from .custom_gnn import ACM_RandomWalk


class PCTConv(nn.Module):
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
        """
        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        num_ppi_relations : int
            Number of PPI relations.
        num_mg_relations : int
            Number of metagraph relations.
        ppi_data : dict
            Dictionary containing PPI data for each cell type.
        gnn_method: str
        
        out_channels : int
            Output feature dimension.
        sem_att_channels : int
            Number of semantic attention channels.
        pc_att_channels : int
            Number of protein-cell attention channels.
        node_heads : int, optional
            Number of attention heads. The default is 3.
        tissue_update : int, optional
            Number of iterations for tissue embedding (non-parametric) update. The default is 100.
        shared_ppi_gnn : bool, optional
            Whether to use a shared GNN for all cell types. The default is False.
        """
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
        self.tissue_update = 100
        self.shared_ppi_gnn = shared_ppi_gnn
        
        # Cell-type specific PPI weights
        self.ppi_attn = dict()

        # Independent GAT per cell type specific PPI network
        if not shared_ppi_gnn:
            # Default for Pinnacle method
            # Independent GAT per cell type specific PPI network
            self.ppi_w = torch.nn.ModuleList()
            for celltype, ppi in ppi_data.items():
                if self.gnn_method == 'GATv2':
                    local_layer = GATv2Conv(in_channels, out_channels, node_heads)
                    output_dimension = out_channels * node_heads
                elif self.gnn_method == 'GIN':
                    local_layer = GINConv(
                        nn.Linear(self.in_channels, self.out_channels)
                    )
                    output_dimension = self.out_channels
                elif self.gnn_method == 'ACM_RandomWalk':
                    local_layer = ACM_RandomWalk(
                            self.in_channels,
                            self.out_channels,
                            gnn_activation='relu',
                            use_batchnorm=True,
                            T=3.,
                            device=device)
                    output_dimension = self.out_channels
                self.ppi_w.append(local_layer)
        else:
            # Shared GAT for all cell types
            self.ppi_w = torch.nn.ModuleList()
            if self.gnn_method == 'GATv2':
                local_layer = GATv2Conv(in_channels, out_channels, node_heads)
                output_dimension = out_channels * node_heads
            elif self.gnn_method == 'GIN':
                    local_layer = GINConv(
                        nn.Linear(self.in_channels, self.out_channels)
                    )
                    output_dimension = self.out_channels
            elif self.gnn_method == 'ACM_RandomWalk':
                local_layer = ACM_RandomWalk(
                        self.in_channels,
                        self.out_channels,
                        gnn_activation='relu',
                        use_batchnorm=True,
                        T=3.,
                        device=device)
                output_dimension = self.out_channels
            self.ppi_w.append(local_layer)

        # Independent GAT for metagraph
        if self.gnn_method == 'GATv2':
            self.mg_conv_in = GATv2Conv(in_channels, out_channels, node_heads)
            self.mg_conv_out = GATv2Conv(output_dimension, out_channels, node_heads)
        
        elif self.gnn_method == 'GIN':
            self.mg_conv_in = GINConv(
                nn.Linear(self.in_channels, self.out_channels)
            )
            self.mg_conv_out = GINConv(
                nn.Linear(output_dimension, self.out_channels)
            )
            output_dimension = self.out_channels
        elif self.gnn_method == 'ACM_RandomWalk':
            self.mg_conv_in = ACM_RandomWalk(
                self.in_channels,
                self.out_channels,
                gnn_activation='relu',
                use_batchnorm=True,
                T=3.,
                device=device)
            self.mg_conv_out = ACM_RandomWalk(
                self.out_channels,
                self.out_channels,
                gnn_activation='relu',
                use_batchnorm=True,
                T=3.,
                device=device)
            output_dimension = self.out_channels
        # Semantic attention (shared across networks)
        # W: Projects GAT output to attention space.
        self.W = nn.Parameter(torch.Tensor(1, 1, output_dimension, sem_att_channels))
        # b: Bias for non-linearity.
        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        # q: Query vector for attention scores.
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        # Weight initialization
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
        
        # reset_parameters
        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)
 
    # This processes a single cell type's PPI graph
    def _per_data_forward(self, x, edgetypes, node_conv):
        """
            x: Node features of proteins in the graph.
            metapaths/edgetypes: A list of edge indices for different meta-paths (i.e., sampled neighborhoods).
                - edgetypes is a list of edge indices for different meta-paths.
                - So only 1 for proteins / 4 for tissue-cell types.
            node_conv: The GATv2 layer for this cell type.
        """

        
        # Calculate node-level attention representations
        out = [node_conv(x, edgetype) for edgetype in edgetypes if edgetype.shape[1] > 0]
        #print('out ', [a.shape for a in out])
        out = torch.stack(out, dim=1).to(x.device)
        #print('stacked out:', out.shape)
        # Apply non-linearity
        out = F.leaky_relu(out)

        # Aggregate node-level representation using semantic level attention     
        #print('parameters of W, b, q: ', self.W.shape, self.b.shape, self.q.shape) 
        #print('out.unsqueeze(-1) ', out.unsqueeze(-1).shape)  
        w = torch.sum(self.W * out.unsqueeze(-1), dim=-2) + self.b
        w = torch.tanh(w)
        #print('w ', w.shape)
        beta = torch.sum(self.q * w, dim=-1)
        beta = torch.softmax(beta, dim=1)
        #print('beta ', beta.shape)
        z = torch.sum(out * beta.unsqueeze(-1), dim=1)
        #print('z ', z.shape)
        
        return z

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
        """ 
        Parameters
        ----------
        ppi_x : dict
            Dictionary containing PPI data for each cell type.
        mg_x : torch.Tensor
            Metagraph embeddings.
        ppi_metapaths : dict
            Dictionary containing PPI metapaths for each cell type.
        mg_metapaths : list
            List of metapaths for the metagraph.
        ppi_edge_index : dict
            Dictionary containing edge indices for PPI data for each cell type.
        mg_edge_index : list
            List of edge indices for the metagraph.
        tissue_neighbors : dict
            Dictionary containing tissue neighbors for each cell type.
        init_cci : bool, optional
            Whether to initialize CCI embeddings. The default is False."""
        
        #print('tissue_neighbors:', tissue_neighbors)
        if init_cci:
            mg_x_list = []
            # print('init_cci is True')
            
        else: # Project metagraph embeddings to the same dimension as PPI
            # Apply shared GATv2Conv to metagraph embeddings independently for each cell type
            # then aggregate tissue/cell representations via attention-based pooling 
            mg_x = self._per_data_forward(mg_x, mg_metapaths, self.mg_conv_in)
        
        for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers
            #print('celltype:', celltype)
            if len(ppi_metapaths[celltype]) == 0:
                ppi_x[celltype] = []
            else:
                if self.shared_ppi_gnn: # Shared GNN for all cell types
                    ppi_x[celltype] = self._per_data_forward(x, ppi_metapaths[celltype], self.ppi_w[0])
                else: # Independent GNN for each cell type
                    # Default for Pinnacle method
                    # Independent GAT per cell type specific PPI network
                    ppi_x[celltype] = self._per_data_forward(x, ppi_metapaths[celltype], self.ppi_w[celltype])

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
            #print('init_cci is True, concatenating metagraph embeddings')
            mg_x = torch.stack(mg_x_list)
            #print('mg_x shape:', mg_x.shape)
            #print('tissue_neighbors:', len(tissue_neighbors))
            bto = torch.zeros(
                len(tissue_neighbors),
                mg_x.shape[1],
                device=mg_x.device,
                dtype=mg_x.dtype,
            )
            mg_x = torch.cat((mg_x, torch.normal(bto, std=1)))
            #print('finale mg_x shape:', mg_x.shape)
        
        #print('updating tissue embeddings')
        # Update tissue embeddings
        # This is done by averaging the embeddings of neighboring tissues
        # for as many round than set in self.tissue_update - hence there are doing non-parametric graph convolutions here
        #print('self.tissue_update:', self.tissue_update) 
        for i in range(self.tissue_update): # Initialize tissue embeddings in a more meaningful way
            for t in sorted(tissue_neighbors):
                assert len(tissue_neighbors[t]) != 0
                mg_x[t, :] = torch.mean(mg_x[tissue_neighbors[t]], 0)
        
        mg_x = self._per_data_forward(mg_x, mg_metapaths, self.mg_conv_out)
        
        return ppi_x, mg_x


class PPIConv(nn.Module):
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
        super().__init__()
        """ 
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        num_ppi_relations : int
            Number of PPI relations.
        out_channels : int
            Output feature dimension.
        ppi_data : dict
            Dictionary containing PPI data for each cell type.
        gnn_method: str

        sem_att_channels : int
            Number of semantic attention channels.
        node_heads : int, optional
            Number of attention heads. The default is 3.
        shared_ppi_gnn : bool, optional
            Whether to use a shared GNN for all cell types. The default is False.
        """
        # Initialization
        self.in_channels = in_channels
        self.num_ppi_relations = num_ppi_relations
        self.out_channels = out_channels
        self.gnn_method = gnn_method
        self.sem_att_channels = sem_att_channels
        self.node_heads = node_heads
        self.shared_ppi_gnn = shared_ppi_gnn
        
        if not shared_ppi_gnn:
            # Default for Pinnacle method
            # Independent GAT per cell type specific PPI network
            self.ppi_w = torch.nn.ModuleList()
            for celltype, ppi in ppi_data.items():
                if self.gnn_method == 'GATv2':
                    local_layer = GATv2Conv(in_channels, out_channels, node_heads)
                    output_dimension = out_channels * node_heads

                elif self.gnn_method == 'GIN':
                    local_layer = GINConv(
                        nn.Linear(self.in_channels, self.out_channels)
                    )
                    output_dimension = self.out_channels

                elif self.gnn_method == 'ACM_RandomWalk':
                    local_layer = ACM_RandomWalk(
                            self.in_channels,
                            self.out_channels,
                            gnn_activation='relu',
                            use_batchnorm=True,
                            T=3.,
                            device=device)
                    output_dimension = self.out_channels
                self.ppi_w.append(local_layer)
        else:
            # Shared GAT for all cell types
            self.ppi_w = torch.nn.ModuleList()
            if self.gnn_method == 'GATv2':
                local_layer = GATv2Conv(in_channels, out_channels, node_heads)
                output_dimension = out_channels * node_heads
            elif self.gnn_method == 'GIN':
                    local_layer = GINConv(
                        nn.Linear(self.in_channels, self.out_channels)
                    )
                    output_dimension = self.out_channels

            elif self.gnn_method == 'ACM_RandomWalk':
                local_layer = ACM_RandomWalk(
                        self.in_channels,
                        self.out_channels,
                        gnn_activation='relu',
                        use_batchnorm=True,
                        T=3.,
                        device=device)
                output_dimension = self.out_channels
            self.ppi_w.append(local_layer)
            
        # Define semantic-level attention parameters
        self.W = nn.Parameter(torch.Tensor(1, 1, output_dimension, sem_att_channels))
        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        # Initialization
        nn.init.xavier_uniform_(self.W, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.b, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.q, gain = nn.init.calculate_gain('leaky_relu'))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)

    def _per_data_forward(self, x, metapaths, node_conv):

        # Calculate node-level attention representations
        out = [node_conv(x, metapath) for metapath in metapaths if metapath.shape[1] > 0]
        out = torch.stack(out, dim=1).to(x.device)
        
        # Apply non-linearity
        out = F.leaky_relu(out)

        # Aggregate node-level representation using semantic level attention
        w = torch.sum(self.W * out.unsqueeze(-1), dim=-2) + self.b
        w = torch.tanh(w)
        beta = torch.sum(self.q * w, dim=-1)
        beta = torch.softmax(beta, dim=1)
        z = torch.sum(out * beta.unsqueeze(-1), dim=1)

        return z

    def forward(self, ppi_x, ppi_metapaths, mg_x, ppi_attn):
        
        for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers

            if len(ppi_metapaths[celltype]) == 0:
                ppi_x[celltype] = []
            else: # Update using meta-path attention
                if not self.shared_ppi_gnn: # default for Pinnacle method
                    ppi_x[celltype] = self._per_data_forward(x, ppi_metapaths[celltype], self.ppi_w[celltype])
                else:
                    ppi_x[celltype] = self._per_data_forward(x, ppi_metapaths[celltype], self.ppi_w[0])
                
            # Downpool using cell-type embedding
            gamma = ppi_attn[celltype] # (n,) where n = number of proteins in the cell type (in the batch)
            ppi_x[celltype] += (mg_x[celltype, :].repeat(len(gamma), 1) * gamma.unsqueeze(-1))
        
        return ppi_x
