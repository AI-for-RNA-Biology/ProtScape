import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm

from .pinnacle_conv import PCTConv, PPIConv


class Pinnacle(nn.Module):
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
        super(Pinnacle, self).__init__()
        """
        Pinnacle model for protein-protein interaction and metagraph learning.
        Args:
            nfeat (int): Number of input features.
                - This is typically the size of the feature vector for each protein or cell type.
                (e.g 1024 for random feature vectors)
            hidden (int): Number of hidden units.
            output (int): Number of output units.
            num_ppi_relations (int): Number of PPI relations.
                - usually only 1 (edge type in [4])
            num_mg_relations (int): Number of metagraph relations.
                - usually 4 (edge types in [0, 1, 2, 3]) - notice that we observe that 1 was missing i.e directed edges
            ppi_data: PPI data.
            n_heads (int): Number of attention heads.
            pc_att_channels (int): Number of channels for protein-cell attention.
            dropout (float): Dropout rate.
            shared_ppi_gnn (bool): Whether to use shared PPI GNN.
        """

        self.gnn_method = gnn_method
        self.dropout = dropout

        # Layer dimensions
        self.layer1_in = nfeat
        self.layer1_out = hidden
        self.layer2_in = hidden if n_heads is None else self.layer1_out * n_heads 
        self.layer2_out = output
        self.output = hidden if n_heads is None else self.layer2_out * n_heads
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
            hidden if n_heads is None else self.layer2_out * n_heads ,
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
        """
        Forward pass of the Pinnacle model.
        This function applies two layers of PCTConv and PPIConv to update the protein-celltype-tissue representations and the metagraph representations.
        It also applies layer normalization, leaky ReLU activation, batch normalization, and dropout to the outputs of each layer.
        The first layer updates the protein-celltype-tissue representations and the metagraph representations, while the second layer updates the protein-celltype-tissue representations and down-pools the metagraph representations.
        The outputs of the second layer are returned as the final representations of the protein-celltype-tissue and metagraph.
        The function also includes a metagraph decoder that learns the weights for the metagraph relations.
        The metagraph decoder is initialized with Xavier uniform initialization.
        The function takes the following arguments:

        Args:
            ppi_x (dict): Dictionary of protein-celltype-tissue representations.
            mg_x (torch.Tensor): Metagraph representations.
            ppi_metapaths (list): List of PPI metapaths.
            mg_metapaths (list): List of metagraph metapaths.
            ppi_edge_index (torch.Tensor): Edge index for PPI graph.
            mg_edge_index (torch.Tensor): Edge index for metagraph.
            tissue_neighbors (dict): Dictionary of tissue neighbors.
        Returns:
            ppi_x (dict): Updated dictionary of protein-celltype-tissue representations.
            mg_x (torch.Tensor): Updated metagraph representations.
        """
        ########################################
        # Complete layer #1
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, mg_x = self.conv1_up(
            ppi_x, mg_x, ppi_metapaths, mg_metapaths,
            ppi_edge_index, mg_edge_index, tissue_neighbors, init_cci=True)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv1_down(
            ppi_x, ppi_metapaths, mg_x, self.conv1_up.ppi_attn)

        ########################################
        # Apply Leaky ReLU, dropout, and normalize
        ########################################
        for celltype, x in ppi_x.items():
            ppi_x[celltype] = self.layer_norm1(x)
            ppi_x[celltype] = F.leaky_relu(ppi_x[celltype])
            ppi_x[celltype] = self.batch_norm1(ppi_x[celltype])
            ppi_x[celltype] = F.dropout(ppi_x[celltype], p = self.dropout, training = self.training)
            
        mg_x = self.layer_norm1(mg_x)
        mg_x = F.leaky_relu(mg_x)
        mg_x = self.batch_norm1(mg_x)
        mg_x = F.dropout(mg_x, p = self.dropout, training = self.training)

        ########################################
        # Complete layer #2
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, mg_x = self.conv2_up(
            ppi_x, mg_x, ppi_metapaths, mg_metapaths,
            ppi_edge_index, mg_edge_index, tissue_neighbors)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv2_down(
            ppi_x, ppi_metapaths, mg_x, self.conv2_up.ppi_attn)

        return ppi_x, mg_x
