"""Gated pooling from Attention-based Deep Multiple Instance Learning.
Ilse et al. (2018): https://arxiv.org/abs/1802.04712.
"""

import torch as th
from torch.nn import functional as F
from torch import nn

class GatedAttention(nn.Module):
    def __init__(self, input_dim, emb_dim):
        """Build tanh and sigmoid attention branches and their shared scalar score projection."""
        super(GatedAttention, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.ATTENTION_BRANCHES = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.emb_dim, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)


    def forward(self, x_nodes, data=None):
        # first compute unnormalized attention weights for all samples simultaneously

        """Return normalized protein attention weights and one pooled vector per graph."""
        A_V = self.attention_V(x_nodes)  # K x L
        A_U = self.attention_U(x_nodes)  # K x L
        Att_row = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES

        batch_mode = data is not None and data.batch is not None

        if batch_mode:
            list_x_graphs = []
            list_att = [] # list of normalized attention weights
            for i in range(data.ptr.shape[0] - 1):
                start_curr_graph = data.ptr[i].item()
                end_curr_graph = data.ptr[i + 1].item()
                local_att = F.softmax(
                    Att_row[start_curr_graph : end_curr_graph, :], dim=0)

                x_graphs = th.mm(
                    local_att.T,
                    x_nodes[start_curr_graph : end_curr_graph, :])
                list_x_graphs.append(x_graphs)
                list_att.append(local_att)

            x_graphs = th.cat(list_x_graphs, dim=0)
            Att = th.cat(list_att, dim=0)

        else:
            Att = F.softmax(Att_row, dim=0)

            x_graphs = th.mm(Att.T, x_nodes)

        return Att, x_graphs
