
from typing import Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    OptPairTensor,
    OptTensor,
    Size
)

from torch_geometric.utils import scatter
from torch import nn


class RandomWalk_weighted(MessagePassing):
    """Weighted neighbor averaging with a GIN-style self term and a learnable projection.

    How Powerful are Graph Neural Networks? https://arxiv.org/abs/1810.00826
    Wasserstein Weisfeiler-Lehman Graph Kernels: https://arxiv.org/abs/1906.01277"""
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False,
                 **kwargs):
        """Store the node projection and fixed or trainable self-loop coefficient."""
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the node projection and restore the initial self-loop coefficient."""
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Tensor,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        """Project the weighted neighbor average plus the scaled self feature."""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        deg = scatter(edge_weight, edge_index[1], dim=0, dim_size=out.size(0), reduce='sum')
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        out = deg_inv.view(-1, 1) * out

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """Weight each source feature before summing messages at the target."""
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        """Show the node projection used by this convolution."""
        return f'{self.__class__.__name__}(nn={self.nn})'


class ACM_RandomWalk_conv(MessagePassing):
    """Mix low-pass, high-pass and identity channels with learned node-wise attention.

    Low/high-pass channels use (x + A_rw x)/2 and (x - A_rw x)/2.
    Revisiting Heterophily for Graph Neural Networks: https://arxiv.org/abs/2210.07606"""
    def __init__(self,
                 nn_lowpass: torch.nn.Module,
                 nn_highpass: torch.nn.Module,
                 nn_fullpass: torch.nn.Module,
                 nn_lowpass_proj: torch.nn.Module,
                 nn_highpass_proj: torch.nn.Module,
                 nn_fullpass_proj: torch.nn.Module,
                 nn_mix:torch.nn.Module,
                 T:float = 3.,
                 **kwargs):
        """Register three channel projections and their node-wise attention mixing network."""
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn_lowpass = nn_lowpass
        self.nn_highpass = nn_highpass
        self.nn_fullpass = nn_fullpass
        self.nn_lowpass_proj = nn_lowpass_proj
        self.nn_highpass_proj = nn_highpass_proj
        self.nn_fullpass_proj = nn_fullpass_proj
        self.nn_mix = nn_mix
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.T = T
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the three channel networks and attention projections."""
        reset(self.nn_lowpass)
        reset(self.nn_highpass)
        reset(self.nn_fullpass)
        reset(self.nn_lowpass_proj)
        reset(self.nn_highpass_proj)
        reset(self.nn_fullpass_proj)
        reset(self.nn_mix)

    def forward(
        self, x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        size: Size = None) -> Tensor:
        """Mix projected low-pass, high-pass and self features using learned attention."""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor, edge_index: OptTensor, edge_weight: Optional[Tensor])
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x[1].dtype,
                                 device=edge_index.device)
        out = self.propagate(x=x, edge_index=edge_index, edge_weight=edge_weight,
                                    size=size)
        deg = scatter(edge_weight, edge_index[1], dim=0, dim_size=out.size(0), reduce='sum')
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        out = deg_inv.view(-1, 1) * out

        x_r = x[1]
        if x_r is not None:
            out_lowpass = (x_r + out) / 2.
            out_highpass = (x_r - out) / 2.

        # compute embeddings for each filter
        out_lowpass = self.nn_lowpass(out_lowpass)
        out_highpass = self.nn_highpass(out_highpass)
        out_fullpass = self.nn_fullpass(x_r)
        # compute importance weights per filter
        alpha_lowpass = self.sigmoid(self.nn_lowpass_proj(out_lowpass))
        alpha_highpass = self.sigmoid(self.nn_highpass_proj(out_highpass))
        alpha_fullpass = self.sigmoid(self.nn_fullpass_proj(out_fullpass))
        alpha_cat = torch.concat([alpha_lowpass, alpha_highpass, alpha_fullpass], dim=1)
        alpha_cat = self.softmax(self.nn_mix(alpha_cat / self.T))

        out = alpha_cat[:, 0].view(-1, 1) * out_lowpass
        out = out + alpha_cat[:, 1].view(-1, 1) * out_highpass
        out = out + alpha_cat[:, 2].view(-1, 1) * out_fullpass

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """Send source features, scaled by edge weights when supplied."""
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        """Show the attention temperature used by this convolution."""
        return (f'{self.__class__.__name__}({self.T}')


class ACM_RandomWalk_weightedconv(MessagePassing):
    """ACM with supplied sampling weights for GraphSAINT message passing.

    Revisiting Heterophily for Graph Neural Networks: https://arxiv.org/abs/2210.07606"""
    def __init__(self,
                 nn_lowpass: torch.nn.Module,
                 nn_highpass: torch.nn.Module,
                 nn_fullpass: torch.nn.Module,
                 nn_lowpass_proj: torch.nn.Module,
                 nn_highpass_proj: torch.nn.Module,
                 nn_fullpass_proj: torch.nn.Module,
                 nn_mix:torch.nn.Module,
                 T:float = 3.,
                 **kwargs):
        """Register ACM channel and attention networks for sampling-weighted propagation."""
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn_lowpass = nn_lowpass
        self.nn_highpass = nn_highpass
        self.nn_fullpass = nn_fullpass
        self.nn_lowpass_proj = nn_lowpass_proj
        self.nn_highpass_proj = nn_highpass_proj
        self.nn_fullpass_proj = nn_fullpass_proj
        self.nn_mix = nn_mix
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.T = T
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the channel networks and their attention projections."""
        reset(self.nn_lowpass)
        reset(self.nn_highpass)
        reset(self.nn_fullpass)
        reset(self.nn_lowpass_proj)
        reset(self.nn_highpass_proj)
        reset(self.nn_fullpass_proj)
        reset(self.nn_mix)

    def forward(
        self, x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_weight: OptTensor,
        size: Size = None) -> Tensor:


        """Mix ACM channels using GraphSAINT weights or random-walk weights at evaluation."""
        if edge_weight is None:
            # compute weights corresponding to randomwalk matrix
            ones_ = torch.ones(edge_index.shape[1], dtype=x.dtype, device=x.device)

            deg = scatter(ones_, edge_index[1], dim=0, dim_size=x.shape[0], reduce='sum')
            deg_inv = 1. / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[edge_index[0, :]]
            self_weight = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)

        else:
            self_weight = scatter(edge_weight, edge_index[0], dim=0, dim_size=x.shape[0], reduce='sum')

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor, edge_index: OptTensor, edge_weight: Optional[Tensor])


        out = self.propagate(x=x, edge_index=edge_index, edge_weight=edge_weight,
                            size=size)

        x_r = self_weight[:, None] * x[1]

        if x_r is not None:
            out_lowpass = (x_r + out) / 2.
            out_highpass = (x_r - out) / 2.

        # compute embeddings for each filter
        out_lowpass = self.nn_lowpass(out_lowpass)
        out_highpass = self.nn_highpass(out_highpass)
        out_fullpass = self.nn_fullpass(x_r)
        # compute importance weights per filter
        alpha_lowpass = self.sigmoid(self.nn_lowpass_proj(out_lowpass))
        alpha_highpass = self.sigmoid(self.nn_highpass_proj(out_highpass))
        alpha_fullpass = self.sigmoid(self.nn_fullpass_proj(out_fullpass))
        alpha_cat = torch.concat([alpha_lowpass, alpha_highpass, alpha_fullpass], dim=1)
        alpha_cat = self.softmax(self.nn_mix(alpha_cat / self.T))

        out = alpha_cat[:, 0].view(-1, 1) * out_lowpass
        out = out + alpha_cat[:, 1].view(-1, 1) * out_highpass
        out = out + alpha_cat[:, 2].view(-1, 1) * out_fullpass

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """Scale source features by the supplied sampling/propagation weights."""
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        """Show the attention temperature used by the weighted convolution."""
        return (f'{self.__class__.__name__}({self.T}')


class ACM_RandomWalk(nn.Module):
    """Project and normalize the three ACM channels before mixing them at each node."""
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 gnn_activation='relu',
                 use_batchnorm=True,
                 T=3.,
                 graph_saint_norm: bool = False,
                 device='cpu'):

        """Build the three channel MLPs and choose ordinary or GraphSAINT-weighted propagation."""
        super(ACM_RandomWalk, self).__init__()
        # set hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_batchnorm = use_batchnorm
        self.gnn_activation = gnn_activation
        assert self.gnn_activation in ['relu', 'leaky_relu']
        self.T = T
        self.graph_saint_norm = graph_saint_norm

        # set neural networks for mixing of each filter
        if self.use_batchnorm:
            self.nn_lowpass = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                )

            self.nn_highpass = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                )
            self.nn_fullpass = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                )
        else:
            self.nn_lowpass = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                )

            self.nn_highpass = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                )
            self.nn_fullpass = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU() if self.gnn_activation == 'relu' else nn.LeakyReLU(0.2),
                )


        self.nn_lowpass_proj = nn.Linear(self.hidden_dim, 1)
        self.nn_highpass_proj = nn.Linear(self.hidden_dim, 1)
        self.nn_fullpass_proj = nn.Linear(self.hidden_dim, 1)
        self.nn_mix = nn.Linear(3, 3)

        if not self.graph_saint_norm:
            self.ACM_conv = ACM_RandomWalk_conv(
                self.nn_lowpass,
                self.nn_highpass,
                self.nn_fullpass,
                self.nn_lowpass_proj,
                self.nn_highpass_proj,
                self.nn_fullpass_proj,
                self.nn_mix,
                self.T)
        else:
            self.ACM_conv = ACM_RandomWalk_weightedconv(
                self.nn_lowpass,
                self.nn_highpass,
                self.nn_fullpass,
                self.nn_lowpass_proj,
                self.nn_highpass_proj,
                self.nn_fullpass_proj,
                self.nn_mix,
                self.T)

    def reset_parameters(self):
        """Reinitialize linear, normalization and ACM convolution parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, ACM_RandomWalk_conv):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """Apply the configured ACM convolution to node features and graph edges."""
        return self.ACM_conv(x, edge_index, edge_weight)
