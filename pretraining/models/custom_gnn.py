
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

from torch_geometric.utils import scatter, degree
from torch import nn



class RandomWalk_weighted(MessagePassing):
    r"""
    Fusing implementation of Graph Isomorphism Network [A]
    with message propagation framework from [B] to incorporate edge weights.

    [A] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018, September).
        How Powerful are Graph Neural Networks?. In International Conference on
        Learning Representations.

    Pygeo code: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GIN.html

    [B] Togninalli, M., Ghisu, E., Llinares-López, F., Rieck, B., & Borgwardt, K. (2019).
        Wasserstein weisfeiler-lehman graph kernels. Advances in neural information
        processing systems, 32

    Pygeo code: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.WLConvContinuous.html#torch_geometric.nn.conv.WLConvContinuous

    
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \frac{1}{deg_i}\sum_{j \in \mathcal{N}(i)} 
         \mathbf{e}_{j, i} . \mathbf{x}_j  

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|,)` 
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False,
                 **kwargs):
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
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Tensor,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

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
        return edge_weight.view(-1, 1) * x_j
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


#%%

class ACM_RandomWalk_conv(MessagePassing):
    r"""
    Adaptative Channel mixing [A] with filters:
        - low-pass filter: I + \hat{A}_{rw} = I + deg(tilde{A})^{-1} \tilde{A}
        - high-pass filter: I - \hat{A}_{rw} 
        - full-pass filter: I 
        with \tilde{A} = A + I 
    with message propagation framework from [B] to incorporate edge weights.

    [B] Luan, S., Hua, C., Lu, Q., Zhu, J., Zhao, M., Zhang, S., ... & Precup, D. (2022).
    Revisiting heterophily for graph neural networks.
    Advances in neural information processing systems, 35, 1362-1375.

    
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \frac{1}{deg_i}\sum_{j \in \mathcal{N}(i)} 
         \mathbf{e}_{j, i} . \mathbf{x}_j  

    Args:
        nn_highpass (torch.nn.Module):
            A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn_lowpass (torch.nn.Module):
            
        nn_fullpass (torch.nn.Module):
        
        nn_highpass_proj (torch.nn.Module): shape (output_dim, 1)
            
        nn_lowpass_proj (torch.nn.Module): shape (output_dim, 1)
            
        nn_fullpass_proj (torch.nn.Module): shape (output_dim, 1)
        
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|,)` 
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
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
        kwargs.setdefault('aggr', 'add') ### propagate and aggregate 
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
                
        #out = alpha_cat[:, 0][:, None] * out_lowpass
        #out = out + alpha_cat[:, 1][:, None] * out_highpass
        #out = out + alpha_cat[:, 2][:, None] * out_fullpass
        out = alpha_cat[:, 0].view(-1, 1) * out_lowpass
        out = out + alpha_cat[:, 1].view(-1, 1) * out_highpass
        out = out + alpha_cat[:, 2].view(-1, 1) * out_fullpass
        
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.T}')
    

class ACM_RandomWalk_weightedconv(MessagePassing):
    r"""
    Adaptative Channel mixing [A] with inputted weights as filters:
        - low-pass filter: I + \hat{A} = I + W @ A
        - high-pass filter: I - \hat{A} 
        - full-pass filter: I 
        
    Args:
        nn_highpass (torch.nn.Module):
            A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn_lowpass (torch.nn.Module):
            
        nn_fullpass (torch.nn.Module):
        
        nn_highpass_proj (torch.nn.Module): shape (output_dim, 1)
            
        nn_lowpass_proj (torch.nn.Module): shape (output_dim, 1)
            
        nn_fullpass_proj (torch.nn.Module): shape (output_dim, 1)
        
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|,)` 
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
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
        kwargs.setdefault('aggr', 'add') ### propagate and aggregate 
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
        
        
        if edge_weight is None:
            # compute weights corresponding to randomwalk matrix
            ones_ = torch.ones(edge_index.shape[1], dtype=x.dtype, device=x.device)
            
            deg = scatter(ones_, edge_index[1], dim=0, dim_size=x.shape[0], reduce='sum')
            deg_inv = 1. / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            # edge_weight = deg_inv_by_edge
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
                
        #out = alpha_cat[:, 0][:, None] * out_lowpass
        #out = out + alpha_cat[:, 1][:, None] * out_highpass
        #out = out + alpha_cat[:, 2][:, None] * out_fullpass
        out = alpha_cat[:, 0].view(-1, 1) * out_lowpass
        out = out + alpha_cat[:, 1].view(-1, 1) * out_highpass
        out = out + alpha_cat[:, 2].view(-1, 1) * out_fullpass
        
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.T}')
    

class ACM_RandomWalk(nn.Module):
    """
    Adaptive Channel Mixing Random Walk GNN module.

    This class implements a graph neural network layer that adaptively mixes low-pass, high-pass,
    and full-pass graph filters using learnable neural networks and attention-like mixing weights.
    It is designed to capture both homophilic and heterophilic patterns in graph data by combining
    information from different spectral channels.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Dimension of hidden layers in the filter networks.
        use_batchnorm (bool, optional): Whether to use BatchNorm1d in the filter networks. Default is True.
        T (float, optional): Temperature parameter for softmax mixing. Default is 3.0.
        device (str, optional): Device to use ('cpu' or 'cuda'). Default is 'cpu'.

    Attributes:
        nn_lowpass (nn.Sequential): Neural network for low-pass filter.
        nn_highpass (nn.Sequential): Neural network for high-pass filter.
        nn_fullpass (nn.Sequential): Neural network for full-pass filter.
        nn_lowpass_proj (nn.Linear): Projection layer for low-pass filter importance.
        nn_highpass_proj (nn.Linear): Projection layer for high-pass filter importance.
        nn_fullpass_proj (nn.Linear): Projection layer for full-pass filter importance.
        nn_mix (nn.Linear): Mixing layer for combining filter importances.
        ACM_conv (ACM_RandomWalk_conv): The underlying message passing layer.

    Methods:
        reset_parameters(): Resets all learnable parameters.
        forward(x, edge_index, edge_weight=None): Performs a forward pass of the ACM Random Walk GNN.

    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 gnn_activation='relu', 
                 use_batchnorm=True,
                 T=3.,
                 graph_saint_norm: bool = False,
                 device='cpu'):
        
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, ACM_RandomWalk_conv):
                m.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight=None):
        return self.ACM_conv(x, edge_index, edge_weight)
    
    
