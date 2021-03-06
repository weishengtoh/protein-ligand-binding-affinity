"""Custom model components (encoders and regressors) to build a custom model

ENCODERS:
    1. Conv1DEncoder(feature_dim: int, out_channels: List[int], 
                     kernel_size: List[int], embed_dim: int, 
                     activation: str = 'ReLU', dropout: float = 0.2)
    2. DenseEncoder(feature_dim: int, out_features: List, 
                    activation: str = 'ReLU', dropout: float = 0.2)
    3. TransformerEncoder(feature_dim: int, nhead: int, dim_hid: int,
                          nlayers: int, dropout: float = 0.2)
    4. GCNEncoder(out_channels: List[int], feature_dim: int,
                  activation: str = 'ReLU', dropout: float = 0.2)
    5. GATEncoder(out_channels: List[int], feature_dim: int, heads: List[int], 
                  activation: str = 'ReLU', dropout: float = 0.2)
    6. PoolingEncoder(pooling: str)
    7. DGL_GIN_AttrMasking(feature_dim: int, predictor_dim: int, 
                        dropout: float = 0.1, pooling: str = 'average')
    8. DGL_GIN_ContextPred(feature_dim: int, predictor_dim: int, 
                        dropout: float = 0.1, pooling: str = 'average')
    9. DGL_GIN_EdgePred(feature_dim: int, predictor_dim: int, 
                        dropout: float = 0.1, pooling: str = 'average')
    10. DGL_GIN_InfoMax(feature_dim: int, predictor_dim: int, 
                        dropout: float = 0.1, pooling: str = 'average')

REGRESSORS:
    1. DenseRegressor(feature_dim: int, out_features: List, 
                      activation: str = 'ReLU', dropout: float = 0.2)
    2. Conv1DRegressor(feature_dim: int, out_channels: List[int],
                       kernel_size: List[int], dense_nodes: int,
                       activation: str = 'ReLU', dropout: float = 0.2))
"""

import math
from typing import List

import torch
from torch import nn
from dgllife.model import load_pretrained
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling
from dgl.heterograph import DGLHeteroGraph
from torch_geometric.nn import (GATConv, GCNConv, global_max_pool,
                                global_mean_pool)

############### ENCODER COMPONENTS DEFINED HERE ###############


class DGL_GIN_AttrMasking(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self,
                 feature_dim: int = 300,
                 predictor_dim: int = 256,
                 dropout: int = 0.1,
                 pooling: str = 'average'):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.gnn = load_pretrained('gin_supervised_masking')

        if pooling == 'average':
            self.pool = AvgPooling()
        elif pooling == 'max':
            self.pool = MaxPooling()

        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(feature_dim, predictor_dim))

    def forward(self, x: DGLHeteroGraph) -> torch.Tensor:

        node_feats = [
            x.ndata.pop('atomic_number'),
            x.ndata.pop('chirality_type')
        ]
        edge_feats = [
            x.edata.pop('bond_type'),
            x.edata.pop('bond_direction_type')
        ]

        node_feats = self.gnn(x, node_feats, edge_feats)
        graph_feats = self.pool(x, node_feats)
        return torch.squeeze(self.dense(graph_feats))


class DGL_GIN_ContextPred(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self,
                 feature_dim: int = 300,
                 predictor_dim: int = 256,
                 dropout: int = 0.1,
                 pooling: str = 'average'):
        super().__init__()

        ## this is fixed hyperparameters as it is a pretrained model
        self.gnn = load_pretrained('gin_supervised_contextpred')

        if pooling == 'average':
            self.pool = AvgPooling()
        elif pooling == 'max':
            self.pool = MaxPooling()

        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(feature_dim, predictor_dim))

    def forward(self, x: DGLHeteroGraph) -> torch.Tensor:

        node_feats = [
            x.ndata.pop('atomic_number'),
            x.ndata.pop('chirality_type')
        ]

        edge_feats = [
            x.edata.pop('bond_type'),
            x.edata.pop('bond_direction_type')
        ]

        node_feats = self.gnn(x, node_feats, edge_feats)
        graph_feats = self.pool(x, node_feats)
        return torch.squeeze(self.dense(graph_feats))


class DGL_GIN_InfoMax(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self,
                 feature_dim: int = 300,
                 predictor_dim: int = 256,
                 dropout: int = 0.1,
                 pooling: str = 'average'):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.gnn = load_pretrained('gin_supervised_infomax')

        if pooling == 'average':
            self.pool = AvgPooling()
        elif pooling == 'max':
            self.pool = MaxPooling()

        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(feature_dim, predictor_dim))

    def forward(self, x: DGLHeteroGraph) -> torch.Tensor:

        node_feats = [
            x.ndata.pop('atomic_number'),
            x.ndata.pop('chirality_type')
        ]
        edge_feats = [
            x.edata.pop('bond_type'),
            x.edata.pop('bond_direction_type')
        ]

        node_feats = self.gnn(x, node_feats, edge_feats)
        graph_feats = self.pool(x, node_feats)
        return torch.squeeze(self.dense(graph_feats))


class DGL_GIN_EdgePred(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self,
                 feature_dim: int = 300,
                 predictor_dim: int = 256,
                 dropout: int = 0.1,
                 pooling: str = 'average'):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.gnn = load_pretrained('gin_supervised_edgepred')

        if pooling == 'average':
            self.pool = AvgPooling()
        elif pooling == 'max':
            self.pool = MaxPooling()

        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(feature_dim, predictor_dim))

    def forward(self, x: DGLHeteroGraph) -> torch.Tensor:

        node_feats = [
            x.ndata.pop('atomic_number'),
            x.ndata.pop('chirality_type')
        ]
        edge_feats = [
            x.edata.pop('bond_type'),
            x.edata.pop('bond_direction_type')
        ]

        node_feats = self.gnn(x, node_feats, edge_feats)
        graph_feats = self.pool(x, node_feats)
        return torch.squeeze(self.dense(graph_feats))


class Conv1DEncoder(nn.Module):
    """Encode a feature using a stack of Conv1D layers
    
    Input -> Embedding -> Stack of Conv1D layers -> GlobalMaxPool -> Output
    
    """

    def __init__(self,
                 feature_dim: int,
                 out_channels: List[int],
                 kernel_size: List[int],
                 embed_dim: int = None,
                 activation: str = 'ReLU',
                 dropout: float = 0.2,
                 pooling: str = 'max',
                 batch_norm: bool = False):
        """Inits Conv1DEncoder 

        Number of Conv1D layers generated is determined by the length of 'out_channels'.
        Expects 'out_channels' and 'kernel_size' to be of the same length.

        Applies a Global Max Pooling layer on top of the final Conv1D layer

        Args: 
            feature_dim (int): 
                input feature dimension
            out_channels (List[int]): 
                List of out_channels for the Conv1D layers 
            kernel_size (List[int]):
                List of kernel size for the Conv1D layers. 
            embed_dim (int):
                Embedding dimension to be used for the input
            activation (str):
                Activation function to use from torch.nn (e.g. 'ReLU', 'Tanh', 'ELU' etc.)
            dropout (float): 
                Dropout ratio

        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        if pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'average':
            self.pool = nn.AdaptiveAvgPool1d(1)

        if embed_dim is not None:
            self.embedding = nn.Embedding(feature_dim + 1, embed_dim)
            in_c = embed_dim
        else:
            self.embedding = None
            in_c = feature_dim

        self.layers_bn = None
        if batch_norm:
            self.layers_bn = nn.ModuleList()
            for out_c in out_channels:
                self.layers_bn.append(nn.BatchNorm1d(out_c))

        self.layers = nn.ModuleList()
        for out_c, k in zip(out_channels, kernel_size):
            self.layers.append(nn.Conv1d(in_c, out_c, k))
            in_c = out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.embedding is not None:
            x = self.embedding(x)
            x = torch.transpose(x, 1, 2)

        final_layer = self.layers[-1]

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.layers_bn[i](x) if self.layers_bn is not None else x
            x = self.activation(x)
            x = self.dropout(x)

        x = final_layer(x)
        x = self.pool(x)

        return torch.squeeze(x)


class DenseEncoder(nn.Module):
    """Encode a feature using a stack of Dense layers

        Input -> Stack of Dense layers -> Output

    """

    def __init__(self,
                 feature_dim: int,
                 out_features: List[int],
                 activation: str = 'ReLU',
                 dropout: float = 0.2):
        """Inits DenseEncoder

        Number of Dense layers is determined by length of 'out_features'

        Args:
            feature_dim (int):
                Input feature dimension 
            out_features (List[int]):
                List of int representing the number of nodes
            activation (str):
                Activation function to use from torch.nn (e.g. 'ReLU', 'Tanh', 'ELU' etc.)
            dropout (float):
                Dropout ratio
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        self.layers = nn.ModuleList()
        in_c = feature_dim
        for out_c in out_features:
            self.layers.append(nn.Linear(in_c, out_c))
            in_c = out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        final_layer = self.layers[-1]

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        return final_layer(x)


class TransformerEncoder(nn.Module):
    """Encode a feature using a stack of Transformer Encoder layers
        
    """

    def __init__(self,
                 feature_dim: int,
                 nhead: int,
                 dim_hid: int,
                 nlayers: int,
                 dropout: float = 0.2,
                 pooling: str = 'max'):
        """Inits a Transformer Encoder

        Number of encoder layers used is determined by 'nlayers'

        Args:
            feature_dim (int):
                Input feature dimension
            nhead (int):
                Number of attention heads
            dim_hid (int):
                Dimension of feedforward network layer
            nlayers (int)
                Number of Transformer encoder layers
            dropout (float):
                Dropout ratio
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding(feature_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(feature_dim, nhead, dim_hid,
                                                    dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

        if pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'average':
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = _generate_square_subsequent_mask(src.size(dim=0)).to(device)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, mask)
        x = self.pool(x)

        return torch.squeeze(x)


class GCNEncoder(nn.Module):
    """Encode a feature using a stack of Graph convolutional layers

    Expects input to be of type torch_geometric.data.Data

    Input -> Stack of GCNConv layers -> GlobalMaxPool -> Output
    
    """

    def __init__(self,
                 out_channels: List[int],
                 feature_dim: int,
                 activation: str = 'ReLU',
                 dropout: int = 0.2,
                 pooling: str = 'max',
                 batch_norm: bool = False):
        """Inits a Graph Convolutional Encoder

        Number of layers is determined by the length of 'out_channels'

        Args:
            out_channels (List[int]):
                List of number of attention heads in each layer
            feature_dim (int):
                Input feature dimension
            activation (str):
                Activation function to use from torch.nn (e.g. 'ReLU', 'Tanh', 'ELU' etc.)
            dropout (int):
                Dropout ratio
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        self.layers_bn = None
        if batch_norm:
            self.layers_bn = nn.ModuleList()
            for out_c in out_channels:
                self.layers_bn.append(nn.BatchNorm1d(out_c))

        self.layers = nn.ModuleList()
        in_c = feature_dim
        for out_c in out_channels:
            self.layers.append(GCNConv(in_c, out_c))
            in_c = out_c

        if pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'average':
            self.pool = global_mean_pool

    def forward(self, input) -> torch.Tensor:

        x, edge_index = input['x'], input['edge_index']
        batch = input['batch']

        final_layer = self.layers[-1]

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.layers_bn[i](x) if self.layers_bn is not None else x
            x = self.activation(x)
            x = self.dropout(x)

        x = final_layer(x, edge_index)
        x = self.pool(x, batch)

        return torch.squeeze(x)


class GATEncoder(nn.Module):
    """Encode a feature using a stack of Graph attention layers

    Expects input to be of type torch_geometric.data.Data

    Input -> Stack of GATConv layers -> GlobalMaxPool -> Output
        
    """

    def __init__(self,
                 out_channels: List[int],
                 feature_dim: int,
                 heads: List[int],
                 activation: str = 'ReLU',
                 dropout: int = 0.2,
                 pooling: str = 'max'):
        """Inits a Graph attention encoder 

        Args:
            out_channels (List[int]):
                List of output size for each layer
            feature_dim (int):
                Input feature dimension
            heads: List[int]:
                List of number of heads in each layer
            activation (str):
                Activation function to use from torch.nn (e.g. 'ReLU', 'Tanh', 'ELU' etc.)
            dropout (int):
                Dropout ratio
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        self.layers = nn.ModuleList()
        in_c = feature_dim
        for h, out_c in zip(heads, out_channels):
            self.layers.append(GATConv(in_c, out_c, heads=h, dropout=dropout))
            in_c = out_c * h

        if pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'average':
            self.pool = global_mean_pool

    def forward(self, input) -> torch.Tensor:

        x, edge_index = input['x'], input['edge_index']
        batch = input['batch']

        final_layer = self.layers[-1]

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        x = final_layer(x, edge_index)
        x = self.pool(x, batch)

        return torch.squeeze(x)


class PoolingEncoder(nn.Module):

    def __init__(self, pooling: str):
        super().__init__()

        if pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'average':
            self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.transpose(x, 1, 2)
        x = self.pooling(x)

        return torch.squeeze(x)


############### REGRESSOR COMPONENTS DEFINED HERE ###############


class DenseRegressor(nn.Module):
    """Produce a regression output using a stack of dense layers

    Input -> Stack of Dense Layers -> Output

    """

    def __init__(self,
                 feature_dim: int,
                 out_features: List,
                 activation: str = 'ReLU',
                 dropout: float = 0.2):
        """Inits a stack of dense layers as a regressor

        Number of layers are determined by length of 'out_features'

        Args:
            feature_dim (int):
                Input dimension
            out_features (List[int]):
                Number of nodes in each layer
            activation (str):
                Activation function to use from torch.nn (e.g. 'ReLU', 'Tanh', 'ELU' etc.)
            dropout (float):
                Dropout ratio
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        self.layers = nn.ModuleList()
        in_c = feature_dim
        for out_c in out_features:
            self.layers.append(nn.Linear(in_c, out_c))
            in_c = out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        final_layer = self.layers[-1]

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        return final_layer(x)


class Conv1DRegressor(nn.Module):
    """Model to produce a regression output using a stack of Conv1D layers
    
    Input -> Stack of Conv1D layers -> One Dense Layer -> Output
    """

    def __init__(self,
                 feature_dim: int,
                 out_channels: List[int],
                 kernel_size: List[int],
                 dense_nodes: int,
                 activation: str = 'ReLU',
                 dropout: float = 0.2):
        """Inits a stack of Conv1D as a regressor

        Args:
            feature_dim (int):
                Input dimension
            out_channels (List[int]):
                Number of channels in each layer
            kernel_size (List[int):
                Kernel size in each layer
            dense_nodes (int):
                Number of nodes in the final linear layer
            activation (str):
                Activation function to use from torch.nn (e.g. 'ReLU', 'Tanh', 'ELU' etc.)
            dropout (float):
                Dropout ratio
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()
        self.reg = nn.Linear(in_features=dense_nodes, out_features=1)

        self.layers = nn.ModuleList()
        in_c = feature_dim
        for out_c, k in zip(out_channels, kernel_size):
            self.layers.append(nn.Conv1d(in_c, out_c, k))
            in_c = out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        final_layer = self.layers[-1]

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        x = final_layer(x)
        x = torch.flatten(x, 1)

        return self.reg(x)


############### OTHER COMPONENTS DEFINED HERE ###############


def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
