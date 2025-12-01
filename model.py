"""
Module purpose:
    MeshGraphNets-inspired graph neural network for semiconductor field surrogates.
Inputs:
    torch_geometric.data.Data with x [N, F], edge_index [2, E], pos [N, 2].
Outputs:
    Dictionary of node-wise predictions per output head; each value has shape [N, 1].
"""

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

import config


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Create a simple MLP.
    Inputs:
        input_dim, hidden_dim, output_dim: Layer sizes.
        num_layers: Total layers including output.
        activation: "relu" or "gelu".
        dropout: Dropout probability applied between hidden layers.
    Outputs:
        nn.Sequential MLP module.
    """
    acts = {"relu": nn.ReLU(), "gelu": nn.GELU()}
    act_layer = acts.get(activation, nn.GELU())
    layers: List[nn.Module] = []
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_layer)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GraphNetBlock(MessagePassing):
    """
    Residual graph network block with edge and node MLPs.
    Inputs:
        x: [N, H] node embeddings.
        edge_index: [2, E] edges.
        pos: [N, 2] coordinates for relative geometry.
    Outputs:
        Updated node embeddings [N, H].
    """

    def __init__(self, hidden_dim: int, activation: str, dropout: float) -> None:
        super().__init__(aggr="mean")
        self.edge_mlp = build_mlp(
            input_dim=hidden_dim * 2 + 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )
        self.node_mlp = build_mlp(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x_res = x
        out = self.propagate(edge_index=edge_index, x=x, pos=pos)
        out = self.node_mlp(torch.cat([x, out], dim=-1))
        out = x_res + out  # residual connection
        out = self.norm(out)
        return out

    def message(self, x_i, x_j, pos_i, pos_j):
        rel = pos_j - pos_i
        edge_input = torch.cat([x_i, x_j, rel], dim=-1)
        return self.edge_mlp(edge_input)


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet-style surrogate model with multi-head decoders.
    Inputs:
        Data.x: [N, F_in] normalized node features.
        Data.edge_index: [2, E] edges.
        Data.pos: [N, 2] raw coordinates (used for relative edge features).
    Outputs:
        Dict mapping field name to predictions [N, 1].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_message_passing_steps: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        output_fields: Optional[Iterable[str]] = None,
        target_field: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.target_field = target_field or config.OUTPUT_FIELD
        self.output_fields = list(output_fields) if output_fields is not None else config.AVAILABLE_OUTPUT_FIELDS
        self.activation = activation

        self.input_proj = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList(
            [
                GraphNetBlock(hidden_dim=hidden_dim, activation=activation, dropout=dropout)
                for _ in range(num_message_passing_steps)
            ]
        )

        self.decoder_heads = nn.ModuleDict()
        for field in self.output_fields:
            self.decoder_heads[field] = build_mlp(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=2,
                activation=activation,
                dropout=dropout,
            )

    def forward(
        self,
        data,
        fields: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Inputs:
            data: PyG Data object with x, edge_index, pos.
            fields: Iterable of field names to decode; defaults to target_field.
        Outputs:
            Dict of {field_name: predictions [N, 1]}.
        """
        x = data.x
        edge_index = data.edge_index
        pos = data.pos

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x=x, edge_index=edge_index, pos=pos)

        requested_fields = list(fields) if fields is not None else [self.target_field]
        outputs: Dict[str, torch.Tensor] = {}
        for field in requested_fields:
            head = self.decoder_heads[field]
            outputs[field] = head(x)
        return outputs
