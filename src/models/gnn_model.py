import torch
from torch.nn import PReLU
from torch_geometric.nn import (
    HeteroConv,
    GATv2Conv,
    Linear,
    HeteroDictLinear,
    BatchNorm,
)
from src.utils.helpers import timeit


class ProteinGNN(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.edge_attrs = config["model"]["edge_attrs"]
        hidden_channels = config["model"]["hidden_channels"]
        out_channels = dataset.go_vocab_size
        node_types = ["protein", "aa"]
        self.lin_in = HeteroDictLinear(
            in_channels=-1,
            out_channels=hidden_channels,
            types=node_types,
        )
        self.prelu1 = torch.nn.ModuleDict(
            {node_type: PReLU(hidden_channels) for node_type in node_types}
        )
        self.bn1 = torch.nn.ModuleDict(
            {node_type: BatchNorm(hidden_channels) for node_type in node_types}
        )
        self.conv1 = HeteroConv(
            {
                ("aa", "belongs_to", "protein"): GATv2Conv(
                    in_channels=(-1, -1),
                    out_channels=hidden_channels,
                    add_self_loops=False,
                ),
                ("protein", "aligned_with", "protein"): GATv2Conv(
                    (-1, -1),
                    hidden_channels,
                    add_self_loops=False,
                    edge_dim=(
                        dataset.data.num_edge_features[
                            ("protein", "aligned_with", "protein")
                        ]
                        if self.edge_attrs
                        else None
                    ),
                ),
            },
            aggr="sum",
        )
        self.prelu_gnn1 = PReLU()
        self.bn_gnn1 = BatchNorm(hidden_channels)

        self.conv2 = HeteroConv(
            {
                ("aa", "belongs_to", "protein"): GATv2Conv(
                    in_channels=(-1, -1),
                    out_channels=hidden_channels,
                    add_self_loops=False,
                ),
                ("protein", "aligned_with", "protein"): GATv2Conv(
                    (-1, -1),
                    hidden_channels,
                    add_self_loops=False,
                    edge_dim=(
                        dataset.data.num_edge_features[
                            ("protein", "aligned_with", "protein")
                        ]
                        if self.edge_attrs
                        else None
                    ),
                ),
            },
            aggr="sum",
        )
        self.prelu_gnn2 = PReLU()
        self.bn_gnn2 = BatchNorm(hidden_channels)

        self.prelu_prot = PReLU()
        self.bn_prot = BatchNorm(hidden_channels * 2)  # Due to SkipCat

        self.lin_post = Linear(hidden_channels * 2, hidden_channels)
        self.prelu_post = PReLU()
        self.bn_post = BatchNorm(hidden_channels)

        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch):
        edge_attrs_dict = batch.edge_attr_dict if self.edge_attrs else None

        # Input linear
        x_dict = self.lin_in(x_dict)
        x_dict = {k: self.prelu1[k](v) for k, v in x_dict.items()}
        x_dict = {k: self.bn1[k](v) for k, v in x_dict.items()}

        # GNN layer 1
        if self.edge_attrs:
            x_gnn1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attrs_dict)
        else:
            x_gnn1 = self.conv1(x_dict, edge_index_dict)
        x_gnn1 = {k: self.prelu_gnn1(v) for k, v in x_gnn1.items()}
        x_gnn1 = {k: self.bn_gnn1(v) for k, v in x_gnn1.items()}
        x_gnn1["aa"] = x_dict["aa"]

        # GNN layer 2
        if self.edge_attrs:
            x_gnn2 = self.conv2(x_gnn1, edge_index_dict, edge_attr_dict=edge_attrs_dict)
        else:
            x_gnn2 = self.conv2(x_gnn1, edge_index_dict)
        x_gnn2 = {k: self.prelu_gnn2(v) for k, v in x_gnn2.items()}
        x_gnn2 = {k: self.bn_gnn2(v) for k, v in x_gnn2.items()}

        # SkipCat: concatenate transformed input features and GNN output
        x_prot = torch.cat(
            [
                x_dict["protein"][: batch["protein"].batch_size],
                x_gnn2["protein"][: batch["protein"].batch_size],
            ],
            dim=1,
        )
        x_prot = self.prelu_prot(x_prot)
        x_prot = self.bn_prot(x_prot)

        # Post-process lin layers
        x_prot = self.lin_post(x_prot)
        x_prot = self.prelu_post(x_prot)
        x_prot = self.bn_post(x_prot)

        x_prot = self.lin_out(x_prot)

        return x_prot
