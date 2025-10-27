from collections import defaultdict
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


class HeteroGATConv(torch.nn.Module):
    def __init__(self, edge_types, channels):
        super().__init__()
        self.convs = torch.nn.ModuleDict()
        for edge_type in edge_types:
            key = "__".join(edge_type)
            self.convs[key] = GATv2Conv((-1, -1), channels, add_self_loops=False)

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict=None,
        return_attention_weights=None,
        **kwargs,
    ):
        out_dict = defaultdict(list)
        attention_dict = {} if return_attention_weights else None
        for edge_type, edge_index in edge_index_dict.items():
            key = "__".join(edge_type)
            x_src = x_dict[edge_type[0]]
            x_dst = x_dict[edge_type[2]]
            if return_attention_weights:
                out, attn = self.convs[key](
                    (x_src, x_dst),
                    edge_index,
                    return_attention_weights=True,
                )
                attention_dict[edge_type] = attn
            else:
                out = self.convs[key]((x_src, x_dst), edge_index)
            out_dict[edge_type[2]].append(out)

        out_dict = {
            dst: torch.stack(outputs).sum(dim=0) for dst, outputs in out_dict.items()
        }

        return out_dict, attention_dict


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
        self.conv1 = HeteroGATConv(
            edge_types=[
                # ("aa", "close_to", "aa"),
                ("aa", "belongs_to", "protein"),
                ("protein", "aligned_with", "protein"),
                # ("protein", "stringdb", "protein"),
            ],
            channels=hidden_channels,
        )
        self.prelu_gnn1 = PReLU()
        self.bn_gnn1 = BatchNorm(hidden_channels)

        self.conv2 = HeteroGATConv(
            edge_types=[
                # ("aa", "close_to", "aa"),
                ("aa", "belongs_to", "protein"),
                ("protein", "aligned_with", "protein"),
                # ("protein", "stringdb", "protein"),
            ],
            channels=hidden_channels,
        )
        self.prelu_gnn2 = PReLU()
        self.bn_gnn2 = BatchNorm(hidden_channels)

        self.prelu_prot = PReLU()
        self.bn_prot = BatchNorm(hidden_channels * 2)  # Due to SkipCat

        self.lin_post = Linear(hidden_channels * 2, hidden_channels)
        self.prelu_post = PReLU()
        self.bn_post = BatchNorm(hidden_channels)

        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch, return_attention_weights=None):
        edge_attr_dict = batch.edge_attr_dict if self.edge_attrs else None

        # Input linear
        x_dict = self.lin_in(x_dict)
        x_dict = {k: self.prelu1[k](v) for k, v in x_dict.items()}
        x_dict = {k: self.bn1[k](v) for k, v in x_dict.items()}

        # GNN layer 1
        x_gnn1, attn1 = self.conv1(
            x_dict,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            return_attention_weights=return_attention_weights,
        )
        x_gnn1 = {k: self.prelu_gnn1(v) for k, v in x_gnn1.items()}
        x_gnn1 = {k: self.bn_gnn1(v) for k, v in x_gnn1.items()}
        x_gnn1["aa"] = x_dict["aa"]  # Preserve 'aa' node features

        # GNN layer 2
        x_gnn2, attn2 = self.conv2(
            x_gnn1,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            return_attention_weights=return_attention_weights,
        )
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

        if not self.training:
            x_prot = torch.sigmoid(x_prot)

        if return_attention_weights:
            attentions = (attn1, attn2)
            return x_prot, attentions

        return x_prot
