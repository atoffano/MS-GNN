import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, SAGEConv
from src.utils.helpers import timeit


class ProteinGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ("aa", "belongs_to", "protein"): GATv2Conv(
                    in_channels=(-1, -1),
                    out_channels=hidden_channels,
                    add_self_loops=False,
                ),
                ("protein", "aligned_with", "protein"): GATv2Conv(
                    (-1, -1), hidden_channels, add_self_loops=False
                ),
                # ("protein", "rev_belongs_to", "aa"): GATv2Conv(
                #     in_channels=(-1, -1),
                #     out_channels=hidden_channels,
                #     add_self_loops=False,
                # ),
            },
            aggr="sum",
        )

        # self.conv2 = HeteroConv(
        #     {
        #         ("aa", "belongs_to", "protein"): GATv2Conv(
        #             in_channels=(-1, -1),
        #             out_channels=hidden_channels,
        #             add_self_loops=False,
        #         ),
        #         ("protein", "aligned_with", "protein"): GATv2Conv(
        #             (-1, -1), hidden_channels, add_self_loops=False
        #         ),
        #         ("protein", "rev_belongs_to", "aa"): GATv2Conv(
        #             in_channels=(-1, -1),
        #             out_channels=hidden_channels,
        #             add_self_loops=False,
        #         ),
        #     },
        #     aggr="sum",
        # )

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch):
        x_dict = self.conv1(x_dict, edge_index_dict)
        # x_dict = self.conv2(x_dict, edge_index_dict)
        x_prot = x_dict["protein"][: batch["protein"].batch_size]
        x_prot = x_prot.relu()
        x_prot = self.lin(x_prot)
        return x_prot
