import random

import numpy as np

import torch
import torch.nn.functional as F

from e3nn.o3._spherical_harmonics import _spherical_harmonics


# From Meta:
# https://github.com/facebookresearch/fairchem @d513ffa6f638a0f4c0f16e31dfa8f7469b36bb2f

@torch.jit.script
def gaussian(x: torch.Tensor, mean, std) -> torch.Tensor:
    a = (2 * math.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


# Different encodings for the atom distance embeddings
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SigmoidSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_sigmoid=50, basis_width_scalar=1.0
    ) -> None:
        super().__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp_dist)


class LinearSigmoidSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_sigmoid: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp_dist) + 0.001 * exp_dist


class SiLUSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_output: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_output
        self.fc1 = nn.Linear(2, num_output)
        self.act = nn.SiLU()

    def forward(self, dist):
        x_dist = dist.view(-1, 1)
        x_dist = torch.cat([x_dist, torch.ones_like(x_dist)], dim=1)
        return self.act(self.fc1(x_dist))


# From Eric Qu & Aditi S. Krishnapriyan, "The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains", NeurIPS 2024
# See https://github.com/ASK-Berkeley/EScAIP

def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.jit.script
def get_node_direction_expansion(
    distance_vec: torch.Tensor, edge_index: torch.Tensor, lmax: int, num_nodes: int
):
    """
    Calculate Bond-Orientational Order (BOO) for each node in the graph.
    Ref: Steinhardt, et al. "Bond-orientational order in liquids and glasses." Physical Review B 28.2 (1983): 784.
    Return: (N, )
    """
    distance_vec = F.normalize(distance_vec, dim=-1)
    edge_sh = _spherical_harmonics(
        lmax=lmax,
        x=distance_vec[:, 0],
        y=distance_vec[:, 1],
        z=distance_vec[:, 2],
    )
    node_boo = torch.zeros((num_nodes, edge_sh.shape[1]), device=edge_sh.device)
    node_boo = scatter(edge_sh, edge_index[1], dim=0, out=node_boo, reduce="mean")
    sh_index = torch.arange(lmax + 1, device=node_boo.device)
    sh_index = torch.repeat_interleave(sh_index, 2 * sh_index + 1)
    node_boo = scatter(node_boo**2, sh_index, dim=1, reduce="sum").sqrt()
    return node_boo


def convert_neighbor_list(edge_index: torch.Tensor, max_neighbors: int, num_nodes: int):
    """
    Convert edge_index to a neighbor list format.
    """
    src = edge_index[0, :]
    dst = edge_index[1, :]

    # Count the number of neighbors for each node
    neighbor_counts = torch.bincount(dst, minlength=num_nodes)

    # Calculate the offset for each node
    offset = max_neighbors - neighbor_counts
    offset = torch.cat(
        [torch.tensor([0], device=offset.device), torch.cumsum(offset, dim=0)]
    )

    # Create an index mapping
    index_mapping = torch.arange(0, edge_index.shape[1], device=edge_index.device)

    # Calculate the indices in the neighbor list
    index_mapping = offset[dst] + index_mapping

    # Initialize the neighbor list and mask
    neighbor_list = torch.full(
        (num_nodes * max_neighbors,), -1, dtype=torch.long, device=edge_index.device
    )
    mask = torch.zeros(
        (num_nodes * max_neighbors,), dtype=torch.bool, device=edge_index.device
    )

    # Scatter the neighbors
    neighbor_list.scatter_(0, index_mapping, src)
    mask.scatter_(
        0,
        index_mapping,
        torch.ones_like(src, dtype=torch.bool, device=edge_index.device),
    )

    # Reshape to [N, max_num_neighbors]
    neighbor_list = neighbor_list.view(num_nodes, max_neighbors)
    mask = mask.view(num_nodes, max_neighbors)

    return neighbor_list, mask, index_mapping


def map_neighbor_list(x, index_mapping, max_neighbors, num_nodes):
    """
    Map from edges to neighbor list.
    x: (num_edges, h)
    index_mapping: (num_edges, )
    return: (num_nodes, max_neighbors, h)
    """
    output = torch.zeros((num_nodes * max_neighbors, x.shape[1]), device=x.device)
    output.scatter_(0, index_mapping.unsqueeze(1).expand(-1, x.shape[1]), x)
    return output.view(num_nodes, max_neighbors, x.shape[1])


def map_sender_receiver_feature(sender_feature, receiver_feature, neighbor_list):
    """
    Map from node features to edge features.
    sender_feature, receiver_feature: (num_nodes, h)
    neighbor_list: (num_nodes, max_neighbors)
    return: sender_features, receiver_features (num_nodes, max_neighbors, h)
    """
    # sender feature
    sender_feature = sender_feature[neighbor_list.flatten()].view(
        neighbor_list.shape[0], neighbor_list.shape[1], -1
    )

    # receiver features
    receiver_feature = receiver_feature.unsqueeze(1).expand(
        -1, neighbor_list.shape[1], -1
    )

    return (sender_feature, receiver_feature)


@torch.compile
def get_attn_mask(
    edge_direction: torch.Tensor,
    neighbor_mask: torch.Tensor,
    num_heads: int,
    use_angle_embedding: bool,
):
    # create a mask for empty neighbors
    batch_size, max_neighbors = neighbor_mask.shape
    attn_mask = torch.zeros(
        batch_size, max_neighbors, max_neighbors, device=neighbor_mask.device
    )
    attn_mask = attn_mask.masked_fill(~neighbor_mask.unsqueeze(1), float("-inf"))

    # repeat the mask for each head
    attn_mask = (
        attn_mask.unsqueeze(1)
        .expand(batch_size, num_heads, max_neighbors, max_neighbors)
        .reshape(batch_size * num_heads, max_neighbors, max_neighbors)
    )

    # get the angle embeddings
    dot_product = torch.matmul(edge_direction, edge_direction.transpose(1, 2))
    dot_product = (
        dot_product.unsqueeze(1)
        .expand(-1, num_heads, -1, -1)
        .reshape(batch_size * num_heads, max_neighbors, max_neighbors)
    )

    return attn_mask, dot_product


def compilable_scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    dim: int = 0,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    torch_scatter scatter function with compile support.
    Modified from torch_geometric.utils.scatter_.
    """

    def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
        dim = ref.dim() + dim if dim < 0 else dim
        size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
        return src.view(size).expand_as(ref)

    dim = src.dim() + dim if dim < 0 else dim
    size = src.size()[:dim] + (dim_size,) + src.size()[dim + 1 :]

    if reduce == "sum" or reduce == "add":
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    if reduce == "mean":
        count = src.new_zeros(dim_size)
        count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        count = count.clamp(min=1)

        index = broadcast(index, src, dim)
        out = src.new_zeros(size).scatter_add_(dim, index, src)

        return out / broadcast(count, out, dim)

    raise ValueError((f"Invalid reduce option '{reduce}'."))