import dataclasses
from pathlib import Path

from ase.io import read
from ase.io.trajectory import Trajectory

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import torch

from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.loader import DataLoader

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import generate_graph

from umpire.utils import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
    get_node_direction_expansion,
    convert_neighbor_list,
    map_neighbor_list,
    get_attn_mask
)
from umpire.config import UmpireConfig


# From Eric Qu & Aditi S. Krishnapriyan, "The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains", NeurIPS 2024
# See https://github.com/ASK-Berkeley/EScAIP
@dataclasses.dataclass
class ChemicalGraphData:
    """
    Custom dataclass for storing chemical/materials graph data
    atomic_numbers: (N)
    edge_distance_expansion: (N, max_nei, edge_distance_expansion_size)
    edge_direction: (N, max_nei, 3)
    node_direction_expansion: (N, node_direction_expansion_size)
    attn_mask: (N * num_head, max_nei, max_nei) Attention mask with angle embeddings
    angle_embedding: (N * num_head, max_nei, max_nei) Angle embeddings (cosine)
    neighbor_list: (N, max_nei)
    neighbor_mask: (N, max_nei)
    node_batch: (N)
    # node_padding_mask: (N)
    # graph_padding_mask: (num_graphs)
    """

    atomic_numbers: torch.Tensor
    edge_distance_expansion: torch.Tensor
    edge_direction: torch.Tensor
    node_direction_expansion: torch.Tensor
    attn_mask: torch.Tensor
    angle_embedding: torch.Tensor
    neighbor_list: torch.Tensor
    neighbor_mask: torch.Tensor
    node_batch: torch.Tensor
    # node_padding_mask: torch.Tensor
    # graph_padding_mask: torch.Tensor


torch.export.register_dataclass(
    ChemicalGraphData, serialized_type_name="ChemicalGraphData"
)


def data_from_ase_atoms(
    atoms: List[Atoms] | Trajectory,
    config: UmpireConfig,
    device: str | torch.device | int,
    batch: torch.Tensor | None = None
) -> ChemicalGraphData:
    """
    Construct a ChemicalGraphData object from a collection of ASE Atoms objects

    Args:
        atoms (List[Atoms] | Trajectory): Collection of snapshots to be compiled into a dataset
        config (UmpireConfig): Config object
        device (str | torch.device | int): Either a device object or a reference to a torch-recognized device
        batch (torch.Tensor | None): Either information regarding which samples will be included in which batch, or
            None (default)

    Returns:
        ChemicalGraphData

    """

    if isinstance(device, str):
        device = torch.device(device)

    positions = list()
    numbers = list()
    cells = list()
    pbcs = list()
    
    natoms = torch.tensor([len(positions)], dtype=torch.int32, device=device)
    edge_index = torch.zeros([2,1], dtype=torch.long, device=device)
    cell_offsets = torch.zeros([1,3], dtype=torch.float, device=device)
    nedges = torch.tensor([1], dtype=torch.long, device=device)
    charge = torch.zeros([len(atoms)], dtype=torch.long, device=device)
    spin = torch.zeros([len(atoms)], dtype=torch.long, device=device)
    fixed = torch.zeros([len(positions)], dtype=torch.long, device=device)
    tags = torch.zeros([len(positions)], dtype=torch.long, device=device)

    for snapshot in atoms:
        snap_pos = snapshot.get_positions().tolist()
        for a in snap_pos:
            positions.append(a)
        
        for an in snapshot.get_atomic_numbers():
            numbers.append(an)
        
        snap_cell = snapshot.get_cell().tolist()
        cells.append(snap_cell)

        if snapshot.pbc is False:
            snap_pbc = [False, False, False]
        elif snapshot.pbc is True:
            snap_pbc = [True, True, True]
        else:
            snap_pbc = [bool(i) for i in snapshot.pbc]
        
        pbcs.append(snap_pbc)

    pos = torch.tensor(positions, dtype=torch.float, device=device)
    atomic_numbers = torch.tensor(numbers, dtype=torch.long, device=device)
    cell = torch.tensor(cells, dtype=torch.float, device=device)
    pbc = torch.tensor(pbcs, dtype=torch.bool, device=device)

    fill_data = AtomicData(
        pos=pos,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        natoms=natoms,
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        nedges=nedges,
        charge=charge,
        spin=spin,
        fixed=fixed,
        tags=tags,
        batch=batch
    )

    graph_info = generate_graph(
        fill_data,
        cutoff=config.max_radius,
        max_neighbors=config.max_neighbors,
        enforce_max_neighbors_strictly=config.enforce_max_neighbors_strict,
        radius_pbc_version=2,  # TODO: test/benchmark this
        pbc=pbc[0, :]
    )

    # edge distance expansion
    expansion_func = {
        "gaussian": GaussianSmearing,
        "sigmoid": SigmoidSmearing,
        "linear_sigmoid": LinearSigmoidSmearing,
        "silu": SiLUSmearing,
    }[config.distance_function]

    edge_distance_expansion_func = expansion_func(
        0.0,
        config.max_radius,
        config.edge_distance_expansion_size,
        basis_width_scalar=2.0,
    ).to(device)

    # sort edge index according to receiver node
    edge_index, edge_attr = torch_geometric.utils.sort_edge_index(
        graph_info["edge_index"],
        [graph_info["edge_distance"], graph_info["edge_distance_vec"],
        sort_by_row=False,
    )
    edge_distance, edge_distance_vec = edge_attr[0], edge_attr[1]

    # edge directions (for direct force prediction, ref: gemnet)
    edge_direction = -edge_distance_vec / edge_distance[:, None]

    # edge distance expansion (ref: scn)
    edge_distance_expansion = edge_distance_expansion_func(edge_distance)

    # node direction expansion
    node_direction_expansion = get_node_direction_expansion(
        distance_vec=edge_distance_vec,
        edge_index=edge_index,
        lmax=config.node_direction_expansion_size - 1,
        num_nodes=fill_data.num_nodes,
    )

    # convert to neighbor list
    neighbor_list, neighbor_mask, index_mapping = convert_neighbor_list(
        edge_index, config.max_neighbors, fill_data.num_nodes
    )

    # map neighbor list
    map_neighbor_list_ = partial(
        map_neighbor_list,
        index_mapping=index_mapping,
        max_neighbors=config.max_neighbors,
        num_nodes=fill_data.num_nodes,
    )
    edge_direction = map_neighbor_list_(edge_direction)
    edge_distance_expansion = map_neighbor_list_(edge_distance_expansion)

    node_batch = fill_data.batch

    # get attention mask
    attn_mask, angle_embedding = get_attn_mask(
        edge_direction=edge_direction,
        neighbor_mask=neighbor_mask,
        num_heads=config.atten_num_heads,
        use_angle_embedding=config.use_angle_embedding,
    )

    if config.atten_name in ["memory_efficient", "flash", "math"]:
        torch.backends.cuda.enable_flash_sdp(config.atten_name == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(
            config.atten_name == "memory_efficient"
        )
        torch.backends.cuda.enable_math_sdp(config.atten_name == "math")
    else:
        raise NotImplementedError(
            f"Attention name {config.atten_name} not implemented"
        )
    
    return ChemicalGraphData(
        atomic_numbers=atomic_numbers,
        edge_distance_expansion=edge_distance_expansion,
        edge_direction=edge_direction,
        node_direction_expansion=node_direction_expansion,
        attn_mask=attn_mask,
        angle_embedding=angle_embedding,
        neighbor_list=neighbor_list,
        neighbor_mask=neighbor_mask,
        node_batch=node_batch
    )


def data_from_pmg_structures(
    structs: List[Structure],
    config: UmpireConfig,
    device: str | torch.device | int,
    batch: torch.Tensor | None = None
) -> ChemicalGraphData:
    """
    Construct a ChemicalGraphData object from a collection of pymatgen Structure objects

    Args:
        atoms (List[Structure]): Collection of snapshots to be compiled into a dataset
        config (UmpireConfig): Config object
        device (str | torch.device | int): Either a device object or a reference to a torch-recognized device
        batch (torch.Tensor | None): Either information regarding which samples will be included in which batch, or
            None (default)

    Returns:
        ChemicalGraphData

    """
    
    ad = AseAtomsAdaptor()

    atoms = [ad.get_atoms(struct) for struct in structs]

    return data_from_ase_atoms(atoms, config, device, batch=batch)


def data_from_trajectory_file(
    traj: str | Path,
    config: UmpireConfig,
    device: str | torch.device | int,
    batch: torch.Tensor | None = None
) -> ChemicalGraphData:
    """
    Construct a ChemicalGraphData object from a single molecular dynamics trajectory file

    Args:
        traj (str | Path): Either a path to a trajectory file
        config (UmpireConfig): Config object
        device (str | torch.device | int): Either a device object or a reference to a torch-recognized device
        batch (torch.Tensor | None): Either information regarding which samples will be included in which batch, or
            None (default)

    Returns:
        data (ChemicalGraphData)

    """

    atoms = Trajectory(str(traj))
    return data_from_ase_atoms(atoms, config, device, batch=batch)
