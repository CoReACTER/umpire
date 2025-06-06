from dataclasses import dataclass


# Heavily inspired by/taken from:
# Eric Qu & Aditi S. Krishnapriyan, "The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains", NeurIPS 2024
# See https://github.com/ASK-Berkeley/EScAIP
@dataclass
class UmpireConfig:
    """
    Config for UMPIRE models.

    TODO: docstring
    """

    # NN config
    num_layers: int
    hidden_size: int  # Should be divisible by 2
    batch_size: int
    atom_embedding_size: int
    node_direction_embedding_size: int
    node_direction_expansion_size: int
    edge_distance_expansion_size: int
    edge_distance_embedding_size: int
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ]
    attention: Literal[
        "math",
        "memory_efficient",
        "flash",
    ]
    max_num_nodes_per_batch: int
    atten_num_heads: int
    readout_hidden_layer_multiplier: int
    output_hidden_layer_multiplier: int
    ffn_hidden_layer_multiplier: int

    # Regularization config
    mlp_dropout: float
    atten_dropout: float
    stochastic_depth_prob: float
    normalization: Literal["layernorm", "rmsnorm", "skip"]

    # Graph config
    # on_the_fly_graph: bool = True  <-- Since we don't know connectivity, is this even helpful?
    max_neighbors: int
    max_num_elements: int
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"]  # TODO: what is this?

    # kwargs
    use_angle_embedding: bool = True
    enforce_max_neighbors_strict: bool = False  # TODO: what is this?
    max_radius: float = 5.0  # In Angstrom
    use_pbc: bool = True