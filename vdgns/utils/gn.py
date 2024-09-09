import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import time

try:
    from builder import build_mlp
    from scatter import scatter
except:
    from utils.builder import build_mlp
    from utils.scatter import scatter


class GraphEdgeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, out_dim: int, activation='relu'):
        """
        Initialize a GraphEdgeEncoder module.

        Args:
            input_dim (int): Dimensionality of the input edge features.
            hidden_layers (list): A list of integers representing the sizes of hidden layers in the MLP.
            out_dim (int): Dimensionality of the output edge features.
            activation (str, optional): Activation function for the MLP. Default is 'relu'.

        """
        super().__init__()
        self.encoder = build_mlp(input_dim, hidden_layers, out_dim, activation)

    def forward(self, graph):
        """
        Forward pass for the GraphEdgeEncoder.

        Args:
            graph (Data): A PyTorch Geometric Data object representing the graph.

        Returns:
            torch.Tensor: The encoded edge features of the input graph.

        """
        return self.encoder(graph.edge_attr)


class GraphNodeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, out_dim: int, activation='relu'):
        """
        Initialize a GraphNodeEncoder module.

        Args:
            input_dim (int): Dimensionality of the input node features.
            hidden_layers (list): A list of integers representing the sizes of hidden layers in the MLP.
            out_dim (int): Dimensionality of the output node features.
            activation (str, optional): Activation function for the MLP. Default is 'relu'.

        """
        super().__init__()
        self.encoder = build_mlp(input_dim, hidden_layers, out_dim, activation)

    def forward(self, graph):
        """
        Forward pass for the GraphNodeEncoder.

        Args:
            graph (Data): A PyTorch Geometric Data object representing the graph.

        Returns:
            torch.Tensor: The encoded node features of the input graph.

        """
        return self.encoder(graph.x)


class GraphEncoder(nn.Module):
    def __init__(self, node_encoder: GraphNodeEncoder, edge_encoder: GraphEdgeEncoder):
        """
        Initialize a GraphEncoder module that combines node and edge encoders.

        Args:
            node_encoder (GraphNodeEncoder): A GraphNodeEncoder instance for encoding node features.
            edge_encoder (GraphEdgeEncoder): A GraphEdgeEncoder instance for encoding edge features.

        """
        super().__init__()
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder

    def forward(self, graph):
        """
        Forward pass for the GraphEncoder.

        Args:
            graph (Data): A PyTorch Geometric Data object representing the graph.

        Returns:
            tuple: A tuple containing the encoded node features and encoded edge features of the input graph.

        """
        return self.node_encoder(graph), self.edge_encoder(graph)


class GraphNetworkBlock(gnn.MessagePassing):
    """
    Graph Network Block for message-passing on a graph.

    Args:
        phi_e (nn.Module): Edge message function.
        phi_v (nn.Module): Node update function.
        aggregation_fun (str): Aggregation function for message aggregation. Supported values: 'sum', 'max', 'min', 'mean'.

    Attributes:
        phi_e (nn.Module): Edge message function.
        phi_v (nn.Module): Node update function.
        aggregation_fun (str): Aggregation function for message aggregation.

    """

    def __init__(self, phi_e: nn.Module, phi_v: nn.Module, aggregation_fun='sum'):
        super().__init__()

        self.phi_e = phi_e
        self.phi_v = phi_v

        self.aggregation_fun = aggregation_fun
        assert aggregation_fun in ['sum', 'max', 'min', 'mean']

    def message(self, x_i, x_j, edge_features):
        """
        Compute and return the message passed from source nodes to target nodes.

        Args:
            x_i (Tensor): Feature representations of source nodes.
            x_j (Tensor): Feature representations of target nodes.
            edge_features (Tensor): Features associated with edges.

        Returns:
            msg (Tensor): Computed message.
        """
        msg = torch.cat((x_i, x_j, edge_features), dim=-1)
        msg = self.phi_e(msg)

        return msg

    def aggregate(self, inputs, index):
        """
        Aggregate edge attributes incoming into a node

        Args:
            inputs (Tensor): Input data for aggregation.
            index (LongTensor): Index information for nodes.

        Returns:
            inputs (Tensor): Unchanged input data.
            out (Tensor): Aggregated output after applying the aggregation function.

        """
        out = scatter(
            inputs, index, dim=0, reduce=self.aggregation_fun)
        return inputs, out

    def forward(self, graph):
        """
        Forward pass through the Graph Network Block.

        Args:
            graph (torch_geometric.data.Data): Input graph.

        Returns:
            e_i_dash (Tensor): New edge attributes.
            v_i_dash (Tensor): New node attributes.

        """
        x = graph.x
        edge_index = graph.edge_index
        edge_features = graph.edge_attr


        e_i_dash, e_aggr = self.propagate(edge_index, x=(
            x, x), edge_features=edge_features)
        v_input = torch.cat((x, e_aggr), dim=-1)
        v_i_dash = self.phi_v(v_input)

        return e_i_dash, v_i_dash


class GraphProcessor(nn.Module):
    """
    Graph Processor for processing graph data using Graph Network Blocks.

    Args:
        gn_blocks (GraphNetworkBlock): Graph Network Block to be applied.
        num_gn_layers (int): Number of times to apply the Graph Network Block.

    """

    def __init__(self, gn_blocks):
        super().__init__()

        self.gn_blocks = gn_blocks
        self.num_gn_layers = len(gn_blocks)

    def forward(self, graph):
        """
        Forward pass through the Graph Processor.

        Args:
            graph (torch_geometric.data.Data): Input graph data.

        """
        input_latent_x = graph.x.clone()
        input_latent_edges = graph.edge_attr.clone()

        for i in range(self.num_gn_layers):
            out_edges, out_nodes = self.gn_blocks[i](graph)
            graph.edge_attr = out_edges
            graph.x = out_nodes

        graph.x += input_latent_x
        graph.edge_attr += input_latent_edges

        graph.x = torch.tanh(graph.x)
        graph.edge_attr = torch.tanh(graph.x)

        return graph


class GraphEncodeProcessDecode(nn.Module):
    """
    Graph-based Encoder-Processor-Decoder model for structured data.

    This class represents a complete model for processing graph-structured data.
    The model follows the Encoder-Processor-Decoder (EPD) architecture.

    Args:
        encoder (GraphEncoder): Graph encoder responsible for encoding input graph data.
        processor (GraphProcessor): Graph processor that applies message-passing iterations.
        decoder (GraphNodeEncoder): Graph node decoder for generating the final output.

    """

    def __init__(self, encoder: GraphEncoder, processor: GraphProcessor, decoder: GraphNodeEncoder, preprocessing_data: dict, postprocessing_data: dict):
        super().__init__()

        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

        self.node_mean = preprocessing_data['node_mean']
        self.node_std = preprocessing_data['node_std']
        self.edge_mean = preprocessing_data['edge_mean']
        self.edge_std = preprocessing_data['edge_std']
        
        self.out_mean = postprocessing_data['out_mean']
        self.out_std = postprocessing_data['out_std']

    def preprocess(self, g):
        g.x[:,:self.node_mean.shape[0]] -= self.node_mean 
        g.x[:,:self.node_std.shape[0]] /= self.node_std 

        g.edge_attr[:,:self.edge_mean.shape[0]] -= self.edge_mean
        g.edge_attr[:,:self.edge_std.shape[0]] /= self.edge_std
        
        return g
        
    def postprocess(self, g):
        g.x *= self.out_std
        g.x += self.out_mean

        return g

    def forward(self, g):
        """
        Forward pass through the Graph Encode-Process-Decode model.

        Args:
            g (Data.Data): Input graph data in PyTorch Geometric format.

        Returns:
            g (Data.Data): Processed graph data after encoding, processing, and decoding.

        The forward method applies the complete EPD model to the input graph data. It consists of the following steps:
        1. Encoding: The input graph's node and edge attributes are encoded using the specified `encoder`. This step typically involves feature extraction and initial representations.
        2. Processing: The encoded graph is passed through the `processor`, which may apply message-passing iterations, transforming the node and edge features iteratively.
        3. Decoding: The processed graph is decoded using the `decoder`, which generates the final output, typically node-level predictions.

        Example:
        ```
        encoder = GraphEncoder(...)
        processor = GraphProcessor(...)
        decoder = GraphNodeEncoder(...)
        model = GraphEncodeProcessDecode(encoder, processor, decoder)
        output_graph = model(input_graph)
        ```

        """
        g = self.preprocess(g)

        g.x, g.edge_attr = self.encoder(g)
        g = self.processor(g)
        g.x = self.decoder(g)

        g = self.postprocess(g)

        return g

def build_encoder_processor_decoder(
    node_size: int,
    edge_size: int,
    node_latent: int,
    edge_latent: int,
    output_size: int,
    num_gn_layers: int = 5,
    shared_gn_layers: bool = True,
    processor_depth: int = 1,
    aggregation_fun: str = 'sum',
    preprocessing_data: dict = {'node_mean': 0.0, 'node_std': 1.0, 'edge_mean': 0.0, 'edge_std': 1.0},
    postprocessing_data: dict = {'out_mean': 0.0, 'out_std': 1.0},
    ) -> GraphEncodeProcessDecode:
    """
    Build a GraphEncodeProcessDecode model with specified configurations.

    Args:
        node_size (int): Size of node features.
        edge_size (int): Size of edge features.
        node_latent (int): Size of the latent space for node features.
        edge_latent (int): Size of the latent space for edge features.
        output_size (int): Size of the output node features.
        num_gn_layers (int, optional): Number of graph network layers in the processor. Default is 5.
        aggregation_fun (str, optional): Aggregation function for graph network layers. Default is 'sum'.
        preprocess_fun (Callable[[Data], Data], optional): A function to preprocess the input graph data before building the model.
        postprocess_fun (Callable[[Data], Data], optional): A function to postprocess the output graph data after building the model.

    Returns:
        GraphEncodeProcessDecode: Configured GraphEncodeProcessDecode model.

    """
    
    # Graph Encoder
    node_encoder = GraphNodeEncoder(node_size, [node_latent], node_latent)
    edge_encoder = GraphEdgeEncoder(edge_size, [edge_latent], edge_latent)
    graph_encoder = GraphEncoder(node_encoder, edge_encoder)

   
    blocks = []
    if shared_gn_layers:
        gn_block = GraphNetworkBlock(
            phi_e=build_mlp(edge_latent + 2 * node_latent,
                            [edge_latent] * processor_depth, edge_latent, 'tanh'),
            phi_v=build_mlp(node_latent + edge_latent,
                            [node_latent] * processor_depth, node_latent, 'tanh'),
            aggregation_fun=aggregation_fun,
        )
        for _ in range(num_gn_layers):
            blocks.append(gn_block)
        
    else:
        for _ in range(num_gn_layers):
            blocks.append(
                GraphNetworkBlock(
                    phi_e=build_mlp(edge_latent + 2 * node_latent,
                                    [edge_latent] * processor_depth, edge_latent, 'tanh'),
                    phi_v=build_mlp(node_latent + edge_latent,
                                    [node_latent] * processor_depth, node_latent, 'tanh'),
                    aggregation_fun=aggregation_fun
                )
            )
            
    graph_processor = GraphProcessor(
        gn_blocks=blocks
    )


    # Graph Decoder
    graph_decoder = GraphNodeEncoder(node_latent, [node_latent], output_size)

    model = GraphEncodeProcessDecode(
        preprocessing_data=preprocessing_data,
        encoder=graph_encoder,
        processor=graph_processor,
        decoder=graph_decoder,
        postprocessing_data=postprocessing_data
    )

    return model


def get_edge_attributes(edges: torch.Tensor, particles: torch.Tensor):
    """
    Compute edge attributes based on particle positions.
    This function calculates edge attributes as magnitude and direction based on the positions of particles connected by edges.

    Args:
        edges (torch.Tensor): Edge indices representing connections between particles.
        particles (torch.Tensor): Particle positions.

    Returns:
        torch.Tensor: Computed edge attributes, including magnitude, and direction.

    """
    
    edge_attributes = torch.zeros((edges.shape[1], 3), dtype=torch.float)

    for j, edge in enumerate(edges.t()):
        sender_particle = particles[edge[0]][:2]
        receiver_particle = particles[edge[1]][:2]

        magnitude = torch.norm(receiver_particle - sender_particle)
        direction = receiver_particle - sender_particle

        edge_attributes[j][0] = magnitude
        edge_attributes[j][1] = direction[0]
        edge_attributes[j][2] = direction[1]

    return edge_attributes

def get_connectivity(particles, system='spring', limit_edges_per_particle=150000000, radius=0.05):
    """
    Generate connectivity for a graph system.
    This function generates the connectivity sparse matrix for a graph system, where particles are connected based on the specified system.
    In the case of a 'spring' system, the particles are connected to their neighbours in the particles tensor, forming a string.

    Args:
        n_particles (int): Number of particles in the system.
        system (str, optional): Type of system. Default is 'spring'.

    Returns:
        torch.Tensor: Graph connectivity sparse matrix.
    """
    
    n_particles = particles.shape[0]
    if system == 'spring':
        indices = torch.linspace(0,n_particles - 1,n_particles,dtype=torch.int64)
        sender_nodes = torch.cat((indices, torch.flip(indices, dims=[0])[1:-1]))
        receiver_nodes = torch.cat((indices[1:-1], torch.flip(indices, dims=[0])))

        graph_connectivity = torch.cat((sender_nodes.view(-1, 1), receiver_nodes.view(-1, 1)), dim=0).view(2,-1)

        return graph_connectivity
    elif system == 'fluid':
        neigh = NearestNeighbors(radius=radius)
        neigh.fit(particles)
        _, closest = neigh.radius_neighbors(particles, return_distance=True, sort_results=True)

        total_edges = 0
        for destinations in closest:
            total_edges += min(limit_edges_per_particle, len(destinations))

        graph_connectivity = torch.zeros((2, total_edges),dtype=torch.int64)
        edge_index = 0

        particle_indices = range(n_particles)
        for source_index, destinations in zip(particle_indices, closest):
            for dst_index in destinations[:limit_edges_per_particle]:
                graph_connectivity[0][edge_index] = source_index
                graph_connectivity[1][edge_index] = dst_index
                edge_index += 1

        return graph_connectivity

"""
Dataset for the VDGNS experiments.
"""
class ParticleVideoDataset(Dataset):
    def __init__(self, number_of_classes, raw_particles, videos, video_encoder, system='spring', connectivity_radius=0.05, minimum_rollout_step=0, limit_edges_per_particle=20, edge_noise_std=0.01, particle_noise_std=0.001, one_hot_encode_class=False, **kwargs):
        super(ParticleVideoDataset, self).__init__()

        self.system = system
        self.one_hot_encode_class = one_hot_encode_class
        self.edge_noise_std = edge_noise_std
        self.particle_noise_std = particle_noise_std
        self.minimum_rollout_step = minimum_rollout_step
        self.connectivity_radius = connectivity_radius
        self.limit_edges_per_particle = limit_edges_per_particle
        self.graphs = self.convert_to_graphs(raw_particles, **kwargs)
        self.number_of_classes = number_of_classes
        self.video_encoder = video_encoder

        self.reset_videos(videos)
        self.node_std, self.node_mean, self.edge_std, self.edge_mean, self.out_std, self.out_mean = self.get_std()

    def len(self):
        return len(self.graphs) * len(self.graphs[0])

    def reset_videos(self, videos):
        assert len(videos) == len(self.graphs) and "Videos do not correspond to particles"

        self.videos = self.group_by_class(videos, self.number_of_classes)
        self.class_cardinality = videos.shape[0] // self.number_of_classes

    def get(self, idx):
        sequence_idx = idx // len(self.graphs[0])
        step_idx = idx % len(self.graphs[0])

        while step_idx < self.minimum_rollout_step:
            idx = torch.randint(0, len(self.graphs) * len(self.graphs[0]), (1,))[0]
            sequence_idx = idx // len(self.graphs[0])
            step_idx = idx % len(self.graphs[0])

        graph = self.graphs[sequence_idx][step_idx]
        
        class_index = sequence_idx // self.class_cardinality
        video_index = torch.randint(low=0, high=self.class_cardinality, size=(1, ))[0]
        video = self.videos[class_index][video_index]
        
        if self.one_hot_encode_class:
            encoding = torch.zeros((self.number_of_classes,), dtype=torch.float32)
            encoding[class_index] = 1
        else:
            encoding = self.video_encoder(video.view(1, *video.shape))[:,-1]
        expanded_encoding = encoding.repeat(graph.x.shape[0], 1)
        
        out_graph = graph.clone()
        out_graph.x += torch.randn_like(out_graph.x) * self.particle_noise_std
        out_graph.edge_attr += torch.randn_like(out_graph.edge_attr) * self.edge_noise_std

        out_graph.x = torch.cat([out_graph.x, expanded_encoding], dim=1)

        return out_graph
    
    def group_by_class(self, videos, number_of_classes):
        class_cardinality = videos.shape[0] // number_of_classes
        videos_per_class = [videos[i * class_cardinality : (i + 1) * class_cardinality] for i in range(number_of_classes)]

        return videos_per_class

    def preprocess(self, graph):  
        graph.x[:,:self.node_mean.shape[0]] -= self.node_mean 
        graph.x[:,:self.node_std.shape[0]] /= self.node_std 

        graph.edge_attr[:,:self.edge_mean.shape[0]] -= self.edge_mean
        graph.edge_attr[:,:self.edge_std.shape[0]] /= self.edge_std
          
        return graph

    def postprocess(self, graph):
        graph.x *= self.out_std
        graph.x += self.out_mean
        
        return graph

    def get_std(self):
        number_of_particles = 0
        number_of_edges = 0
        
        for g in [graph for sequence in self.graphs for graph in sequence]:
            number_of_particles += g.x.shape[0]
            number_of_edges += g.edge_attr.shape[0]

        node_all = torch.zeros((number_of_particles, self.graphs[0][0].x.shape[1]), dtype=torch.float32)
        out_all = torch.zeros((number_of_particles, 2), dtype=torch.float32)
        edge_all = torch.zeros((number_of_edges, self.graphs[0][0].edge_attr.shape[1]), dtype=torch.float32)

        node_idx = 0
        edge_idx = 0

        for g in [graph for sequence in self.graphs for graph in sequence]:
            x = g.x 
            edge_attr = g.edge_attr
            y = g.y 

            node_all[node_idx:node_idx + x.shape[0]] = x
            out_all[node_idx:node_idx + y.shape[0]] = y
            edge_all[edge_idx:edge_idx + edge_attr.shape[0]] = edge_attr

            node_idx += x.shape[0]
            edge_idx += edge_attr.shape[0]

        node_std = torch.std(node_all, dim=0)
        node_std[node_std == 0] = 1

        edge_std = torch.std(edge_all, dim=0)
        out_std = torch.std(out_all, dim=0)

        node_mean = torch.mean(node_all, dim=0)
        edge_mean = torch.mean(edge_all, dim=0)
        out_mean = torch.mean(out_all, dim=0)

        return node_std, node_mean, edge_std, edge_mean, out_std, out_mean

    def convert_to_graphs(self, particles, past_velocities=3):
        graphs = []
        
        # We iterate through each rollout
        for i, rollout in tqdm(enumerate(particles)):
            graphs.append([])
            true_rollout = rollout.clone()
            rollout = rollout.clone()

            # Set the initial velocities for every particle
            velocity = torch.zeros((rollout.shape[1], 2), dtype=torch.float32)
            for j, particle in enumerate(rollout[0]):
                velocity[j][0] = rollout[1][j][0] - rollout[0][j][0]
                velocity[j][1] = rollout[1][j][1] - rollout[0][j][1]

            # We iterate through each timestep of the rollout
            # We use finite difference to calculate accelrations, so we need to drop the border records
            for j in range(1, rollout.shape[0] - 1): 
                # Defining an alias for the current step of the rollout 
                step = rollout[j]           
                
                # Get the connectivity of the graph. In case of a spring, it simply links neighbouring particles
                graph_edges = get_connectivity(particles=step, system=self.system, radius=self.connectivity_radius, limit_edges_per_particle=self.limit_edges_per_particle)
                
                # Store the direction an magnitude of an edge as edge attribute
                edge_attributes = get_edge_attributes(graph_edges, step)
                
                # Defining the graph node attributes
                number_of_particles = step.shape[0]

                if self.system == 'spring':
                    graph_x = torch.zeros((number_of_particles, past_velocities * 2), dtype=torch.float32)
                else:
                    graph_x = torch.zeros((number_of_particles, past_velocities * 2 + 4), dtype=torch.float32)

                # For each particle in the current timestep
                for k, particle in enumerate(step):
                    # We store this node's attributes: [vx_t-1, vy_t-1, vx_t-2, vy_t-2, ...]
                    particle_feature = torch.zeros((past_velocities * 2), dtype=torch.float32)

                    # We iterate through past positions to get the past velocities
                    for past in range(1, past_velocities + 1):
                        rollout_idx = j - past 

                        if rollout_idx >= 0:
                            # Get the finite difference between positions
                            past_vx = rollout[rollout_idx + 1][k][0] - rollout[rollout_idx][k][0]
                            past_vy = rollout[rollout_idx + 1][k][1] - rollout[rollout_idx][k][1]
                        else:
                            # We pad the rollout with 0 velocities at the beginning of the rollout
                            past_vx = 0
                            past_vy = 0

                        particle_feature[2 * (past - 1)] = past_vx
                        particle_feature[2 * (past - 1) + 1] = past_vy
                    
                    graph_x[k, :particle_feature.shape[0]] = particle_feature

                    if self.system == 'fluid':
                        dist_to_top = min(self.connectivity_radius, 1 - rollout[j][k][1])
                        dist_to_left = min(self.connectivity_radius, rollout[j][k][0])
                        dist_to_right = min(self.connectivity_radius, 1 - rollout[j][k][0])
                        dist_to_bottom = min(self.connectivity_radius, rollout[j][k][1])

                        graph_x[k, -1] = dist_to_top
                        graph_x[k, -2] = dist_to_left
                        graph_x[k, -3] = dist_to_right
                        graph_x[k, -4] = dist_to_bottom


                # Target accelerations we will predict
                graph_y = torch.zeros((number_of_particles, 2), dtype=torch.float32)

                # Calculate the accelerations at each timestep using finite difference
                for k in range(number_of_particles):
                    graph_y[k][0] = true_rollout[j + 1][k][0] - 2 * true_rollout[j][k][0] + true_rollout[j - 1][k][0]
                    graph_y[k][1] = true_rollout[j + 1][k][1] - 2 * true_rollout[j][k][1] + true_rollout[j - 1][k][1]

                # Add the constructed graph to the dataset
                graphs[-1].append(Data(x=graph_x, edge_index=graph_edges, edge_attr=edge_attributes, y=graph_y))    

        return graphs

class ParticleVideoBaselineDataset(ParticleVideoDataset):
    def __init__(self, **kwargs):
        super(ParticleVideoBaselineDataset, self).__init__(**kwargs)
        
    def get(self, idx):
        sequence_idx = idx // len(self.graphs[0])
        step_idx = idx % len(self.graphs[0])
        
        graph = self.graphs[sequence_idx][step_idx]
        
        class_index = sequence_idx // self.class_cardinality
        video_index = torch.randint(low=0, high=self.class_cardinality, size=(1, ))[0]
        
        encoding = torch.linspace(-1, 1, self.number_of_classes)[class_index]
        expanded_encoding = encoding.repeat(graph.x.shape[0], 1)
        
        out_graph = graph.clone()
        out_graph.x = torch.cat([out_graph.x, expanded_encoding], dim=1)

        return out_graph
    