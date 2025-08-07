import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_undirected
import config

VNF_SIZE = [4, 3, 2, 2, 1, 1, 1, 1] # node resource of a VNF need
VNF_LATENCY = [20, 20, 40, 40, 80, 80, 100, 100]    # VNF processing delay
VNF_BANDWIDTH = [100, 80, 60, 60, 60, 60, 40, 20]   # VNF bandwidth requirement

class Environment:
    def __init__(self, graph):
        # no update after initialization
        self.graph = graph  # node id in graphml is saved as str type
        self.num_nodes = graph.number_of_nodes()
        self.num_links = graph.number_of_edges()
        self.num_vnf_types = config.NUM_VNF_TYPES

        self.node_properties = [{'capacity': random.randint(10, 20)} for _ in range(self.num_nodes)]

        self.links = list(graph.edges()) # match graph links to indexes
        self.link_index = self.link_to_index(self.links)  # {('node a', 'node b'): index}
        self.link_properties = [{'bandwidth': random.randint(5, 10) * 20, 'latency': random.randint(5, 20)} for _ in
                                range(self.num_links)]

        self.vnf_properties = [{'size': VNF_SIZE[i], 'latency': VNF_LATENCY[i], 'bandwidth': VNF_BANDWIDTH[i]} for i in
                               range(config.NUM_VNF_TYPES)]

        self.node_state_dim = 0
        self.vnf_state_dim = 0
        self.state_dim = 0

        self.p_min = 2    # power consumption to activate a node
        self.p_unit = 1   # power consumption per unit node resource
        self.p_link = 0.05  # power consumption per link length

        self.path_penalty = 200  # a penalty factor added to reward if there is no path found between two vnf

        # update per episode
        self.node_used = np.zeros(self.num_nodes)
        self.link_used = np.zeros(self.num_links)   # link bandwidth usage

        self.node_occupied = np.zeros(self.num_nodes)  # list to record whether a node is used to place a VNF

        # update per sfc
        self.exceeded_capacity = 0
        self.exceeded_latency = 0
        self.exceeded_bandwidth = 0

        self.link_occupied = np.zeros(self.num_links)   # list to record whether a link is used by a sfc

        self.sfc_path = []  # a list to record sfc path
        self.sfc_latency = 0    # over all latency of a sfc
        self.vnf_placement = None   # vnf placement result list for a sfc: 1 for success and -1 for failure

        self.latency_requirement = 0    # latency requirement of a sfc
        self.bandwidth_requirement = 0  # bandwidth requirement of a sfc

        self.placement_reward = 0
        self.power_consumption = 0
        self.exceeded_penalty = 0
        self.reward = 0
        self.sfc_placed_num = 0

        self.lambda_placement = 20
        self.lambda_power = 5
        self.lambda_capacity = 0.25
        self.lambda_bandwidth = 0.01
        self.lambda_latency = 0.01

        # record episode data
        self.placement_reward_list = []
        self.power_consumption_list = []
        self.exceeded_penalty_list = []
        self.reward_list = []
        self.sfc_latency_list = []
        self.exceeded_node_capacity_list = []
        self.exceeded_link_bandwidth_list = []

    # place VNF and record exceeded node capacity
    def place_vnf(self, sfc, placement):
        for i, (vnf, node) in enumerate(zip(sfc, placement)):
            if self.vnf_properties[vnf]['size'] <= self.node_properties[node]['capacity'] - self.node_used[node]:
                self.node_used[node] += self.vnf_properties[vnf]['size']
                self.vnf_placement[i] = 1
            else:
                self.node_used[node] += self.vnf_properties[vnf]['size']
                # self.vnf_placement[i] = -1

    # find the shortest path between two nodes
    def find_route(self, source, target):
        # source and target are elements of a tensor
        source, target = str(source), str(target)   # convert to str type
        if nx.has_path(self.graph, source, target):
            return nx.shortest_path(self.graph, source, target)
        else:
            return False

    # determine sfc path
    def find_sfc_path(self, source_dest_node_pair, placement):
        source_node, dest_node = source_dest_node_pair[0], source_dest_node_pair[1]
        self.sfc_path.append(self.find_route(source_node, placement[0]))    # source node -> vnf 0
        for i in range(len(placement) - 1):
            route = self.find_route(placement[i], placement[i + 1])
            if not route:
                self.vnf_placement[i + 1:] = 0   # vnf placement failed
            self.sfc_path.append(route)
        self.sfc_path.append(self.find_route(placement[-1], dest_node)) # vnf N -> dest node

    # compute node exceeded capacity, + for reward and - for penalty
    def compute_capacity(self, placement):
        for node in placement:
            self.exceeded_capacity += self.node_used[node] - self.node_properties[node]['capacity']

    # compute the overall latency and exceeded latency of a path
    def compute_latency(self, path, sfc):
        path_length = len(path)
        for i, route in enumerate(path):
            if route:   # if path exist, compute latency
                for i in range(len(route) - 1):
                    if int(route[i]) > int(route[i+1]): # adjust nodes order to select link
                        index = self.link_index[(route[i+1], route[i])]
                    else:
                        index = self.link_index[(route[i], route[i+1])]
                    self.sfc_latency += self.link_properties[index]['latency']
            else:
                self.exceeded_latency += self.path_penalty * (path_length - i)
                continue
        for vnf in sfc:
            self.latency_requirement += self.vnf_properties[vnf]['latency']
        self.exceeded_latency += self.sfc_latency - self.latency_requirement

    # compute and update bandwidth occupancy
    def compute_bandwidth(self, sfc, path):
        self.bandwidth_requirement = max([self.vnf_properties[vnf]['bandwidth'] for vnf in sfc])
        path_length = len(path)
        for i, route in enumerate(path):
            if route:  # if path exist, update bandwidth
                for i in range(len(route) - 1):
                    if int(route[i]) > int(route[i + 1]):  # adjust nodes order to select link
                        index = self.link_index[(route[i + 1], route[i])]
                    else:
                        index = self.link_index[(route[i], route[i + 1])]
                    self.link_used[index] += self.bandwidth_requirement
                    self.link_occupied[index] = 1
            else:
                self.exceeded_bandwidth += self.path_penalty * (path_length - i)
                break
        self.exceeded_bandwidth += sum(self.link_occupied \
                                  * (self.link_used - [link['bandwidth'] for link in self.link_properties]))

    # compute sfc power consumption, update node occupancy
    def compute_power(self, sfc, placement):
        for vnf, node in zip(sfc, placement):
            if self.node_occupied[node]:
                self.power_consumption += self.p_unit * VNF_SIZE[vnf]
            else:
                self.power_consumption += self.p_min + self.p_unit * VNF_SIZE[vnf]
                self.node_occupied[node] = 1
        self.power_consumption += self.p_link * self.sfc_latency

    # compute the overall reward of a sfc
    def compute_reward(self, sfc):
        # vnf deployment reward
        for i in range(len(sfc)):
            index = sfc[i]
            # size * bandwidth / latency
            vnf_reward = self.vnf_properties[index]['size'] \
                         * (self.vnf_properties[index]['bandwidth'] / 20) \
                         / (self.vnf_properties[index]['latency'] / 20)
            self.placement_reward += self.vnf_placement[i] * vnf_reward

        if len(sfc) == sum(self.vnf_placement):
            self.placement_reward = self.placement_reward * 2
            self.sfc_placed_num += 1

        self.placement_reward = self.lambda_placement * self.placement_reward

        self.power_consumption = self.lambda_power * self.power_consumption

        self.exceeded_penalty = (self.lambda_capacity * self.exceeded_capacity
                                 + self.lambda_bandwidth * self.exceeded_bandwidth
                                 + self.lambda_latency * self.exceeded_latency)

        self.reward = self.placement_reward - self.power_consumption - self.exceeded_penalty

        self.placement_reward_list.append(self.placement_reward)
        self.power_consumption_list.append(self.power_consumption)
        self.exceeded_penalty_list.append(self.exceeded_penalty)
        self.reward_list.append(self.reward)
        self.sfc_latency_list.append(self.sfc_latency)
        self.exceeded_node_capacity_list.append(self.exceeded_capacity)
        self.exceeded_link_bandwidth_list.append(self.exceeded_bandwidth)

    @staticmethod
    def link_to_index(links):
        index_dict = {link: index for index, link in enumerate(links)}  # node a < node b
        return index_dict

    # get edge index for GAT
    def get_edge_index(self):
        G = self.graph
        edge_list = []
        for edge in G.edges:
            edge_list.append(list(map(int, edge[:2])))
        edge_index = torch.tensor(
            edge_list).t().contiguous()  # shape: 2 * num_edges, represent for the connection relationship
        edge_index = to_undirected(edge_index)
        return edge_index

    # aggregate node features and link features to generate net states: [node_state, neighbor_node_state, neighbor_link_state]
    def aggregate_features(self):
        G = self.graph
        aggregate_features = []
        for node in G.nodes():
            features = np.zeros(4)
            features[0] = self.node_properties[int(node)]['capacity'] - self.node_used[int(node)]
            num_neighbor = len(list(G.neighbors(node)))
            for neighbor in G.neighbors(node):
                if int(node) > int(neighbor):  # adjust nodes order to select link
                    index = self.link_index[(neighbor, node)]
                else:
                    index = self.link_index[(node, neighbor)]
                # add neighbor node and connected link properties to state
                features[1] += self.node_properties[int(neighbor)]['capacity'] - self.node_used[int(neighbor)]
                features[2] += self.link_properties[index]['bandwidth'] - self.link_used[index]
                features[3] += self.link_properties[index]['latency']
            features[1:] /= num_neighbor
            aggregate_features.append(features.tolist())
        aggregate_features = torch.tensor(aggregate_features, dtype=torch.float32)
        return aggregate_features

    def get_state_dim(self, sfc_generator):
        node_features = self.aggregate_features()
        self.node_state_dim = node_features.shape[1]

        sfc_generator.get_sfc_batch()
        sfc_states = sfc_generator.get_sfc_states()
        self.vnf_state_dim = sfc_states.shape[2]

        self.state_dim = (self.num_nodes + config.MAX_SFC_LENGTH) * self.vnf_state_dim

    def step(self, sfc, placement):
        source_dest_node_pair = sfc[:2]  # sfc[0:1] is the source dest node pair
        sfc = sfc[2:]   # sfc[2:] is the sfc
        # print(placement)
        self.clear_sfc()

        self.vnf_placement = np.zeros(len(sfc), dtype='int32')
        self.place_vnf(sfc, placement)
        self.find_sfc_path(source_dest_node_pair, placement)

        self.compute_capacity(placement)
        self.compute_latency(self.sfc_path, sfc)
        self.compute_bandwidth(sfc, self.sfc_path)
        self.compute_power(sfc, placement)
        self.compute_reward(sfc)

        next_node_states = self.aggregate_features()

        return next_node_states, self.reward / 1000 # reward scaling

    def render(self):
        net_states = self.aggregate_features()
        print('node number:', self.graph.number_of_nodes())
        print('link number:', self.graph.number_of_edges())
        print('net state dim:', net_states.shape[1])

    # clear records related to current sfc
    def clear_sfc(self):
        self.exceeded_capacity = 0
        self.exceeded_latency = 0
        self.exceeded_bandwidth = 0

        # self.node_occupied is used to calculate power and update per epoch
        self.link_occupied.fill(0)

        self.sfc_path.clear()
        self.sfc_latency = 0
        self.vnf_placement = None

        self.latency_requirement = 0
        self.bandwidth_requirement = 0

        self.placement_reward = 0
        self.power_consumption = 0
        self.exceeded_penalty = 0
        self.reward = 0

    def clear(self):
        self.node_used.fill(0)
        self.link_used.fill(0)
        self.node_occupied.fill(0)
        self.link_occupied.fill(0)

        self.exceeded_capacity = 0
        self.exceeded_latency = 0
        self.exceeded_bandwidth = 0

        self.sfc_path.clear()
        self.sfc_latency = 0
        self.vnf_placement = None
        self.latency_requirement = 0
        self.bandwidth_requirement = 0

        self.placement_reward = 0
        self.power_consumption = 0
        self.exceeded_penalty = 0
        self.reward = 0
        self.sfc_placed_num = 0

        self.placement_reward_list.clear()
        self.power_consumption_list.clear()
        self.exceeded_penalty_list.clear()
        self.reward_list.clear()
        self.sfc_latency_list.clear()
        self.exceeded_node_capacity_list.clear()
        self.exceeded_link_bandwidth_list.clear()
