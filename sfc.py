import numpy as np
import random
import torch
import networkx as nx
from environment import VNF_SIZE, VNF_LATENCY, VNF_BANDWIDTH, SFC_RELIABILITY, Environment

class SFCBatchGenerator:
    def __init__(self, cfg):
        self.graph = nx.read_graphml('graph/' + cfg.graph + '.graphml')  # node id in graphml is saved as str type
        self.num_nodes = self.graph.number_of_nodes()
        self.batch_size = cfg.batch_size    # each episode contains batch size data
        self.min_sfc_length = cfg.min_sfc_length
        self.max_sfc_length = cfg.max_sfc_length
        self.num_vnf_types = cfg.num_vnf_types

        self.sfc_length = np.zeros(self.batch_size, dtype='int32')
        self.sfc = np.zeros((self.batch_size, self.max_sfc_length), dtype='int32')
        self.node_list = list(range(self.num_nodes)) # sample source and dest node
        self.source_dest_node_pairs = np.zeros((self.batch_size, 2), dtype='int32')
        self.reliability_requirements = np.zeros((self.batch_size, 1), dtype='float32')
        self.mask = np.zeros((self.batch_size, self.max_sfc_length), dtype='int32')
        self.sfc_batch_list = []

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.sfc_length = np.zeros(batch_size, dtype='int32')
        self.sfc = np.zeros((batch_size, self.max_sfc_length), dtype='int32')
        self.source_dest_node_pairs = np.zeros((batch_size, 2), dtype='int32')
        self.reliability_requirements = np.zeros((self.batch_size, 1), dtype='float32')
        self.mask = np.zeros((batch_size, self.max_sfc_length), dtype='int32')

    # generate vnf types in sfc
    def _generate_vnf_types(self):
        for batch in range(self.batch_size):
            self.sfc_length[batch] = random.randint(self.min_sfc_length, self.max_sfc_length)
            for i in range(self.sfc_length[batch]):
                vnf_type = random.randint(1, self.num_vnf_types)    # vnf type = 0 means empty
                self.sfc[batch][i] = vnf_type
                self.mask[batch][i] = 1

    # generate source and dest nodes in sfc
    def _generate_source_dest_nodes(self):
        for batch in range(self.batch_size):
            source_node, dest_node = random.sample(self.node_list, 2)
            self.source_dest_node_pairs[batch] = [source_node, dest_node]

    # generate reliability requirements in sfc
    def _generate_reliability_requirements(self):
        for batch in range(self.batch_size):
            reliability = random.choice(SFC_RELIABILITY)
            self.reliability_requirements[batch] = reliability

    # set a new batch of sfc
    def get_sfc_batch(self):
        self.sfc.fill(0)
        self.sfc_length.fill(0)
        self.source_dest_node_pairs.fill(0)
        self.mask.fill(0)
        self.sfc_batch_list.clear()

        self._generate_vnf_types()
        self._generate_source_dest_nodes()
        self._generate_reliability_requirements()

        # return sfc batch list with mask(dynamic length)
        for batch in range(self.batch_size):
            mask = self.mask[batch] == 1    # convert to bool list
            sfc = self.sfc[batch]
            sfc_masked = sfc[mask] - 1  # change vnf type to start from 0
            self.sfc_batch_list.append(sfc_masked.tolist())
        return self.sfc_batch_list

    # show all the sfc in current batch
    def show_sfc_batch(self):
        print('sfc batch:')
        for sfc in self.sfc_batch_list:
            print(sfc)

    # convert sfc batch list to sfc state batch list [batch_size, len=sfc_length, dim=3(size, latency, bandwidth)]
    def _convert_sfc_to_tensor(self):
        sfc_states = []
        for i in range(self.batch_size):
            sfc = self.sfc[i]
            sfc_list = [self.vnf_type_to_qos(vnf) for vnf in sfc]
            sfc_states.append(sfc_list)
        return torch.tensor(sfc_states, dtype=torch.float32)

    # get sfc states: sfc state batch list, source dest node pair, sfc reliability requirement
    def get_sfc_states(self):
        sfc_states = self._convert_sfc_to_tensor()
        source_dest_node_pairs = torch.tensor(self.source_dest_node_pairs, dtype=torch.float32)
        reliability_requirements = torch.tensor(self.reliability_requirements, dtype=torch.float32)
        return sfc_states, source_dest_node_pairs, reliability_requirements

    # generate qos vector according to vnf type
    @staticmethod
    def vnf_type_to_qos(vnf):
        vnf = vnf - 1   # vnf type: 1 - 8 -> 0 - 7, 0 means empty
        if vnf >= 0:
            return [VNF_SIZE[vnf], VNF_LATENCY[vnf], VNF_BANDWIDTH[vnf]]
        else:
            return [0, 0, 0]
