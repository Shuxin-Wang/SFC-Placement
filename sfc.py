import numpy as np
import random
import torch
import config
from environment import VNF_SIZE, VNF_LATENCY, VNF_BANDWIDTH

class SFCBatchGenerator:
    def __init__(self, batch_size, min_sfc_length, max_sfc_length, num_vnf_types):
        self.batch_size = batch_size    # each episode contains batch size data
        self.min_sfc_length = min_sfc_length
        self.max_sfc_length = max_sfc_length
        self.num_vnf_types = num_vnf_types

        self.sfc_length = np.zeros(self.batch_size, dtype='int32')
        self.sfc = np.zeros((self.batch_size, config.MAX_SFC_LENGTH), dtype='int32')
        self.mask = np.zeros((self.batch_size, config.MAX_SFC_LENGTH), dtype='int32')
        self.sfc_batch_list = []

    # set a new batch of sfc
    def get_sfc_batch(self):
        self.sfc.fill(0)
        self.sfc_length.fill(0)
        self.mask.fill(0)
        self.sfc_batch_list.clear()

        # set sfc batch without mask(max sfc length)
        for batch in range(self.batch_size):
            self.sfc_length[batch] = random.randint(self.min_sfc_length, self.max_sfc_length)
            for i in range(self.sfc_length[batch]):
                vnf_type = random.randint(1, self.num_vnf_types)    # vnf type = 0 means empty
                self.sfc[batch][i] = vnf_type
                self.mask[batch][i] = 1

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
    def get_sfc_states(self):
        sfc_states = []
        for i in range(self.batch_size):
            sfc = self.sfc[i]
            sfc_list = []
            position = 1
            for vnf in sfc:
                qos = self.vnf_type_to_qos(vnf)
                qos.append(position)    # position embedding
                sfc_list.append(qos)
                position += 1
            sfc_states.append(sfc_list)
        sfc_states = torch.tensor(sfc_states, dtype=torch.float32)
        return sfc_states

    def get_sfc_masks(self):
        return self.mask

    # generate qos vector according to vnf type
    @staticmethod
    def vnf_type_to_qos(vnf):
        vnf = vnf - 1   # vnf type: 1 - 8 -> 0 - 7, 0 means empty
        if vnf >= 0:
            return [VNF_SIZE[vnf], VNF_LATENCY[vnf], VNF_BANDWIDTH[vnf]]
        else:
            return [0, 0, 0]

if __name__ == '__main__':

    random.seed(27)
    # sfc_generator = SFCBatchGenerator(config.BATCH_SIZE, config.MIN_SFC_LENGTH, config.MAX_SFC_LENGTH, config.NUM_VNF_TYPES)
    sfc_generator = SFCBatchGenerator(config.BATCH_SIZE, config.MIN_SFC_LENGTH, 10,
                                      config.NUM_VNF_TYPES)
    sfc_generator.get_sfc_batch()
    # sfc_generator.show_sfc_batch()

    # get sfc states test
    test_states = sfc_generator.get_sfc_states()
    print('test states:\n', test_states)

    # get sfc masks test
    # sfc_masks = sfc_generator.get_sfc_masks()
    # print('sfc masks:\n', sfc_masks)
