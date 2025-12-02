import torch

# unpack state as input
def unpack_state(state):
    net_state, sfc_state, source_dest_node_pair, reliability_requirement = zip(*state)

    if len(sfc_state) > 1:
        sfc_state = torch.stack(sfc_state, dim=0)
        source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)
        reliability_requirement = torch.stack(reliability_requirement, dim=0).unsqueeze(2)
    else:
        sfc_state = sfc_state[0].unsqueeze(0)
        source_dest_node_pair = source_dest_node_pair[0].unsqueeze(0)
        reliability_requirement = reliability_requirement[0].unsqueeze(0)

    return net_state, sfc_state, source_dest_node_pair, reliability_requirement

# reshape state into batch view
def reshape_state_batch(sfc_state, source_dest_node_pair, reliability_requirement, vnf_state_dim, max_sfc_length):
    batch_size = sfc_state.shape[0]
    sfc_state = sfc_state.view(batch_size, max_sfc_length, vnf_state_dim)
    source_dest_node_pair = source_dest_node_pair.view(batch_size, 2)
    reliability_requirement = reliability_requirement.view(batch_size, 1, 1)

    return sfc_state, source_dest_node_pair, reliability_requirement
