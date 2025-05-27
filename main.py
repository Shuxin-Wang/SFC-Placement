import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import environment
import sfc
import config
from agent import DDPG

# cuda check
is_cuda_available = torch.cuda.is_available()
current_gpu_index = torch.cuda.current_device()
current_gpu_name = torch.cuda.get_device_name(current_gpu_index)

print('-' * 20 + 'CUDA info' + '-' * 20)
print('CUDA is available:', is_cuda_available)
print('Current GPU index:', current_gpu_index)
print('Current GPU name:', current_gpu_name)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # initialization
    G = nx.read_graphml('Cogentco.graphml')
    env = environment.Environment(G)
    sfc_generator = sfc.SFCBatchGenerator(20, config.MIN_SFC_LENGTH, config.MAX_SFC_LENGTH,
                                          config.NUM_VNF_TYPES)

    env.clear()
    env.get_state_dim(sfc_generator)

    node_state_dim = env.node_state_dim
    vnf_state_dim = env.vnf_state_dim
    state_dim = env.state_dim
    state_input_dim = node_state_dim * env.num_nodes + config.MAX_SFC_LENGTH * vnf_state_dim
    state_output_dim = (env.num_nodes + config.MAX_SFC_LENGTH) * vnf_state_dim

    agent = DDPG(node_state_dim, vnf_state_dim, state_input_dim, state_output_dim,
                      config.MAX_SFC_LENGTH, device)

    # train

    actor_loss_list = []
    critic_loss_list = []
    reward_list = []

    # plt.ion()
    plt.figure()

    tqdm.write('-' * 20 + 'Agent training start' + '-' * 20 + '\t')

    pbar = tqdm(range(config.ITERATION), desc='Training Progress')
    for iteration in pbar:
        env.clear()
        agent.fill_replay_buffer(env, sfc_generator, 50)    # each episode contains sfc_generator.batch_size sfc
        episode_average_reward = np.mean(agent.episode_reward_list)
        reward_list.append(episode_average_reward)

        agent.train(5, 64)

        actor_loss = np.mean(agent.actor_loss_list)
        actor_loss_list.append(actor_loss)
        critic_loss = np.mean(agent.critic_loss_list)
        critic_loss_list.append(critic_loss)

        pbar.set_postfix({'Actor Loss': actor_loss, 'Critic Loss': critic_loss,'Reward': reward_list[iteration].item()})

        # plt.cla()
        # plt.subplot(1, 3, 1)
        # plt.title('Reward')
        # plt.plot(reward_list[:iteration], label='Reward', color='red')
        # plt.subplot(1, 3, 2)
        # plt.title('Actor Loss')
        # plt.plot(actor_loss_list[:iteration], label='Actor Loss', color='blue')
        # plt.subplot(1, 3, 3)
        # plt.title('Critic Loss')
        # plt.plot(critic_loss_list[:iteration], label='Critic Loss', color='green')
        # plt.legend()
        # plt.pause(0.1)

    # todo: save model
    # torch.save(agent, 'agent.pth')
    # load_agent = torch.load('agent.pth')

    # plt.ioff()

    plt.subplot(1, 3, 1)
    plt.title('Reward')
    plt.plot(reward_list, label='Reward', color='red')
    plt.subplot(1, 3, 2)
    plt.title('Actor Loss')
    plt.plot(actor_loss_list, label='Actor Loss', color='blue')
    plt.subplot(1, 3, 3)
    plt.title('Critic Loss')
    plt.plot(critic_loss_list, label='Critic Loss', color='green')
    plt.legend()

    plt.show()

    # test
    # print(reward_list)
    # print(actor_loss_list)
    # print(critic_loss_list)
