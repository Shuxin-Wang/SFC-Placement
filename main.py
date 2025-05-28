import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data
import time
import environment
import sfc
import config
from agent import DDPG

def evaluate(agent, env, sfc_generator, episodes=10):
    agent.actor.eval()
    rewards = []
    for _ in range(episodes):
        env.clear()
        sfc_list = sfc_generator.get_sfc_batch()
        sfc_states = sfc_generator.get_sfc_states()
        for i, sfc in enumerate(sfc_list):
            net_state = Data(x=env.aggregate_features(), edge_index=env.edge_index())
            sfc_state = sfc_states[i]
            state = (net_state.to(agent.device), sfc_state.to(agent.device))
            with torch.no_grad():
                action = agent.select_action([state], exploration=False)
                placement = agent.actor.get_sfc_placement(action, env.num_nodes)
                placement = placement[0][:len(sfc)]
                _, reward = env.step(sfc, placement)
                rewards.append(reward)
            env.clear_sfc()
    agent.actor.train()
    print('Test Average Reward:', np.mean(rewards))


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    start_time = time.time()

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
        agent.fill_replay_buffer(env, sfc_generator, 50)
        reward_list.append(agent.episode_reward)

        agent.train(5, 8)

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

    print('Training complete in {:.2f} seconds.'.format(time.time() - start_time))

    # plt.ioff()

    plt.cla()
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

    plt.savefig('result.png', dpi=300)
    plt.show()

    agent.training_logs = {
        'reward_list': reward_list,
        'actor_loss_list': actor_loss_list,
        'critic_loss_list': critic_loss_list
    }

    csv_file_path = 'results.csv'
    df = pd.DataFrame({'Reward': reward_list, 'Actor Loss': actor_loss_list, 'Critic Loss': critic_loss_list})
    df.to_csv(csv_file_path, index=True)
    print('Results saved to {}'.format(csv_file_path))

    agent_name = agent.__class__.__name__
    file_name = agent_name + '.pth'
    torch.save(agent, file_name)
    print('Agent saved to {}'.format(file_name))

    # test
    # print(reward_list)
    # print(actor_loss_list)
    # print(critic_loss_list)
