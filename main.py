import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
from torch_geometric.data import Data
import time
import environment
import sfc
import config
from agent import DDPG, NCO, EnhancedNCO
import plot

def train(agent):
    actor_loss_list = []
    critic_loss_list = []
    reward_list = []

    tqdm.write('-' * 20 + 'Agent training start' + '-' * 20 + '\t')

    pbar = tqdm(range(config.ITERATION), desc='Training Progress')
    for iteration in pbar:
        env.clear()
        agent.fill_replay_buffer(env, sfc_generator, 50)
        reward_list.append(np.mean(agent.episode_reward_list))

        agent.train(5, 8)

        actor_loss = np.mean(agent.actor_loss_list)
        actor_loss_list.append(actor_loss)
        critic_loss = np.mean(agent.critic_loss_list)
        critic_loss_list.append(critic_loss)

        pbar.set_postfix({'Actor Loss': actor_loss, 'Critic Loss': critic_loss,'Reward': reward_list[iteration].item()})

    training_time = time.time() - start_time
    print('Training complete in {:.2f} seconds.'.format(training_time))

    agent_name = agent.__class__.__name__

    agent.training_logs = {
        'reward_list': reward_list,
        'actor_loss_list': actor_loss_list,
        'critic_loss_list': critic_loss_list,
        'training_time': training_time
    }

    csv_file_path = 'save/result/' + agent_name + '.csv'
    df = pd.DataFrame({'Reward': reward_list, 'Actor Loss': actor_loss_list, 'Critic Loss': critic_loss_list})
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=True)
    print('Results saved to {}'.format(csv_file_path))

    agent_file_path = 'save/model/' + agent_name + '.pth'
    os.makedirs(os.path.dirname(agent_file_path), exist_ok=True)
    torch.save(agent, agent_file_path)
    print('Agent saved to {}'.format(agent_file_path))


def evaluate(agent, env, sfc_generator, episodes=10):
    agent.actor.eval()
    rewards = []
    for _ in range(episodes):
        env.clear()
        sfc_list = sfc_generator.get_sfc_batch()
        sfc_states = sfc_generator.get_sfc_states()
        for i, sfc in enumerate(sfc_list):
            net_state = Data(x=env.aggregate_features(), edge_index=env.get_edge_index())
            sfc_state = sfc_states[i]
            state = (net_state.to(agent.device), sfc_state.to(agent.device))
            with torch.no_grad():
                action = agent.select_action([state], exploration=False)
                placement = action[0][:len(sfc_list[i])].squeeze(0)
                env.step(sfc, placement)
                rewards.append(env.reward)
            env.clear_sfc()
    print(agent.__class__.__name__ + ' Test Average Reward:', np.mean(rewards))
    print(rewards)


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

    # input: batch_size * (num_nodes + max_sfc_length) * vnf_state_dim
    # output: batch_size * max_sfc_length * num_nodes
    # agent = DDPG(env.num_nodes, node_state_dim, vnf_state_dim, state_output_dim,
    #              config.MAX_SFC_LENGTH * env.num_nodes, device)

    # agent = NCO(vnf_state_dim, env.num_nodes, device)

    # agent = EnhancedNCO(env.num_nodes, node_state_dim, vnf_state_dim, state_output_dim,
    #              config.MAX_SFC_LENGTH * env.num_nodes, device)

    # train
    # train(agent)

    all_models = os.listdir('save/model')
    agent_files = [file for file in all_models if file.endswith('.pth')]

    for agent_file in agent_files:
        agent_file_path = 'save/model/' + agent_file
        agent = torch.load(agent_file_path, weights_only=False)
        evaluate(agent, env, sfc_generator)

    # plot.show_result('save/result')
