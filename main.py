import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
from torch_geometric.data import Data
import time
import environment
import main
from sfc import SFCBatchGenerator
import config
from agent import DDPG, NCO, EnhancedNCO
import plot

def train(agent, env, sfc_generator):
    actor_loss_list = []
    critic_loss_list = []
    reward_list = []

    agent_name = agent.__class__.__name__

    tqdm.write('-' * 20 + agent_name + ' Training Start' + '-' * 20 + '\t')

    pbar = tqdm(range(config.ITERATION), desc='Training Progress')

    start_time = time.time()
    for iteration in pbar:
        env.clear()
        agent.fill_replay_buffer(env, sfc_generator, 50)
        reward_list.append(np.mean(agent.episode_reward_list))

        agent.train(5, 10)

        actor_loss = np.mean(agent.actor_loss_list)
        actor_loss_list.append(actor_loss)
        critic_loss = np.mean(agent.critic_loss_list)
        critic_loss_list.append(critic_loss)

        pbar.set_postfix({'Actor Loss': actor_loss, 'Critic Loss': critic_loss,'Avg Episode Reward': reward_list[iteration].item()})
    training_time = time.time() - start_time
    print('Training complete in {:.2f} seconds.'.format(training_time))

    agent.training_logs = {
        'reward_list': reward_list,
        'actor_loss_list': actor_loss_list,
        'critic_loss_list': critic_loss_list,
        'training_time': training_time
    }

    csv_file_path = 'save/result/train/' + agent_name + '.csv'
    df = pd.DataFrame({'Reward': reward_list, 'Actor Loss': actor_loss_list, 'Critic Loss': critic_loss_list})
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=True)
    print('Training results saved to {}'.format(csv_file_path))

    agent_file_path = 'save/model/' + agent_name + '.pth'
    os.makedirs(os.path.dirname(agent_file_path), exist_ok=True)
    torch.save(agent, agent_file_path)
    print('Agent saved to {}'.format(agent_file_path))


def evaluate(agent, env, sfc_generator, sfc_length_list, episodes=10):
    agent_name = agent.__class__.__name__
    agent.actor.eval()

    placement_reward_list = []
    power_consumption_list = []
    exceeded_penalty_list = []
    reward_list = []
    acceptance_ratio_list = []

    avg_placement_reward_list = []
    avg_power_consumption_list = []
    avg_exceeded_penalty_list = []
    avg_reward_list = []

    print('-' * 20 + agent_name + ' Evaluation Start' + '-' * 20)
    start_time = time.time()
    for sfc_length in sfc_length_list:
        sfc_generator.max_sfc_length = sfc_length
        placement_reward_list.clear()
        power_consumption_list.clear()
        exceeded_penalty_list.clear()
        reward_list.clear()
        sfc_placed = 0  # record the number of successfully placed sfc
        for _ in range(episodes):
            env.clear()
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_states = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for i, sfc in enumerate(sfc_list):
                net_state = Data(x=env.aggregate_features(), edge_index=env.get_edge_index())
                sfc_state = sfc_states[i]
                state = (net_state.to(agent.device), sfc_state.to(agent.device))
                with torch.no_grad():
                    action = agent.select_action([state], exploration=False)
                    placement = action[0][:len(sfc_list[i])].squeeze(0)
                    env.step(sfc, placement)
                    placement_reward_list.append(env.placement_reward)
                    power_consumption_list.append(env.power_consumption)
                    exceeded_penalty_list.append(env.exceeded_penalty)
                    reward_list.append(env.reward)
                sfc_placed += (len(sfc) == sum(env.vnf_placement))
                env.clear_sfc()

        avg_placement_reward_list.append(np.mean(placement_reward_list))
        avg_power_consumption_list.append(np.mean(power_consumption_list))
        avg_exceeded_penalty_list.append(np.mean(exceeded_penalty_list))
        avg_reward_list.append(np.mean(reward_list))
        acceptance_ratio_list.append(sfc_placed / episodes / config.BATCH_SIZE)

    evaluation_time = time.time() - start_time
    print('Evaluation complete in {:.2f} seconds.'.format(evaluation_time))

    csv_file_path = 'save/result/evaluate/' + agent_name + '.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df = pd.DataFrame({
        'Max SFC Length': sfc_length_list,
        'Average Placement Reward': avg_placement_reward_list,
        'Average Power Consumption': avg_power_consumption_list,
        'Average Exceeded Penalty': avg_exceeded_penalty_list,
        'Average Reward': avg_reward_list,
        'Acceptance Ratio': acceptance_ratio_list
    })
    df.to_csv(csv_file_path, index=False)
    print('Evaluation results saved to {}'.format(csv_file_path))


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # initialization
    G = nx.read_graphml('Cogentco.graphml')
    env = environment.Environment(G)
    sfc_generator = SFCBatchGenerator(20, config.MIN_SFC_LENGTH, config.MAX_SFC_LENGTH,
                                          config.NUM_VNF_TYPES, env.num_nodes)

    env.clear()
    env.get_state_dim(sfc_generator)

    node_state_dim = env.node_state_dim
    vnf_state_dim = env.vnf_state_dim
    state_dim = env.state_dim
    state_input_dim = node_state_dim * env.num_nodes + config.MAX_SFC_LENGTH * vnf_state_dim
    state_output_dim = (env.num_nodes + config.MAX_SFC_LENGTH + 2) * vnf_state_dim

    # input: batch_size * (num_nodes + max_sfc_length) * vnf_state_dim
    # output: batch_size * max_sfc_length * num_nodes

    # train
    # agent_list = [
    #     NCO(vnf_state_dim, env.num_nodes, device),
    #     EnhancedNCO(env.num_nodes, node_state_dim, vnf_state_dim, state_output_dim,
    #                  config.MAX_SFC_LENGTH * env.num_nodes, device),
    #     DDPG(env.num_nodes, node_state_dim, vnf_state_dim, state_output_dim,
    #          config.MAX_SFC_LENGTH * env.num_nodes, device)
    # ]

    # for agent in agent_list:
    #     train(agent, env, sfc_generator)

    # evaluate
    all_models = os.listdir('save/model')
    agent_files = [file for file in all_models if file.endswith('.pth')]

    sfc_length_list = [8, 10, 12]   # test agent placement under different max sfc length
    for agent_file in agent_files:
        agent_file_path = 'save/model/' + agent_file
        agent = torch.load(agent_file_path, weights_only=False)
        evaluate(agent, env, sfc_generator, sfc_length_list)

    # plot.show_evaluate_result('save/result/evaluate')
