import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
from torch_geometric.data import Data
import time
import plot
import environment
from sfc import SFCBatchGenerator
import config
from agent import NCO, ActorEnhancedNCO, CriticEnhancedNCO, DDPG, DRLSFCP

def train(agent, env, sfc_generator, iteration):
    actor_loss_list = []
    critic_loss_list = []
    reward_list = []

    agent_name = agent.__class__.__name__

    tqdm.write('-' * 20 + agent_name + ' Training Start' + '-' * 20 + '\t')

    pbar = tqdm(range(iteration), desc='Training Progress')

    start_time = time.time()
    for _ in pbar:
        agent.fill_replay_buffer(env, sfc_generator, 5)    # fill replay buffer with 50 * config.BATCH_SIZE data

        agent.train(10, 10)  # episode * batch_size, update parameters for episode times per batch size
        actor_loss_list.extend(agent.actor_loss_list)
        critic_loss_list.extend(agent.critic_loss_list)

        agent.test(env, sfc_generator)
        reward_list.append(agent.episode_reward)

        pbar.set_postfix({
            'Actor Loss': np.mean(agent.actor_loss_list),
            'Critic Loss': np.mean(agent.critic_loss_list),
            'Episode Reward': agent.episode_reward
        })

    training_time = time.time() - start_time
    print('Training complete in {:.2f} seconds.'.format(training_time))

    # fill list to same length
    list_length = len(actor_loss_list)
    reward_list += [np.nan] * (list_length - len(reward_list))
    training_time_list = [training_time] + [np.nan] * (list_length - 1)

    agent.training_logs = {
        'reward_list': reward_list,
        'actor_loss_list': actor_loss_list,
        'critic_loss_list': critic_loss_list,
        'training_time': training_time
    }

    csv_file_path = 'save/result/train/' + agent_name + '.csv'
    df = pd.DataFrame({'Reward': reward_list, 'Actor Loss': actor_loss_list, 'Critic Loss': critic_loss_list, 'Training Time': training_time_list})
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
    avg_acceptance_ratio_list = []

    print('-' * 20 + agent_name + ' Evaluation Start' + '-' * 20)
    start_time = time.time()

    pbar = tqdm(sfc_length_list, desc='Evaluation Progress')

    for sfc_length in pbar:
        env.clear()
        sfc_generator.max_sfc_length = sfc_length
        for _ in range(episodes):
            agent.test(env, sfc_generator)
            placement_reward_list.append(np.mean(env.placement_reward_list))    # episode avg reward
            power_consumption_list.append(np.mean(env.power_consumption_list))
            exceeded_penalty_list.append(np.mean(env.exceeded_penalty_list))
            reward_list.append(np.mean(env.reward_list))
            acceptance_ratio_list.append(env.sfc_placed_num / sfc_generator.batch_size)

        avg_placement_reward_list.append(np.mean(placement_reward_list))    # iteration avg reward
        avg_power_consumption_list.append(np.mean(power_consumption_list))
        avg_exceeded_penalty_list.append(np.mean(exceeded_penalty_list))
        avg_reward_list.append(np.mean(reward_list))
        avg_acceptance_ratio_list.append(np.mean(acceptance_ratio_list))

        placement_reward_list.clear()
        power_consumption_list.clear()
        exceeded_penalty_list.clear()
        reward_list.clear()
        acceptance_ratio_list.clear()

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
        'Average Acceptance Ratio': avg_acceptance_ratio_list
    })
    df.to_csv(csv_file_path, index=False)
    print('Evaluation results saved to {}'.format(csv_file_path))


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # initialization
    G = nx.read_graphml('graph/Cogentco.graphml')
    env = environment.Environment(G)
    sfc_generator = SFCBatchGenerator(config.BATCH_SIZE, config.MIN_SFC_LENGTH, config.MAX_SFC_LENGTH,
                                          config.NUM_VNF_TYPES, env.num_nodes)

    env.clear()
    env.get_state_dim(sfc_generator)

    node_state_dim = env.node_state_dim
    vnf_state_dim = env.vnf_state_dim
    state_dim = env.state_dim
    state_input_dim = node_state_dim * env.num_nodes + config.MAX_SFC_LENGTH * vnf_state_dim
    state_output_dim = (env.num_nodes + config.MAX_SFC_LENGTH + 2) * vnf_state_dim

    # train
    agent_list = [
        NCO(vnf_state_dim, env.num_nodes, device),
        # ActorEnhancedNCO(env.num_nodes, node_state_dim, vnf_state_dim, state_output_dim,
        #                 config.MAX_SFC_LENGTH * env.num_nodes, device),
        # CriticEnhancedNCO(env.num_nodes, node_state_dim, vnf_state_dim, device),
        # DDPG(env.num_nodes, node_state_dim, vnf_state_dim, state_output_dim,
        #      config.MAX_SFC_LENGTH * env.num_nodes, device),
        # DRLSFCP(node_state_dim, vnf_state_dim, device=device)
    ]

    # for agent in agent_list:
    #     train(agent, env, sfc_generator, iteration=config.ITERATION)

    # evaluate
    agent_path = 'save/model/'
    agent_name_list = [
        'NCO',
        'DRLSFCP',
        'ActorEnhancedNCO',
        # 'CriticEnhancedNCO',
        'DDPG'
        ]

    sfc_length_list = [8, 10, 12, 16, 20, 24]   # test agent placement under different max sfc length
    for agent_name in agent_name_list:
        agent_file_path = agent_path + agent_name + '.pth'
        agent = torch.load(agent_file_path, weights_only=False)
        evaluate(agent, env, sfc_generator, sfc_length_list)

    # plot.show_train_result('save/result/train', agent_name_list)
    plot.show_evaluate_result('save/result/evaluate', agent_name_list)