import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
import time
import plot
from environment import Environment
from sfc import SFCBatchGenerator
import config
from agent import NCO, DRLSFCP, EnhancedNCO, PPO

def train(agent, env, graph, sfc_generator, iteration):
    actor_loss_list = []
    critic_loss_list = []
    reward_list = []
    avg_acceptance_ratio_list = []

    agent_name = agent.__class__.__name__

    tqdm.write('-' * 20 + agent_name + ' Training Start' + '-' * 20 + '\t')

    pbar = tqdm(range(iteration), desc='Training Progress')

    start_time = time.time()
    for e in pbar:
        avg_episode_reward, avg_acceptance_ratio = agent.fill_replay_buffer(env, sfc_generator, config.EPISODE)    # fill replay buffer with n * config.BATCH_SIZE data
        agent.train()  # episode * batch_size, update parameters for episode times per batch size
        reward_list.append(avg_episode_reward)
        avg_acceptance_ratio_list.append(avg_acceptance_ratio)
        actor_loss_list.extend(agent.actor_loss_list)
        critic_loss_list.extend(agent.critic_loss_list)

        pbar.set_postfix({
            'Actor Loss': np.mean(agent.actor_loss_list),
            'Critic Loss': np.mean(agent.critic_loss_list),
            'Avg Episode Reward': avg_episode_reward,
            'Avg Acceptance Ratio': avg_acceptance_ratio
        })

    training_time = time.time() - start_time
    print('Training complete in {:.2f} seconds.'.format(training_time))

    # fill list to same length
    list_length = len(actor_loss_list)
    reward_list += [np.nan] * (list_length - len(reward_list))
    avg_acceptance_ratio_list += [np.nan] * (list_length - len(avg_acceptance_ratio_list))
    training_time_list = [training_time] + [np.nan] * (list_length - 1)

    agent.training_logs = {
        'reward_list': reward_list,
        'actor_loss_list': actor_loss_list,
        'critic_loss_list': critic_loss_list,
        'training_time': training_time
    }

    csv_file_path = 'save/result/' + graph + '/train/' + agent_name + '.csv'
    df = pd.DataFrame({'Reward': reward_list, 'Acceptance Ratio': avg_acceptance_ratio_list ,'Actor Loss': actor_loss_list, 'Critic Loss': critic_loss_list, 'Training Time': training_time_list})
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=True)
    print('Training results saved to {}'.format(csv_file_path))

    agent_file_path = 'save/model/' + graph + '/' + agent_name + '.pth'
    os.makedirs(os.path.dirname(agent_file_path), exist_ok=True)
    torch.save(agent, agent_file_path)
    print('Agent saved to {}'.format(agent_file_path))

def evaluate(agent_path, agent_name_list, graph, env, sfc_generator, batch_size_list, episodes):
    agent_dict = {}
    for agent_name in agent_name_list:
        agent_file_path = agent_path + agent_name + '.pth'
        agent = torch.load(agent_file_path, weights_only=False)
        agent.actor.eval()

        agent.placement_reward_list = []
        agent.power_consumption_list = []
        agent.exceeded_penalty_list = []
        agent.reward_list = []
        agent.acceptance_ratio_list = []
        agent.sfc_latency_list = []
        agent.exceeded_node_capacity_list = []
        agent.exceeded_link_bandwidth_list = []
        agent.running_time_list = []

        agent.avg_placement_reward_list = []
        agent.avg_power_consumption_list = []
        agent.avg_exceeded_penalty_list = []
        agent.avg_reward_list = []
        agent.avg_acceptance_ratio_list = []
        agent.avg_sfc_latency_list = []
        agent.avg_exceeded_node_capacity_list = []
        agent.avg_exceeded_link_bandwidth_list = []
        agent.avg_running_time_list = []

        agent_dict[agent_name] = agent

    print('-' * 20 + ' Evaluation Start' + '-' * 20)
    start_time = time.time()

    pbar = tqdm(batch_size_list, desc='Evaluation Progress')

    for batch_size in pbar:
        sfc_generator.set_batch_size(batch_size)
        for _ in range(episodes):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for agent in agent_dict.values():
                agent_start_time = time.time()
                agent.test(env, sfc_list, sfc_state_list, source_dest_node_pairs)
                agent.running_time_list.append(time.time() - agent_start_time)
                agent.placement_reward_list.append(np.sum(env.placement_reward_list))
                agent.power_consumption_list.append(np.sum(env.power_consumption_list))
                agent.exceeded_penalty_list.append(np.sum(env.exceeded_penalty_list))
                agent.reward_list.append(np.sum(env.reward_list))
                agent.acceptance_ratio_list.append(env.sfc_placed_num / batch_size)
                agent.sfc_latency_list.append(np.mean(env.sfc_latency_list))
                agent.exceeded_node_capacity_list.append(np.max((0, env.exceeded_node_capacity_list[-1])))
                agent.exceeded_link_bandwidth_list.append(np.max((0, env.exceeded_link_bandwidth_list[-1])))

        for agent in agent_dict.values():
            agent.avg_placement_reward_list.append(np.mean(agent.placement_reward_list))    # iteration avg reward
            agent.avg_power_consumption_list.append(np.mean(agent.power_consumption_list))
            agent.avg_exceeded_penalty_list.append(np.mean(agent.exceeded_penalty_list))
            agent.avg_reward_list.append(np.mean(agent.reward_list))
            agent.avg_acceptance_ratio_list.append(np.mean(agent.acceptance_ratio_list))
            agent.avg_sfc_latency_list.append((np.mean(agent.sfc_latency_list)))
            agent.avg_exceeded_node_capacity_list.append(np.mean(agent.exceeded_node_capacity_list))
            agent.avg_exceeded_link_bandwidth_list.append(np.mean(agent.exceeded_link_bandwidth_list))
            agent.avg_running_time_list.append(np.mean(agent.running_time_list))

            agent.placement_reward_list.clear()
            agent.power_consumption_list.clear()
            agent.exceeded_penalty_list.clear()
            agent.reward_list.clear()
            agent.acceptance_ratio_list.clear()
            agent.sfc_latency_list.clear()
            agent.exceeded_node_capacity_list.clear()
            agent.exceeded_link_bandwidth_list.clear()
            agent.running_time_list.clear()

    evaluation_time = time.time() - start_time
    print('Evaluation complete in {:.2f} seconds.'.format(evaluation_time))

    for agent_name in agent_name_list:
        agent = agent_dict[agent_name]
        csv_file_path = 'save/result/' + graph + '/evaluate/' + agent_name + '.csv'
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df = pd.DataFrame({
            'Number of SFC': batch_size_list,
            'Average Placement Reward': agent.avg_placement_reward_list,
            'Average Power Consumption': agent.avg_power_consumption_list,
            'Average Exceeded Penalty': agent.avg_exceeded_penalty_list,
            'Average Episode Reward': agent.avg_reward_list,
            'Average Acceptance Ratio': agent.avg_acceptance_ratio_list,
            'Average SFC End-to-End Latency': agent.avg_sfc_latency_list,
            'Average Exceeded Node Resource Usage': agent.avg_exceeded_node_capacity_list,
            'Average Exceeded Link Bandwidth Usage': agent.avg_exceeded_link_bandwidth_list,
            'Average Running Time': agent.avg_running_time_list
        })
        df.to_csv(csv_file_path, index=False)
        print('Evaluation results saved to {}'.format(csv_file_path))

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # initialization
    # graph = 'Cogentco' # 197 nodes and 245 links
    graph = 'Chinanet'    # 42 nodes and 66 links

    G = nx.read_graphml('graph/' + graph + '.graphml')
    env = Environment(G)
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
        NCO(node_state_dim, vnf_state_dim, env.num_nodes, device),
        EnhancedNCO(env.num_nodes, node_state_dim, vnf_state_dim, device),
        DRLSFCP(env.num_nodes, node_state_dim, vnf_state_dim, device),
        PPO(env.num_nodes, node_state_dim, vnf_state_dim, device)
    ]

    for agent in agent_list:
        train(agent, env, graph, sfc_generator, iteration=config.ITERATION)

    # evaluate
    agent_path = 'save/model/' + graph + '/'
    agent_name_list = [
        'NCO',
        'EnhancedNCO',
        'DRLSFCP',
        'PPO',
        ]

    # batch_size_list = [60, 70, 80, 90, 100]
    batch_size_list = [15, 20, 25, 30, 35]

    evaluate(agent_path, agent_name_list, graph, env, sfc_generator, batch_size_list, episodes=50)

    result_path = 'save/result/' + graph

    plot.show_train_result(graph, result_path + '/train', agent_name_list)
    plot.show_evaluate_result(graph, result_path + '/evaluate', agent_name_list)