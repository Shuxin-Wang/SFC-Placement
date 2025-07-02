import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
import time
import plot
import environment
from sfc import SFCBatchGenerator
import config
from agent import NCO, ActorEnhancedNCO, CriticEnhancedNCO, DDPG, DRLSFCP, EnhancedNCO

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


def evaluate(agent_name_list, env, sfc_generator, sfc_length_list, episodes=50):
    agent_dict = {}
    for agent_name in agent_name_list:
        agent_file_path = agent_path + agent_name + '.pth'
        agent = torch.load(agent_file_path, weights_only=False)

        agent.placement_reward_list = []
        agent.power_consumption_list = []
        agent.exceeded_penalty_list = []
        agent.reward_list = []
        agent.acceptance_ratio_list = []

        agent.avg_placement_reward_list = []
        agent.avg_power_consumption_list = []
        agent.avg_exceeded_penalty_list = []
        agent.avg_reward_list = []
        agent.avg_acceptance_ratio_list = []
        agent.actor.eval()

        agent_dict[agent_name] = agent

    print('-' * 20 + ' Evaluation Start' + '-' * 20)
    start_time = time.time()

    pbar = tqdm(sfc_length_list, desc='Evaluation Progress')

    for sfc_length in pbar:
        sfc_generator.max_sfc_length = sfc_length
        for _ in range(episodes):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for agent in agent_dict.values():
                env.clear()
                agent.test(env, sfc_list, sfc_state_list, source_dest_node_pairs)
                agent.placement_reward_list.append(np.mean(env.placement_reward_list))    # episode avg reward
                agent.power_consumption_list.append(np.mean(env.power_consumption_list))
                agent.exceeded_penalty_list.append(np.mean(env.exceeded_penalty_list))
                agent.reward_list.append(np.mean(env.reward_list))
                agent.acceptance_ratio_list.append(env.sfc_placed_num / sfc_generator.batch_size)

        for agent in agent_dict.values():
            agent.avg_placement_reward_list.append(np.mean(agent.placement_reward_list))    # iteration avg reward
            agent.avg_power_consumption_list.append(np.mean(agent.power_consumption_list))
            agent.avg_exceeded_penalty_list.append(np.mean(agent.exceeded_penalty_list))
            agent.avg_reward_list.append(np.mean(agent.reward_list))
            agent.avg_acceptance_ratio_list.append(np.mean(agent.acceptance_ratio_list))

            agent.placement_reward_list.clear()
            agent.power_consumption_list.clear()
            agent.exceeded_penalty_list.clear()
            agent.reward_list.clear()
            agent.acceptance_ratio_list.clear()

    evaluation_time = time.time() - start_time
    print('Evaluation complete in {:.2f} seconds.'.format(evaluation_time))

    for agent_name in agent_name_list:
        agent = agent_dict[agent_name]
        csv_file_path = 'save/result/evaluate/' + agent_name + '.csv'
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df = pd.DataFrame({
            'Max SFC Length': sfc_length_list,
            'Average Placement Reward': agent.avg_placement_reward_list,
            'Average Power Consumption': agent.avg_power_consumption_list,
            'Average Exceeded Penalty': agent.avg_exceeded_penalty_list,
            'Average Reward': agent.avg_reward_list,
            'Average Acceptance Ratio': agent.avg_acceptance_ratio_list
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
        # NCO(vnf_state_dim, env.num_nodes, device),
        # ActorEnhancedNCO(env.num_nodes, node_state_dim, vnf_state_dim, vnf_state_dim,
        #                 env.num_nodes, device),
        EnhancedNCO(env.num_nodes, node_state_dim, vnf_state_dim, vnf_state_dim,
                         env.num_nodes, device),
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
        # 'DRLSFCP',
        'ActorEnhancedNCO',
        # 'CriticEnhancedNCO',
        'EnhancedNCO',
        'DDPG'
        ]
    sfc_length_list = [8, 10, 12, 16, 20, 24]   # test agent placement under different max sfc length
    evaluate(agent_name_list, env, sfc_generator, sfc_length_list, episodes=5)

    # plot.show_train_result('save/result/train', agent_name_list)
    plot.show_evaluate_result('save/result/evaluate', agent_name_list)