import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import torch
import warnings
import plot
from environment import Environment
from sfc import SFCBatchGenerator
from agent import NCO, DRLSFCP, EnhancedNCO, PPO

class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # env init
        self.graph = cfg.graph
        self.batch_size_list = cfg.batch_size_list
        self.env = Environment(cfg)
        self.sfc_generator = SFCBatchGenerator(cfg)
        self.env.clear()

        # model init
        self.agent_list = None
        self.model = cfg.model
        self.set_model()

        self.iteration = cfg.iteration
        self.episode = cfg.episode

        self.save_model = cfg.save_model

        self.agent_path = cfg.agent_path
        self.result_path = cfg.result_path

        warnings.filterwarnings("ignore")
        
    def set_model(self):
        if self.model == 'all':
            self.agent_list = [
                NCO(self.cfg, self.env, self.sfc_generator, self.device),
                EnhancedNCO(self.cfg, self.env, self.sfc_generator, self.device),
                DRLSFCP(self.cfg, self.env, self.sfc_generator, self.device),
                PPO(self.cfg, self.env, self.sfc_generator, self.device)
            ]
        elif self.model == 'NCO':
            self.agent_list = [
                NCO(self.cfg, self.env, self.sfc_generator, self.device)
            ]
        elif self.model == 'EnhancedNCO':
            self.agent_list = [
                EnhancedNCO(self.cfg, self.env, self.sfc_generator, self.device)
            ]
        elif self.model == 'DRLSFCP':
            self.agent_list = [
                DRLSFCP(self.cfg, self.env, self.sfc_generator, self.device)
            ]
        elif self.model == 'PPO':
            self.agent_list = [
                PPO(self.cfg, self.env, self.sfc_generator, self.device)
            ]
        else:
            raise ValueError('Invalid model name.')

    def train(self):
        tqdm.write('-' * 30 + ' Training Start' + '-' * 30 + '\t')
        time.sleep(0.1)
        for agent in self.agent_list:
            actor_loss_list = []
            critic_loss_list = []
            reward_list = []
            avg_acceptance_ratio_list = []

            agent_name = agent.__class__.__name__

            pbar = tqdm(range(self.iteration), desc = agent_name + ' | Training Progress')

            start_time = time.time()
            for e in pbar:
                avg_episode_reward, avg_acceptance_ratio = agent.fill_replay_buffer(self.env, self.sfc_generator,
                                                                                    self.episode)  # fill replay buffer with n * args.batch_size data
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
                'avg_acceptance_ratio_list': avg_acceptance_ratio_list,
                'actor_loss_list': actor_loss_list,
                'critic_loss_list': critic_loss_list,
                'training_time': training_time
            }

            self.save_training_results(agent)

            if self.save_model:
                self.save_training_model(agent)

            time.sleep(0.1)

    def evaluate(self):
        print('-' * 30 + ' Evaluation Start' + '-' * 30)
        time.sleep(0.1)
        for agent in self.agent_list:
            agent_name = agent.__class__.__name__
            agent_file_path = self.agent_path + agent_name + '.pth'
            agent.load_state_dict(torch.load(agent_file_path, weights_only=True))
            agent.actor.eval()

            agent.placement_reward_list = []
            agent.power_consumption_list = []
            agent.exceeded_penalty_list = []
            agent.acceptance_ratio_list = []
            agent.sfc_latency_list = []
            agent.exceeded_node_capacity_list = []
            agent.exceeded_link_bandwidth_list = []
            agent.running_time_list = []

            agent.avg_placement_reward_list = []
            agent.avg_power_consumption_list = []
            agent.avg_exceeded_penalty_list = []
            agent.avg_acceptance_ratio_list = []
            agent.avg_sfc_latency_list = []
            agent.avg_exceeded_node_capacity_list = []
            agent.avg_exceeded_link_bandwidth_list = []
            agent.avg_running_time_list = []

            start_time = time.time()

            pbar = tqdm(self.batch_size_list, desc = agent.__class__.__name__ + ' | Evaluation Progress')

            for batch_size in pbar:
                self.sfc_generator.set_batch_size(batch_size)
                for _ in range(self.episode):
                    sfc_list = self.sfc_generator.get_sfc_batch()
                    sfc_state_list, source_dest_node_pairs, reliability_requirement_list = self.sfc_generator.get_sfc_states()

                    agent_start_time = time.time()
                    agent.test(self.env, sfc_list, sfc_state_list, source_dest_node_pairs, reliability_requirement_list)
                    agent.running_time_list.append(time.time() - agent_start_time)
                    agent.placement_reward_list.append(np.sum(self.env.placement_reward_list))
                    agent.power_consumption_list.append(np.sum(self.env.power_consumption_list))
                    agent.exceeded_penalty_list.append(np.sum(self.env.exceeded_penalty_list))
                    agent.acceptance_ratio_list.append(self.env.sfc_placed_num / batch_size)
                    agent.sfc_latency_list.append(np.mean(self.env.sfc_latency_list))
                    agent.exceeded_node_capacity_list.append(np.max((0, self.env.exceeded_node_capacity_list[-1])))
                    agent.exceeded_link_bandwidth_list.append(np.max((0, self.env.exceeded_link_bandwidth_list[-1])))

                agent.avg_placement_reward_list.append(np.mean(agent.placement_reward_list))
                agent.avg_power_consumption_list.append(np.mean(agent.power_consumption_list))
                agent.avg_exceeded_penalty_list.append(np.mean(agent.exceeded_penalty_list))
                agent.avg_acceptance_ratio_list.append(np.mean(agent.acceptance_ratio_list))
                agent.avg_sfc_latency_list.append((np.mean(agent.sfc_latency_list)))
                agent.avg_exceeded_node_capacity_list.append(np.mean(agent.exceeded_node_capacity_list))
                agent.avg_exceeded_link_bandwidth_list.append(np.mean(agent.exceeded_link_bandwidth_list))
                agent.avg_running_time_list.append(np.mean(agent.running_time_list))

                agent.placement_reward_list.clear()
                agent.power_consumption_list.clear()
                agent.exceeded_penalty_list.clear()
                agent.acceptance_ratio_list.clear()
                agent.sfc_latency_list.clear()
                agent.exceeded_node_capacity_list.clear()
                agent.exceeded_link_bandwidth_list.clear()
                agent.running_time_list.clear()

            evaluation_time = time.time() - start_time
            print('Evaluation complete in {:.2f} seconds.'.format(evaluation_time))

            self.save_evaluation_results(agent)

            time.sleep(0.1)

    def save_training_results(self, agent):
        csv_file_path = 'save/result/' + self.graph + '/train/' + agent.__class__.__name__ + '.csv'

        list_length = len(agent.training_logs['actor_loss_list'])
        reward_list = agent.training_logs['reward_list']
        avg_acceptance_ratio_list = agent.training_logs['avg_acceptance_ratio_list']
        actor_loss_list = agent.training_logs['actor_loss_list']
        critic_loss_list = agent.training_logs['critic_loss_list']
        training_time_list = [agent.training_logs['training_time']] + [np.nan] * (list_length - 1)

        df = pd.DataFrame({'Reward': reward_list,
                           'Acceptance Ratio': avg_acceptance_ratio_list,
                           'Actor Loss': actor_loss_list,
                           'Critic Loss': critic_loss_list,
                           'Training Time': training_time_list})

        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, index=True)
        print('Training results saved to {}'.format(csv_file_path))

    def save_training_model(self, agent):
        agent_file_path = 'save/model/' + self.graph + '/' + agent.__class__.__name__ + '.pth'
        os.makedirs(os.path.dirname(agent_file_path), exist_ok=True)
        torch.save(agent.state_dict(), agent_file_path)
        print('Agent saved to {}'.format(agent_file_path))

    def save_evaluation_results(self, agent):
        csv_file_path = 'save/result/' + self.graph + '/evaluate/' + agent.__class__.__name__ + '.csv'
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df = pd.DataFrame({
            'Number of SFC': self.batch_size_list,
            'Average Placement Reward': agent.avg_placement_reward_list,
            'Average Power Consumption': agent.avg_power_consumption_list,
            'Average Exceeded Penalty': agent.avg_exceeded_penalty_list,
            'Average Acceptance Ratio': agent.avg_acceptance_ratio_list,
            'Average SFC End-to-End Latency': agent.avg_sfc_latency_list,
            'Average Exceeded Node Resource Usage': agent.avg_exceeded_node_capacity_list,
            'Average Exceeded Link Bandwidth Usage': agent.avg_exceeded_link_bandwidth_list,
            'Average Running Time': agent.avg_running_time_list
        })
        df.to_csv(csv_file_path, index=False)
        print('Evaluation results saved to {}'.format(csv_file_path))

    def show_results(self):
        agent_name_list = [agent.__class__.__name__ for agent in self.agent_list]
        plot.show_train_result(self.graph, self.result_path + '/train', agent_name_list)
        plot.show_evaluate_result(self.graph, self.result_path + '/evaluate', agent_name_list)