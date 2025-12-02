import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def show_train_result(graph, dir_path, agent_name_list):
    agent_num = len(agent_name_list)

    df_list = []
    reward_list = []
    avg_acceptance_ratio_list = []
    actor_loss_list = []
    critic_loss_list = []

    for agent_name in agent_name_list:
        csv_file_path = dir_path + '/' + agent_name + '.csv'
        df = pd.read_csv(csv_file_path)
        df_list.append(df)
        reward_list.append(df['Reward'])
        avg_acceptance_ratio_list.append(df['Acceptance Ratio'])
        actor_loss_list.append(df['Actor Loss'])
        critic_loss_list.append(df['Critic Loss'])

    figure_path = 'save/result/' + graph + '/plot/'
    os.makedirs(figure_path, exist_ok=True)
    colors = ['#cc7c71', '#925eb0', '#72b063', '#719aac', '#e29135']
    episode = range(len(actor_loss_list[0]))
    window_size = 50

    # Actor Loss
    plt.figure(figsize=(10, 6))
    for i in range(agent_num):
        df = pd.DataFrame({'Episode': episode, 'Actor Loss': actor_loss_list[i]})
        df['Smoothed Actor Loss'] = df['Actor Loss'].rolling(window=window_size, center=True).mean()
        sns.lineplot(data=df, x='Episode', y='Actor Loss', color=colors[i], alpha=0.2, label=None)
        sns.lineplot(data=df, x='Episode', y='Smoothed Actor Loss', color=colors[i], label=agent_name_list[i] + ' Smoothed Actor Loss')
    plt.title('Actor Training Loss Curve with Smoothing')
    plt.legend()

    # Critic Loss
    plt.figure(figsize=(10, 6))
    for i in range(agent_num):
        df = pd.DataFrame({'Episode': episode, 'Critic Loss': critic_loss_list[i]})
        df['Smoothed Critic Loss'] = df['Critic Loss'].rolling(window=window_size, center=True).mean()
        sns.lineplot(data=df, x='Episode', y='Critic Loss', color=colors[i], alpha=0.2,
                     label=None)
        sns.lineplot(data=df, x='Episode', y='Smoothed Critic Loss', color=colors[i],
                     label=agent_name_list[i] + ' Smoothed Critic Loss')
    plt.title('Critic Training Loss Curve with Smoothing')
    plt.legend()
    plt.show()

def show_evaluate_result(graph, dir_path, agent_name_list):

    # load csv files
    df_list = []
    for agent_name in agent_name_list:
        csv_file_path = dir_path + '/' + agent_name + '.csv'
        df = pd.read_csv(csv_file_path)
        df_list.append(df)

    # row number in csv files
    index_num = len(df_list[0]) if df_list else 0

    bar_width = 0.2
    index = np.arange(index_num)    # bar location
    labels = df_list[0]['Number of SFC']
    colors = ['#925eb0', '#e29135', '#72b063', '#94c6cd', '#cc7c71']

    figure_path = 'save/result/' + graph +'/plot/'
    os.makedirs(figure_path, exist_ok=True)

    for metric in df_list[0].columns.tolist()[1:]:
        plt.figure(figsize=(10, 6))
        for i, (df, agent_name) in enumerate(zip(df_list, agent_name_list)):
            plt.bar(
                index + i * bar_width,  # bar offset
                df[metric],
                width=bar_width,
                label=agent_name,
                color=colors[i]
            )

        # add data text
        for i, df in enumerate(df_list):
            for j, value in enumerate(df[metric]):
                plt.text(
                    j + i * bar_width,  # x
                    value,  # y
                    f'{value:.2f}',
                    ha='center',
                    va='bottom' if value >= 0 else 'top'
                )

        plt.xlabel('Number of SFC', fontsize=12)
        plt.ylabel(metric, fontsize=12)

        plt.xticks(index + bar_width, labels=labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_path + metric + '.png', dpi=300)

    plt.show()

def show_results(runner):
    agent_name_list = [agent.__class__.__name__ for agent in runner.agent_list]
    show_train_result(runner.graph, runner.result_path + '/train', agent_name_list)
    show_evaluate_result(runner.graph, runner.result_path + '/evaluate', agent_name_list)

if __name__ == '__main__':
    agent_name_list = [
        'NCO',
        'EnhancedNCO',
        'DRLSFCP',
        'PPO',
        ]

    # show_train_result('Cogentco', 'save/result/Cogentco/train', agent_name_list)
    # show_evaluate_result('Cogentco', 'save/result/Cogentco/evaluate', agent_name_list)
    # show_train_result('Chinanet', 'save/result/Chinanet/train', agent_name_list)
    # show_evaluate_result('Chinanet', 'save/result/Chinanet/evaluate', agent_name_list)