import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def show_train_result(dir_path, agent_name_list):
    agent_num = len(agent_name_list)

    df_list = []
    reward_list = []
    actor_loss_list = []
    critic_loss_list = []

    for agent_name in agent_name_list:
        csv_file_path = dir_path + '/' + agent_name + '.csv'
        df = pd.read_csv(csv_file_path)
        df_list.append(df)
        reward_list.append(df['Reward'])
        actor_loss_list.append(df['Actor Loss'])
        critic_loss_list.append(df['Critic Loss'])

    figure_path = 'save/result/plot/'
    colors = ['#cc7c71', '#925eb0', '#72b063', '#719aac', '#e29135']
    steps = range(len(actor_loss_list[0]))
    window_size = 10

    # Actor Loss
    plt.figure(figsize=(10, 6))
    for i in range(agent_num):
        df = pd.DataFrame({'Steps': steps, 'Actor Loss': actor_loss_list[i]})
        df['Smoothed Actor Loss'] = df['Actor Loss'].rolling(window=window_size, center=True).mean()
        sns.lineplot(data=df, x='Steps', y='Actor Loss', color=colors[i], alpha=0.2, label=agent_name_list[i] + ' Actor Loss')
        sns.lineplot(data=df, x='Steps', y='Smoothed Actor Loss', color=colors[i], label=agent_name_list[i] + ' Smoothed Actor Loss')
    plt.title('Actor Training Loss Curve with Smoothing')
    plt.legend()

    # Critic Loss
    plt.figure(figsize=(10, 6))
    for i in range(agent_num):
        df = pd.DataFrame({'Steps': steps, 'Critic Loss': critic_loss_list[i]})
        df['Smoothed Critic Loss'] = df['Critic Loss'].rolling(window=window_size, center=True).mean()
        sns.lineplot(data=df, x='Steps', y='Critic Loss', color=colors[i], alpha=0.2,
                     label=agent_name_list[i] + ' Critic Loss')
        sns.lineplot(data=df, x='Steps', y='Smoothed Critic Loss', color=colors[i],
                     label=agent_name_list[i] + ' Smoothed Critic Loss')
    plt.title('Critic Training Loss Curve with Smoothing')
    plt.legend()

    plt.figure(figsize=(10, 6))
    for i in range(agent_num):
        df = pd.DataFrame({'Steps': steps, 'Reward': reward_list[i]})
        df['Smoothed Reward'] = df['Reward'].rolling(window=window_size, center=True).mean()
        sns.lineplot(data=df, x='Steps', y='Reward', color=colors[i], alpha=0.2,
                     label=agent_name_list[i] + ' Reward')
        sns.lineplot(data=df, x='Steps', y='Smoothed Reward', color=colors[i],
                     label=agent_name_list[i] + ' Smoothed Reward')
    plt.title('Reward Curve with Smoothing')
    plt.legend()

    plt.show()

def show_evaluate_result(dir_path, agent_name_list):

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
    labels = df_list[0]['Max SFC Length']
    colors = ['#925eb0', '#e29135', '#72b063', '#94c6cd', '#cc7c71']

    figure_path = 'save/result/plot/'

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
                    value + (0.01 if value >= 0 else -0.01) * max(abs(df[metric])),  # y
                    f'{value:.2f}',
                    ha='center',
                    va='bottom' if value >= 0 else 'top'
                )

        plt.xlabel('Max SFC Length', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(index + bar_width, labels=labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_path + metric + '.png', dpi=300)

    plt.show()

if __name__ == '__main__':
    agent_name_list = [
        'NCO',
        # 'DRLSFCP',
        # 'ActorEnhancedNCO',
        # 'CriticEnhancedNCO',
        # 'DDPG'
        ]
    show_train_result('save/result/train', agent_name_list)
    show_evaluate_result('save/result/evaluate', agent_name_list)