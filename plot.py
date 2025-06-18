import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def show_train_result(dir_path):
    all_files = os.listdir(dir_path)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    agent_name_list = [f.replace('.csv', '') for f in csv_files]
    agent_num = len(agent_name_list)

    df_list = []
    reward_list = []
    actor_loss_list = []
    critic_loss_list = []

    for csv_file in csv_files:
        csv_file_path = dir_path + '/' + csv_file
        df = pd.read_csv(csv_file_path)
        df_list.append(df)
        reward_list.append(df['Reward'])
        actor_loss_list.append(df['Actor Loss'])
        critic_loss_list.append(df['Critic Loss'])

    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.title('Reward')
    for i in range(agent_num):
        plt.plot(reward_list[i], label=agent_name_list[i] + ' Reward')
    # plt.ylim((13,18))
    plt.legend()

    plt.figure()
    # plt.subplot(1, 3, 2)
    plt.title('Actor Loss')
    for i in range(agent_num):
        plt.plot(actor_loss_list[i], label=agent_name_list[i] + ' Actor Loss')
    plt.legend()

    plt.figure()
    # plt.subplot(1, 3, 3)
    plt.title('Critic Loss')
    for i in range(agent_num):
        plt.plot(critic_loss_list[i], label=agent_name_list[i] + ' Critic Loss')
    plt.legend()

    # figure_name = 'result.png'
    # plt.savefig(figure_name, dpi=300)
    plt.show()

def show_evaluate_result(dir_path):
    all_files = os.listdir(dir_path)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    agent_name_list = [f.replace('.csv', '') for f in csv_files]

    # load csv files
    df_list = []
    for csv_file in csv_files:
        csv_file_path = dir_path + '/' + csv_file
        df = pd.read_csv(csv_file_path)
        df_list.append(df)

    # row number in csv files
    index_num = len(df_list[0]) if df_list else 0

    bar_width = 0.25
    index = np.arange(index_num)    # bar location
    labels = df_list[0]['Max SFC Length']
    colors = ['#72b063', '#e29135', '#94c6cd']

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
                    f'{value:.1f}',
                    ha='center',
                    va='bottom' if value >= 0 else 'top'
                )

        plt.xlabel('Max SFC Length', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(index + bar_width, labels=labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    show_train_result('save/result/train')
    # show_evaluate_result('save/result/evaluate')