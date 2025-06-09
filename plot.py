import matplotlib.pyplot as plt
import os
import pandas as pd

def show_result(dir_path):
    all_files = os.listdir(dir_path)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    agent_name = [f.replace('.csv', '') for f in csv_files]
    agent_num = len(agent_name)

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
        plt.plot(reward_list[i], label=agent_name[i] + ' Reward')
    plt.legend()

    plt.figure()
    # plt.subplot(1, 3, 2)
    plt.title('Actor Loss')
    for i in range(agent_num):
        plt.plot(actor_loss_list[i], label=agent_name[i] + ' Actor Loss')
    plt.legend()

    plt.figure()
    # plt.subplot(1, 3, 3)
    plt.title('Critic Loss')
    for i in range(agent_num):
        plt.plot(critic_loss_list[i], label=agent_name[i] + ' Critic Loss')
    plt.legend()

    figure_name = 'result.png'
    plt.savefig(figure_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    show_result('save/result')