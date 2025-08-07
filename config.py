import argparse

# default parameters
BATCH_SIZE = 20
ITERATION = 2000
EPISODE = 10    # fill episode * batch_size data into replay buffer

NUM_VNF_TYPES = 8   # number of VNF types

MIN_SFC_LENGTH = 4
MAX_SFC_LENGTH = 10 # max sfc length agent can finish placement, different from SFCBatchGenerator max_sfc_length

# convert string to boolean value
def str2bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, int):
        return False if s == 0 else True
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if s.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported input type. Boolean value expected.')

parser = argparse.ArgumentParser(description='Configuration file')

train_arg = parser.add_argument_group('train')
train_arg.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size of training')
train_arg.add_argument('--episode', type=int, default=EPISODE, help='number of epochs')

model_arg = parser.add_argument_group('model')
model_arg.add_argument('--model', type=str, default='DDPG', help='agent model')
model_arg.add_argument('--save_model', type=str2bool, default=True, help='save model')
model_arg.add_argument('--load_model', type=str2bool, default=False, help='load model')
model_arg.add_argument('--save_to', type=str, default='save/model', help='directory to save model')
model_arg.add_argument('--load_from', type=str, default='save/model', help='directory to load model')

# get all the parameters
def get_config():
    parsed, unparsed = parser.parse_known_args()
    return parsed, unparsed

# show all the parameters
def show_config():
    parsed_args, unparsed_args = get_config()
    for key, value in vars(parsed_args).items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    config, _ = get_config()
    show_config()