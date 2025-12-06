import argparse

class Config():
    def __init__(self):
        self.parsed = None

    def _parse_args(self):
        parser = argparse.ArgumentParser(description='Configuration file')

        parser.add_argument('--batch_size', type=int, default=20, help='batch size of training')
        parser.add_argument('--iteration', type=int, default=10, help='number of iterations')
        parser.add_argument('--episode', type=int, default=10, help='number of epochs trained in an iteration')

        parser.add_argument('--min_sfc_length', type=int, default=6, help='min sfc length')
        parser.add_argument('--max_sfc_length', type=int, default=12, help='max sfc length')
        parser.add_argument('--num_vnf_types', type=int, default=8, help='number of VNF types, 1-8')

        parser.add_argument('--graph', type=str, default='Chinanet', help='graph name')
        parser.add_argument('--model', type=str, default='all', help='agent model(s) to be trained (all / NCO / EnhancedNCO / DRLSFCP / PPO / ACED)')
        parser.add_argument('--train', type=str2bool, default=True, help='train model')
        parser.add_argument('--evaluate', type=str2bool, default=True, help='evaluate model')
        parser.add_argument('--save_model', type=str2bool, default=True, help='save model')

        parser.add_argument('--model_path', type=str, default='save/model/', help='directory to save model')
        parser.add_argument('--graph_path', type=str, default='graph/', help='directory of graph files')

        self.parsed = parser.parse_args()

    def _set_path(self):
        self.parsed.agent_path = f'save/model/{self.parsed.graph}/'
        self.parsed.result_path = f'save/result/{self.parsed.graph}/'

    def _set_batch_size_list(self):
        if self.parsed.graph == 'Cogentco':
            self.parsed.batch_size_list = [60, 70, 80, 90, 100]
        elif self.parsed.graph == 'Chinanet':
            self.parsed.batch_size_list = [15, 20, 25, 30, 35]
        else:
            raise ValueError('Invalid graph name.')

    def get_config(self):
        self._parse_args()
        self._set_path()
        self._set_batch_size_list()
        return self.parsed

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
