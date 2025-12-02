from runner import ExperimentRunner
from config import Config
import plot

if __name__ == '__main__':
    # get parsed arguments
    cfg = Config().get_config()

    # train and evaluate models
    runner = ExperimentRunner(cfg)
    if cfg.train:
        runner.train()
    if cfg.evaluate:
        runner.evaluate()

    # plot results
    plot.show_results(runner)