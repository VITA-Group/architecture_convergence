import sys
import os, torch, random, PIL, copy, numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger(object):

    def __init__(self, log_dir, seed, create_model_dir=False):
        """Create a summary writer logging to log_dir."""
        self.seed      = int(seed)
        self.log_dir   = Path(log_dir)
        self.model_dir = Path(log_dir) / 'model'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_dir = self.log_dir
        self.logger_path = self.log_dir / 'seed-{:}.log'.format(self.seed)
        self.logger_file = open(self.logger_path, 'w')

        self.tensorboard_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.tensorboard_dir))

    def __repr__(self):
        return ('{name}(dir={log_dir}, writer={writer})'.format(name=self.__class__.__name__, **self.__dict__))

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string); sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write('{:}\n'.format(string))
            self.logger_file.flush()


def prepare_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_logger(xargs=None):
    args = copy.deepcopy(xargs)
    save_dir = args.save_dir if hasattr(args, "save_dir") else "./save"
    seed = args.seed if hasattr(args, "seed") else 0
    logger = Logger(save_dir, seed)
    logger.log('Main Function with logger : {:}'.format(logger))
    logger.log('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.log('{:16} : {:}'.format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
    logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log("CUDA_VISIBLE_DEVICES : {:}".format(os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))
    return logger