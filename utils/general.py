import os
import platform
import logging
import random
import pkg_resources as pkg
import numpy as np
import torch
import pickle


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required, but {name}{current} is currently installed'  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        logging.getLogger('logger').warning(s)
    return result


# def init_seeds(seed=0, benchmark=False, deterministic=False):
#     # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
#     torch.backends.cudnn.benchmark = benchmark #False:选择默认算法，降低性能，增加确定性; True:动态选择卷积算法，增加性能，具有不确定性。此选项控制卷积算法的选择，但选到的算法不一定是确定性的。
#     if deterministic and check_version(torch.__version__, '1.12.0'):
#         torch.use_deterministic_algorithms(True) #此选项控制选择的算法都是确定性的。
#         torch.backends.cudnn.deterministic = True #该选择只控制cudnn，而上面的控制包括cudnn在内的PyTorch其他算法。
#         os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#         os.environ['PYTHONHASHSEED'] = str(seed)
def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True) #此选项控制选择的算法都是确定性的。
        torch.backends.cudnn.deterministic = True #该选择只控制cudnn，而上面的控制包括cudnn在内的PyTorch其他算法。
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = torch.tensor(0, dtype=torch.float64)
        self._sum = torch.tensor(0, dtype=torch.float64)
        self._count = torch.tensor(0, dtype=torch.int64)

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n

    @property
    def val(self):
        return self._val.item()
    
    @property
    def sum(self):
        return self._sum.item()
    
    @property
    def count(self):
        return self._count.item()
    
    @property
    def avg(self):
        return (self._sum / self._count).item()


def partial_train_layers(model, partial_list, logger=None):
    """Train partial layers of a given model."""
    if logger is None:
        print_func = print
    else:
        print_func = logger.info
    train_list = []
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                train_list.append(name)
                break
    print_func(f'Partial train parameters: {train_list}')
    return model


def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content

def count_parameters(model):
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params