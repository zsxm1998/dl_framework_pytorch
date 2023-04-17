import os
import platform
import logging
import random
import pkg_resources as pkg
import numpy as np
import torch


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