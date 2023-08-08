import os
import gc
import sys
import time
import yaml
import shutil
import logging
import platform
import tempfile
from abc import ABC, abstractmethod
from logging import Formatter, StreamHandler, Filter #, FileHandler

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from concurrent_log_handler import ConcurrentRotatingFileHandler # pip install concurrent-log-handler

from models import load_param
from utils.easydict import EasyDict
from utils.general import init_seeds, emojis
from utils.gitignore_parser import ignorer_from_gitignore
from utils.torch_utils import get_local_rank, get_rank, get_world_size, seed_worker
from utils.torch_utils import de_parallel, select_device, smart_inference_mode


class LevelFilter(Filter):
    def __init__(self, name: str='', level: int=logging.INFO) -> None:
        super().__init__(name=name)
        self.level = level

    def filter(self, record):
        if record.levelno < self.level:
            return False
        return True


class Loggers():
    r"""
    控制只在master进程输入info信息以及打印tensorboard日志, 而在其他节点只打印error信息。
    使用了ConcurrentRotatingFileHandler保证多进程写日志文件的安全性。
    """
    def __init__(self, rank, log_dir, tensorboard_dir):
        self.RANK = rank
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG if rank in {-1,0} else logging.ERROR)
        self.logger.propagate = False # 防止向上传播导致root logger也打印log
        stdf = StreamHandler(sys.stdout)
        stdf.addFilter(LevelFilter('std_filter', logging.INFO))
        stdf.setFormatter(Formatter('[%(levelname)s]: %(message)s'))
        self.logger.addHandler(stdf)
        filef = ConcurrentRotatingFileHandler(
            log_dir, 'a', 512*1024, 5,
            lock_file_directory=os.path.join(tempfile.gettempdir(), f'{os.getlogin()}#{os.path.basename(os.path.dirname(log_dir))}')
        )
        filef.addFilter(LevelFilter('file_filter', logging.INFO))
        filef.setFormatter(Formatter('[%(levelname)s %(asctime)s] %(message)s', "%Y%m%d-%H:%M:%S"))
        self.logger.addHandler(filef)

        if platform.system() == 'Windows':
            for fn in self.logger.info, self.logger.warning, self.logger.error:
                setattr(self.logger, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging

        # tensorboard等只在RANK 0节点创建
        if rank in {-1,0}:
            self.tbwriter = SummaryWriter(log_dir=tensorboard_dir)

    def __del__(self):
        if self.RANK in {-1,0}:
            self.tbwriter.close()

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def add_scalar(self, tag, scalar_value, global_step=None):
        if self.RANK not in {-1,0}: return
        self.tbwriter.add_scalar(tag, scalar_value, global_step)

    def add_histogram(self, tag, values, global_step=None):
        if self.RANK not in {-1,0}: return
        self.tbwriter.add_histogram(tag, values, global_step)

    def add_images(self, tag, image_tensor, global_step=None, dataformats='NCHW'):
        if self.RANK not in {-1,0}: return
        self.tbwriter.add_images(tag, image_tensor, global_step, dataformats=dataformats)


def create_code_checkpoint(checkpoint_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        codedir = os.path.join(tmpdir, 'code')
        shutil.copytree('./', codedir, ignore=ignorer_from_gitignore('.gitignore'))
        shutil.make_archive(os.path.join(checkpoint_dir, 'code'), 'zip', codedir)


def merge_args_opt(opt, args):
    if args is None:
        return
    if hasattr(args, 'device') and args.device is not None:
        opt.train.device = args.device


class _BaseTrainer(ABC):
    """net.to(device)->optimizer->scheduler->ModelEMA->load_checkpoint->DP/DDP"""
    def __init__(self, opt_file, args=None):
        # 加载DDP相关环境变量
        self.LOCAL_RANK = get_local_rank()
        self.RANK = get_rank()
        self.WORLD_SIZE = get_world_size()

        #根据控制台修改参数文件
        if hasattr(args, 'config') and args.config is not None:
            opt_file = args.config

        #加载参数，并根据控制台传参args修改
        with open(opt_file) as f:
            opt = EasyDict(yaml.full_load(f))
        merge_args_opt(opt, args)
        self.opt = opt

        #设置device, init DDP如果需要, 设置随机种子
        self.device = select_device(opt.train.device, opt.dataset.batch_size, self.LOCAL_RANK, self.WORLD_SIZE)
        init_seeds(seed=opt.random.seed+1+self.RANK, deterministic=opt.random.deterministic)

        #判断是恢复实验还是新实验
        if not opt.train.get('resume_training', False):
            self.resume_training = False
            self.train_time_str = time.strftime("%m-%d_%H:%M:%S", time.localtime())
        else: #改train.load_model, 改info
            self.resume_training = True
            resume_name = os.path.basename(os.path.dirname(opt.train.resume_training)).split('#')
            assert len(resume_name) == 2, f'len(resume_name) should be 2 splited by "#", but got {resume_name}'
            self.train_time_str, opt.info = resume_name
            opt.train.load_model = opt.train.resume_training

        #创建日志打印器和检查点保存目录
        self.loggers = Loggers(rank = self.RANK,
            log_dir = os.path.join(opt.expe_root, 'logs', opt.engine_name, f'{self.train_time_str}#{opt.info}.txt'),
            tensorboard_dir = os.path.join(opt.expe_root, 'runs', f'{opt.engine_name}#{self.train_time_str}')
        )
        self.checkpoint_dir = os.path.join(opt.expe_root, 'checkpoints', opt.engine_name, f'{self.train_time_str}#{opt.info}/')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        #日志里打印opt文件内容
        self.loggers.info(f'{opt_file} START************************\n'
        f'{yaml.dump(opt.convert2dict(), sort_keys=False, allow_unicode=True)}\n'
        f'************************{opt_file} END**************************\n')

        #保存代码到检查点保存目录
        if self.RANK in {-1,0}:
            create_code_checkpoint(self.checkpoint_dir)

        #训练相关参数初始化设置
        self.start_epoch = 0
        self.epochs = opt.train.epochs
        self.save_cp = opt.train.save_cp
        self.early_stopping = opt.train.early_stopping
        self.useless_epoch_count = 0
        if opt.val.mode == 'max':
            self.best_val_score = float('-inf')
        elif opt.val.mode == 'min':
            self.best_val_score = float('inf')
        else:
            raise ValueError(f'opt.val.mode should be either "max" or "min", but got {opt.val.mode}')

    def get_dataloader(self, dataset, batch_size, train: bool, drop_last=False, shuffle=None, workers=8, collate_fn=None, **kwargs):
        batch_size = batch_size // self.WORLD_SIZE if train and self.RANK != -1 else batch_size
        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()  # number of CUDA devices
        nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
        if shuffle is None:
            if train:
                shuffle = True
            else:
                shuffle = False
        sampler = DistributedSampler(dataset, drop_last=drop_last, shuffle=shuffle) if train and self.RANK != -1 else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=nw,
            pin_memory=str(os.getenv('PIN_MEMORY', True)).lower() == 'true',
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            **kwargs
        )

    def get_optimizer(self):
        assert hasattr(self, 'net'), 'get_optimizer should be called after creating network (self.net).'
        assert hasattr(self.opt, 'optimizer'), 'The config yaml file should contain the setting of "optimizer"'
        name = self.opt.optimizer.name
        kwargs = self.opt.optimizer.get('kwargs', {})
        if hasattr(torch.optim, name):
            self.optimizer = getattr(torch.optim, name)(self.net.parameters(), **kwargs)
            return True
        elif name.startswith('Ranger'):
            import optim.ranger as ranger
            self.optimizer = getattr(ranger, name)(self.net.parameters(), **kwargs)
            return True
        elif name == 'Lion':
            from optim.lion import Lion
            self.optimizer = Lion(self.net.parameters(), **kwargs)
            return True
        else:
            return False
        
    def get_scheduler(self):
        assert hasattr(self, 'optimizer'), 'get_scheduler should be called after creating an optimizer (self.optimizer).'
        assert hasattr(self.opt, 'scheduler'), 'The config yaml file should contain the setting of "scheduler"'
        name = self.opt.scheduler.name
        kwargs = self.opt.scheduler.get('kwargs', {})
        if hasattr(torch.optim.lr_scheduler, name):
            self.scheduler = getattr(torch.optim.lr_scheduler, name)(self.optimizer, **kwargs)
            return True
        elif name == 'CosineAnnealingWithWarmUpLR':
            from optim.lr_scheduler import CosineAnnealingWithWarmUpLR
            self.scheduler = CosineAnnealingWithWarmUpLR(
                self.optimizer,
                T_total = self.epochs if self.epochs != 0 else 1,
                eta_min = self.opt.optimizer.kwargs.lr / kwargs.get('eta_min_ratio', 100),
                warm_up_lr = self.opt.optimizer.kwargs.lr / kwargs.get('warm_up_lr_ratio', 100),
                warm_up_step = kwargs.get('warm_up_step', self.epochs // 10),
                verbose = kwargs.get('verbose', False),
            )
            return True
        else:
            return False
        
    def set_parallel(self):
        """this func should be called after to(device), create optimizer, scheduler and load checkpoint"""
        assert hasattr(self, 'net'), 'set_parallel should be called after creating network (self.net).'
        assert hasattr(self, 'optimizer'), 'set_parallel should be called after creating optimizer (self.optimizer).'
        if self.device.type != 'cpu' and self.RANK == -1 and torch.cuda.device_count() > 1:
            self.net = DataParallel(self.net)
            self.loggers.info('⚠️  Using multiple GPU training based on DP.')
        elif self.device.type != 'cpu' and self.RANK != -1:
            if hasattr(self.opt, 'model') and self.opt.model.get('sync_bn'):
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net).to(self.device)
                self.loggers.info('⚠️  Using SyncBatchNorm()')
            self.net = DistributedDataParallel(self.net, device_ids=[self.LOCAL_RANK], output_device=self.LOCAL_RANK,
                                               find_unused_parameters=self.opt.model.find_unused_parameters)
            self.loggers.info('⚠️  Using multiple GPU training based on DDP.')

    def get_loss_function(self):
        assert hasattr(self.opt, 'loss'), 'The config yaml file should contain the setting of "loss"'
        name = self.opt.loss.name
        kwargs = self.opt.loss.get('kwargs', {})
        if hasattr(torch.nn, name):
            self.loss_function = getattr(torch.nn, name)(**kwargs)
            return True
        else:
            return False

    def load_checkpoint(self, load_dir=None, load_param_func=load_param):
        r"""Load checkpoint. In the :func:`__init__` of :class:`Trainer`, this function should be called
        after creating net, optimizer, and scheduler.

        Args:
            load_dir: the checkpoint dir. If `None`, it will be assigned as `self.opt.train.load_model`
        """
        if load_dir is None:
            load_dir = self.opt.train.load_model
        if not load_dir:
            return
        loaded = torch.load(load_dir, map_location='cpu')
        network = de_parallel(self.net)
        if 'weight' in loaded and 'optimizer' in loaded and 'scheduler' in loaded and 'epoch' in loaded and 'best_val_score' in loaded:
            if self.resume_training:
                self.start_epoch = loaded['epoch'] + 1
                self.best_val_score = loaded['best_val_score']
                try:
                    network.load_state_dict(loaded['weight'])
                except RuntimeError:
                    load_param_func(network, loaded['weight'], logger=self.loggers)
                self.optimizer.load_state_dict(loaded['optimizer'])
                self.scheduler.load_state_dict(loaded['scheduler'])
                self.loggers.info(f'Resume training from checkpoint {load_dir}')
            else:
                try:
                    network.load_state_dict(loaded['weight'])
                except RuntimeError:
                    load_param_func(network, loaded['weight'], logger=self.loggers)
                self.loggers.info(f'Only net weight loaded from checkpoint {load_dir}')
        else:
            if self.resume_training:
                self.loggers.warning('Try to resume training from a pure model weight. '
                                     'Please use checkpoint to resume training.')
            try:
                network.load_state_dict(loaded)
            except RuntimeError:
                load_param_func(network, loaded, logger=self.loggers)
            self.loggers.info(f'Net weight loaded from {load_dir}')
        if self.RANK != -1:
            dist.barrier()

    def save_checkpoint(self, epoch='best'):
        r"""Save checkpoint.

        Args:
            epoch: current epoch during training or "best" representing to save the best weight only.
        """
        if self.RANK not in {-1, 0}:
            return
        network = de_parallel(self.net)
        if isinstance(epoch, int):
            save_dict = {}
            save_dict['weight'] = network.state_dict()
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
            save_dict['epoch'] = epoch
            save_dict['best_val_score'] = self.best_val_score
            file_name = f'Checkpoint_epoch{epoch+1:04d}.pth' if self.save_cp else f'Checkpoint_last.pth'
            torch.save(save_dict, os.path.join(self.checkpoint_dir, file_name))
            self.loggers.info(f'{file_name} saved !')
        elif epoch == 'best':
            torch.save(network.state_dict(), os.path.join(self.checkpoint_dir, 'Net_best.pth'))
            self.loggers.info('Best model saved !')
        else:
            raise ValueError(f'epoch should be either int or "best", but got {epoch}')

    def check_update_val_score(self, val_socre):
        if self.opt.val.mode == 'max':
            best_flag = val_socre > self.best_val_score
        elif self.opt.val.mode == 'min':
            best_flag = val_socre < self.best_val_score
        else:
            raise ValueError(f'self.opt.val.mode should be either "max" or "min", but got {self.opt.val.mode}')
        if best_flag:
            self.best_val_score = val_socre
            self.useless_epoch_count = 0
        else:
            self.useless_epoch_count += 1
        return best_flag

    def check_early_stopping(self):
        if self.early_stopping and self.useless_epoch_count == self.early_stopping:
            self.loggers.info(f'There are {self.useless_epoch_count} useless epochs! Early Stop Training!')
            return True
        return False

    @abstractmethod
    def train(self):
        pass

    def __del__(self):
        gc.collect()
        self.loggers.info(self.opt.info)


def eval_decorator(rank=get_rank()):
    """仅在master上执行evaluate函数对模型进行评估,并将模型评估结果同步到其他节点中。"""
    def decorator(eval_fn):
        def wrapper(*args, **kwargs):
            eval_res = smart_inference_mode()(eval_fn)(*args, **kwargs) if rank in {-1,0} else None
            if rank != -1:
                bc_list = [eval_res]
                dist.broadcast_object_list(bc_list, 0)
                if rank != 0:
                    eval_res = bc_list[0]
            return eval_res
        return wrapper
    return decorator
