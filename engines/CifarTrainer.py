from engines._BaseTrainer import _BaseTrainer, eval_decorator

import os
from collections import OrderedDict

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from utils.torch_utils import torch_distributed_zero_first, de_parallel_ddp

class CifarTrainer(_BaseTrainer):
    def __init__(self, opt_file='configs/cifar.yaml', args=None):
        super().__init__(opt_file, args)

        #创建数据集
        with torch_distributed_zero_first(self.LOCAL_RANK):
            train_set = datasets.CIFAR10('./dataset/cifar10', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        self.train_loader = self.get_dataloader(
            train_set,
            self.opt.dataset.train.batch_size,
            train=True,
        )
        if self.RANK in {-1,0}:
            test_set = datasets.CIFAR10('./dataset/cifar10', train=False, transform=transforms.Compose([transforms.ToTensor(),]))
            self.test_loader = self.get_dataloader(
                test_set,
                self.opt.dataset.test.batch_size,
                train=False
            )

        #创建网络
        self.net = resnet18(num_classes=10)
        self.net.to(device=self.device)

        #创建optimizer, scheduler
        self.get_optimizer()
        self.get_scheduler()

        #尝试加载初始权重或checkpoint
        self.load_checkpoint()

        #设置模型并行
        self.set_parallel()

        #创建损失函数
        self.get_loss_function()

    def train(self):
        # 整个训练过程中中用到的计数变量初始化
        global_step, useless_epoch_count = 0, 0
        #开始迭代epochs轮训练
        for epoch in range(self.start_epoch, self.epochs):
            # 捕获KeyboardInterrupt异常以便提前终止训练进入评测阶段
            try:
                #每轮迭代开始时的设置
                self.net.train()
                if self.RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                
                # 加载训练数据batch
                pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}', bar_format='{l_bar}{bar:10}{r_bar}') \
                       if self.RANK in {-1, 0} else self.train_loader
                for data, target in pbar:
                    global_step += 1
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.net(data)

                    loss = self.loss_function(output, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if self.RANK in {-1, 0}:
                        postfix = OrderedDict()
                        postfix['loss'] = loss.item()
                        pbar.set_postfix(postfix)

                # 一轮epoch结束后的处理
                testloss, acc = self.evaluate()
                self.loggers.info(f'Val epoch {epoch+1} accuracy: {acc}, loss: {testloss}')

                self.scheduler.step() # scheduler处理要放在得到val分数之后

                if self.check_update_val_score(acc): 
                    self.save_checkpoint(epoch='best')
                self.save_checkpoint(epoch=epoch+1) # 保存每轮的save_checkpoint一定要在更新best_val_score后面

                # 检测是否达到Early Stopping
                if self.check_early_stopping():
                    break

            except KeyboardInterrupt:
                self.loggers.info('Receive KeyboardInterrupt, stop training...')
                if 'pbar' in dir():
                    pbar.close()
                break
        
        # 所有轮次训练结束后需要做的事，如评测模型在测试集性能
        pass

    @eval_decorator()
    def evaluate(self, type='val', final=False):
        model = de_parallel_ddp(self.net)
        model.eval()

        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)

        self.net.train()
        return test_loss, 100. * correct / len(self.test_loader.dataset)
