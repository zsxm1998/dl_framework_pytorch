import os
import argparse
import torch
import multiprocess
from typing import Callable, Iterable
import pynvml
import time


MEMORY_NUM = 4


class TaskRunner:
    """
        对于给定的任务, 使用多线程执行命令
    
        可能需要的输入:
            worker function
            arguments iterator
            Pool arguments
    """
    def __init__(self, pool_size=16) -> None:
        self.ctx = multiprocess.get_context("spawn")
        self.pool = self.ctx.Pool(pool_size)

    @staticmethod
    def __wrapper(func: Callable):
        def _wfunc(zipped_kwargs):
            return func(**zipped_kwargs)
        return _wfunc

    def run(self, func: Callable, args_iter: Iterable):
        results = list(self.pool.imap(self.__wrapper(func), args_iter))
        self.pool.terminate()
        return results


def grab_gpu(device_id, nc):
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'
    device = torch.device(f'cuda:{device_id}')
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    pid = os.getpid()
    tensor_list = []
    compute_flag = False
    while True:
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = meminfo.free/1024**2
        if free_memory > MEMORY_NUM * 100:
            print(f'{time.strftime("%m-%d_%H:%M:%S", time.localtime())}: PID {pid} GPU {device_id} get {free_memory}MB free memory, grabing it.')
            while True:
                try:
                    tensor_list.append( torch.zeros(5120, 5120, MEMORY_NUM, dtype=torch.float32, device=device) )
                except RuntimeError:
                    break
        elif not nc:
            proinfos = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
            flag = False
            usedGpuMemory = -1
            for pinfo in proinfos:
                if pinfo.pid == pid and pinfo.usedGpuMemory / meminfo.total > 0.9:
                    usedGpuMemory = pinfo.usedGpuMemory
                    flag = True
                    break
            if flag and len(tensor_list) > 0:
                if not compute_flag:
                    print(f'{time.strftime("%m-%d_%H:%M:%S", time.localtime())}: PID {pid} GPU {device_id} {usedGpuMemory/1024**2}/{meminfo.total/1024**2}={usedGpuMemory/meminfo.total}, start computing.')
                    compute_flag = True
                for i, tensor in enumerate(tensor_list):
                    for _ in range(1000):
                        tensor += 1e-15
                    time.sleep(1)
        time.sleep(1)


def get_args():
    parser = argparse.ArgumentParser(description='Hold GPU memory.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', '-d', type=str, default=None, help='GPU index')
    parser.add_argument('--notcompute', '-nc', action='store_true', help='not use GPU to compute')

    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()
    if opt.device is None:
        device_ids = list(range(torch.cuda.device_count()))
    else:
        device_ids = [int(i) for i in opt.device.replace(' ', '').split(',')]

    kwargs = [{'device_id': i, 'nc': opt.notcompute} for i in device_ids]
    TaskRunner().run(grab_gpu, kwargs)
