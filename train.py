import argparse

from torch.distributed.elastic.multiprocessing.errors import record

from engines import *


def get_args():
    parser = argparse.ArgumentParser(description='Train the Network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', '-d', type=str, help='Set training devices.')
    parser.add_argument('--config', '-c', type=str, help='Path to config file')
    parser.add_argument('--cifar', action='store_true', help='CifarTrainer')

    return parser.parse_args()


@record
def main():
    args = get_args()
    if False:
        pass
    elif args.cifar:
        trainer = CifarTrainer(args=args)
    else:
        raise ValueError('需要传参选择要训练的模型！')

    trainer.train()
    del trainer


if __name__ == '__main__':
    main()