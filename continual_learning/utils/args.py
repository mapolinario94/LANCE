import argparse

__all__ = ["parse_args"]


def parse_args():
    parser = argparse.ArgumentParser(description='LANCE continual learning')
    # General
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--data-aug', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--dropout', action='store_true',
                        help='Use dropout')
    parser.add_argument('--threshold-conv', type=float, default=0.95,
                        help='LANCE energy threshold for convolutional layers')
    parser.add_argument('--threshold-cl', type=float, default=0.97,
                        help='LANCE energy threshold for Continual Learning')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--model', type=str, choices=['AlexNet', 'ResNet18'], default='AlexNet',
                        help='Choice of the model function')
    parser.add_argument('--dataset', type=str, choices=['SplitCIFAR100', 'MiniIMAGENET'],
                        default='SplitCIFAR100',
                        help='Choice of dataset')
    parser.add_argument('--n-experiences', type=int, default=10,
                        help='Number of tasks')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='Frequency to print results')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='~/Datasets',
                        help='Root directory where datasets are downloaded/cached')
    parser.add_argument('--save-path', type=str, default='./experiments_logs',
                        help='path to save the experiments logs')
    parser.add_argument('--experiment-name', type=str, default='LANCE_CL',
                        help='wandb project name')
    parser.add_argument('--patience', type=float, default=6,
                        help='Patience (# of epochs to wait) used in learning rate decay')
    parser.add_argument('--lr-threshold', type=float, default=1e-5,
                        help='Minimum learning rate')
    parser.add_argument('--lr-decay', type=float, default=2.0,
                        help='Learning rate decay factor')
    parser.add_argument('--num-free-dim', type=int, default=0,
                        help='Number of free dimensions (K)')
    parser.add_argument('--threshold-inc', type=float, default=0.0003,
                        help='Per-task increment added to the LANCE energy threshold')
    parser.add_argument('--imagenet-path', type=str, default='/local/a/imagenet/imagenet2012/',
                        help='Path to the ImageNet-2012 root used to build MiniImageNet (MiniIMAGENET only)')

    args = parser.parse_args()
    return args
