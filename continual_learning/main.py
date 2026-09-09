import torch
from torch import nn
import torch.optim as optim
from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100
from avalanche.benchmarks.datasets import MiniImageNetDataset
from avalanche.benchmarks import class_incremental_benchmark, nc_benchmark
from torch.utils.data import DataLoader
from utils.metrics import *
import time
import numpy as np
import wandb
import logging
import warnings
from copy import deepcopy
from models.nn_models import AlexNet, BaseModel, ResNet18
from torchvision import transforms
from cl_method import strategy
from utils import parse_args
import os
from models.nn_models import Linear_LANCE_CL
from models.nn_models import Conv2d_LANCE_CL
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)


class MyStrategy(strategy.MyStrategy):

    def train(self, experience, test_experience=None, experience_zero=None):
        threshold = self.threshold + self.experience_id * self.threshold_inc
        for m in self.model.modules():
            if isinstance(m, Conv2d_LANCE_CL) or isinstance(m, Linear_LANCE_CL):
                m.threshold_cl = threshold
        if hasattr(experience, "dataset"):
            train_dataset = experience.dataset
        else:
            train_dataset = experience

        train_data_loader = DataLoader(
            train_dataset, num_workers=4, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0, weight_decay=0)

        logging.info("#" * 20)
        logging.info(f"Training on task: {self.experience_id}")
        logging.info("#" * 20)
        best_model = get_model(self.model)
        best_loss = 1000
        best_acc = 0
        flag = False
        lr_updated = self.lr

        # Initialize LANCE subspaces from the current task's activations
        self.before_training(train_dataset, task_id=self.experience_id)

        for epoch in range(self.epochs):
            if self.dropout:
                self.model.train()
            else:
                self.model.eval()
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(train_data_loader),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            end = time.time()
            for batch_idx, (inputs, labels, _) in enumerate(train_data_loader):
                data_time.update(time.time() - end)
                batch_size = inputs.size(0)
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs, labels, self.experience_id)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                losses.update(loss.item(), batch_size)
                batch_time.update(time.time() - end)
                end = time.time()
                if batch_idx % self.print_freq == (self.print_freq - 1):
                    progress.display(batch_idx, log=True)
            logging.info(f"Testing on current experience {self.experience_id} at epoch {epoch}:")
            valid_acc = self.testing(test_experience)
            with torch.no_grad():
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = get_model(self.model)
                    logging.info(f"Saving best model with acc: {best_acc} at epoch {epoch}")
                    patience = 0
                else:
                    patience += 1
                    if patience > self.patience:
                        adjust_learning_rate(optimizer, epoch, None)
                        lr_updated /= self.lr_decay
                        patience = 0
                    elif lr_updated < self.lr_threshold:
                        break
        set_model_(self.model, best_model)
        self._after_training_exp(train_dataset, task_id=self.experience_id)
        self.experience_id += 1

    def testing(self, experience, test_id=None):
        if hasattr(experience, "dataset"):
            eval_dataset = experience.dataset
        else:
            eval_dataset = experience
        eval_data_loader = DataLoader(
            eval_dataset, num_workers=4, batch_size=512
        )
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (inputs, labels, _) in enumerate(eval_data_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                batch_size = inputs.size(0)
                outputs = self.model(inputs, experience_id=self.experience_id if test_id is None else test_id)
                loss = self.criterion(outputs, labels)
                losses.update(loss.item(), batch_size)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
            logging.info(
                ' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg}'.format(top1=top1, top5=top5,
                                                                                               loss=losses))
        return top1.avg.item()

    def eval(self, experience, exp_id, exp_test_id):
        if hasattr(experience, "dataset"):
            eval_dataset = experience.dataset
        else:
            eval_dataset = experience
        eval_data_loader = DataLoader(
            eval_dataset, num_workers=4, batch_size=512
        )
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (inputs, labels, _) in enumerate(eval_data_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                batch_size = inputs.size(0)
                outputs = self.model(inputs, experience_id=exp_test_id)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
            logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        self.forgetting_metric.update(k=f"Experience {exp_test_id}", v=top1.avg.item(),
                                      initial=exp_test_id == (self.experience_id - 1))
        wandb.log({f"acc_exp_{exp_test_id}": top1.avg.item()}, step=exp_id)
        return top1.avg.item()


def average_forgetting_metric(forgetting_dict):
    avg_forgetting = 0
    num_exp = 0
    for key in forgetting_dict.keys():
        num_exp += 1
        avg_forgetting += forgetting_dict[key]
    if num_exp == 0:
        return 0
    else:
        avg_forgetting /= num_exp
        return avg_forgetting


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        param_group['lr']=param_group['lr']/2


if __name__ == '__main__':
    args = parse_args()
    now = datetime.now()
    # Format as YYYYMMDD_HHMMSS
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.experiment_name}_{timestamp}"
    args.save_path = args.save_path + "/" + folder_name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Log initialization
    log_path = args.save_path + "/log.log"
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=log_path,
                        filemode="a")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)
    logging.info('=> Everything will be saved to {}'.format(log_path))

    experiment_name = args.experiment_name
    with wandb.init(project=experiment_name, config=vars(args)):
        n_experiences = args.n_experiences
        epochs = args.epochs
        lr = args.lr
        threshold_conv = args.threshold_conv
        threshold_cl = args.threshold_cl
        batch_size = args.batch_size
        dataset = args.dataset
        model = args.model
        dropout = args.dropout
        data_aug = args.data_aug
        print_freq = args.print_freq

        if dataset == "MiniIMAGENET":
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
            if data_aug:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(84, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.RandomErasing(p=0.2)
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            imagenet_dataset = MiniImageNetDataset(args.imagenet_path)
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                imagenet_dataset, [2375/3000, 125/3000, 500/3000], generator=generator)
            benchmark = nc_benchmark(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_experiences=n_experiences,
                task_labels=False,
                seed=1234,
                fixed_class_order=None,
                shuffle=True,
                per_exp_classes=None,
                class_ids_from_zero_in_each_exp=True,
                class_ids_from_zero_from_first_exp=False,
                train_transform=transform_train,
                eval_transform=transform_test,
            )
            benchmark_val = nc_benchmark(
                train_dataset=train_dataset,
                test_dataset=val_dataset,
                n_experiences=n_experiences,
                task_labels=False,
                seed=1234,
                fixed_class_order=None,
                shuffle=True,
                per_exp_classes=None,
                class_ids_from_zero_in_each_exp=True,
                class_ids_from_zero_from_first_exp=False,
                train_transform=transform_train,
                eval_transform=transform_test,
            )

        else:
            raise NotImplementedError(f"{dataset} is not available")

        if model == "ResNet18":
            model_ = ResNet18(n_experiences=n_experiences, n_classes=int(100//n_experiences),
                              threshold=threshold_conv, threshold_cl=threshold_cl,
                              num_free_dim=args.num_free_dim).cuda()
        else:
            raise NotImplementedError(f"{model} is not available")

        cl_strategy = MyStrategy(model_, None, nn.CrossEntropyLoss(), epochs=epochs, batch_size=batch_size,
                                 threshold=threshold_cl,
                                 dropout=dropout, data_aug=data_aug,
                                 transform_test=transform_test, transform_train=transform_train, dataset_name=dataset,
                                 model_name=model, print_freq=print_freq, threshold_inc=args.threshold_inc)

        accuracy_history = []
        accuracy_list = []
        logging.info('Starting experiment...')
        for exp_id, experience in enumerate(benchmark.train_stream):
            logging.info("Start of experience: {0}".format( experience.current_experience))
            cl_strategy.train(experience, benchmark_val.test_stream[exp_id], experience_zero=benchmark.test_stream[0])
            logging.info('Training completed')

            for exp_test_id in range(n_experiences):
                logging.info(f'Testing experience: {exp_test_id}')
                if exp_test_id <= exp_id:
                    acc = cl_strategy.eval(benchmark.test_stream[exp_test_id], exp_id, exp_test_id)
                    accuracy_list.append(acc)
            logging.info(cl_strategy.forgetting_metric.result())
            fogetting_dict = cl_strategy.forgetting_metric.result()
            for key in fogetting_dict.keys():
                wandb.log({"Forgetting_"+key: fogetting_dict[key]}, step=exp_id)
            accuracy_history.append(accuracy_list)
            accuracy_list = []
            avg_forgetting = average_forgetting_metric(fogetting_dict)
            logging.info(f"Average Forgetting: {avg_forgetting}")
            logging.info(f"Average Accuracy: {np.mean(np.array(accuracy_history[-1]))}")
            wandb.log({"avg_forgetting": avg_forgetting,
                       "avg_accuracy": np.mean(np.array(accuracy_history[-1]))}, step=exp_id)

        avg_forgetting = average_forgetting_metric(fogetting_dict)
        wandb.run.summary.update({"final_avg_forgetting": avg_forgetting,
                                  "final_avg_accuracy": float(np.mean(np.array(accuracy_history[-1])))})