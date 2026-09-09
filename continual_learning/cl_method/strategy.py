import torch
import torch.optim as optim
from avalanche.evaluation.metrics import Forgetting
from torch.utils.data import DataLoader
from utils.metrics import *
import time
import wandb
import logging
from copy import deepcopy
from models.nn_models import BaseModel
from torch import nn
from models.nn_models import Linear_LANCE_CL
from models.nn_models import Conv2d_LANCE_CL

__all__ = ["MyStrategy", "average_forgetting_metric"]

class MyStrategy():

    def __init__(self, model:BaseModel, optimizer, criterion, epochs, batch_size, threshold=0.95, lr=0.1,
                 dropout=False, data_aug=False, transform_test=None, transform_train=None,
                 dataset_name=None, model_name=None, print_freq=50, patience=6, lr_decay=2,
                 lr_threshold=1e-5, threshold_inc=0.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.experience_id = 0
        self.threshold = threshold
        self.threshold_inc = threshold_inc
        self.forgetting_metric = Forgetting()
        self.data_aug = data_aug
        self.dropout = dropout
        self.transform_test = transform_test
        self.transform_train = transform_train
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.patience = patience
        self.lr_decay = lr_decay
        self.lr_threshold = lr_threshold

    def criterion(self, output, target):
        return self.loss_fn(output, target)

    def before_training(self, dataset, task_id):
        # From the second task on, record activations for one pass and initialize
        # the LANCE subspace before training; the first task trains uncompressed.
        if task_id > 0:
            dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,  drop_last=True)
            with torch.no_grad():
                for m in self.model.modules():
                    if isinstance(m, Conv2d_LANCE_CL) or isinstance(m, Linear_LANCE_CL):
                        setattr(m, "record_feature", True)
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                        for param in m.parameters():
                            param.requires_grad = False
                for idx, batch in enumerate(dl):
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.cuda()
                    _ = self.model(x, experience_id=task_id)  # forward triggers hooks
                for m in self.model.modules():
                    if isinstance(m, Conv2d_LANCE_CL) or isinstance(m, Linear_LANCE_CL):
                        setattr(m, "record_feature", False)
                        m.initialize_subspace()
                        setattr(m, "activate", True)
        else:
            for m in self.model.modules():
                if isinstance(m, Conv2d_LANCE_CL) or isinstance(m, Linear_LANCE_CL):
                    setattr(m, "activate", False)

    def _after_training_exp(self, dataset, task_id):
        # Record activations from a few batches to update each layer's protected
        # (null-space) subspace, which constrains the next task's subspace.
        with torch.no_grad():
            logging.info(f"Collecting activations after task {task_id} to update protected subspaces")
            dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,  drop_last=True)
            for m in self.model.modules():
                if isinstance(m, Conv2d_LANCE_CL) or isinstance(m, Linear_LANCE_CL):
                    setattr(m, "record_feature", True)
            for idx, batch in enumerate(dl):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.cuda()
                _ = self.model(x, experience_id=task_id)  # forward triggers hooks
                if idx == 10:
                    break
            for m in self.model.modules():
                if isinstance(m, Conv2d_LANCE_CL) or isinstance(m, Linear_LANCE_CL):
                    setattr(m, "record_feature", False)
                    m.initialize_protected_subspace()

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
            for batch_idx, (inputs, labels) in enumerate(train_data_loader):
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
            logging.info("Testing on first experience:")
            _ = self.testing(experience_zero, test_id=0)
            with torch.no_grad():
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = get_model(self.model)
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
            for batch_idx, (inputs, labels) in enumerate(eval_data_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                batch_size = inputs.size(0)
                outputs = self.model(inputs, experience_id=self.experience_id if test_id is None else test_id)
                loss = self.criterion(outputs, labels)
                losses.update(loss.item(), batch_size)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
            logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg}'.format(top1=top1, top5=top5, loss=losses))
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
            for batch_idx, (inputs, labels) in enumerate(eval_data_loader):
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


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        param_group['lr']=param_group['lr']/2