import torch
from torch import nn
import numpy as np
import wandb
import logging
import warnings
from models.nn_models import AlexNet
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from cl_method import strategy
from utils import parse_args
import os
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)

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

        from utils import cifar100 as cf100
        data, taskcla, inputsize = cf100.get(args.data_dir, pc_valid=0.05)

        if model == "AlexNet":
            model_ = AlexNet(n_experiences=n_experiences, n_classes=int(100//n_experiences),
                             threshold=threshold_conv, threshold_cl=threshold_cl,
                             num_free_dim=args.num_free_dim).cuda()
        else:
            raise NotImplementedError(f"{model} is not available")

        cl_strategy = strategy.MyStrategy(model_, None, nn.CrossEntropyLoss(), epochs=epochs,
                                          batch_size=batch_size, threshold=threshold_cl,
                                          dropout=dropout, data_aug=data_aug, transform_test=None,
                                          transform_train=None, dataset_name=dataset, model_name=model,
                                          print_freq=print_freq, patience=args.patience,
                                          lr_decay=args.lr_decay, lr_threshold=args.lr_threshold, threshold_inc=0.003)

        # Training Loop
        accuracy_history = []
        accuracy_list = []
        logging.info('Starting experiment...')

        xtest = data[0]['test']['x']
        ytest = data[0]['test']['y']

        torch_data_test = TensorDataset(xtest, ytest)
        experience_test_zero = AvalancheDataset(torch_data_test)

        task_id = 0
        task_list = []
        for k, ncla in taskcla:
            logging.info('*' * 100)
            logging.info('Task {:2d} ({:s})'.format(k, data[k]['name']))
            logging.info('*' * 100)
            xtrain = data[k]['train']['x']
            ytrain = data[k]['train']['y']
            xvalid = data[k]['valid']['x']
            yvalid = data[k]['valid']['y']
            xtest = data[k]['test']['x']
            ytest = data[k]['test']['y']
            task_list.append(k)

            torch_data_train = TensorDataset(xtrain, ytrain)
            experience_train = AvalancheDataset(torch_data_train)

            torch_data_test = TensorDataset(xvalid, yvalid)
            experience_test = AvalancheDataset(torch_data_test)

            logging.info("Start of experience {0}".format(task_id))
            cl_strategy.train(experience_train, experience_test, experience_zero=experience_test_zero)

            for exp_test_id in task_list:
                xeval = data[exp_test_id]['test']['x']
                yeval = data[exp_test_id]['test']['y']
                torch_data_eval = TensorDataset(xeval, yeval)
                experience_eval = AvalancheDataset(torch_data_eval)
                logging.info('Testing experience: {0}'.format(exp_test_id))
                if exp_test_id <= task_id:
                    acc = cl_strategy.eval(experience_eval, task_id, exp_test_id)
                    accuracy_list.append(acc)
            logging.info(cl_strategy.forgetting_metric.result())
            fogetting_dict = cl_strategy.forgetting_metric.result()
            for key in fogetting_dict.keys():
                wandb.log({"Forgetting_" + key: fogetting_dict[key]}, step=task_id)
            accuracy_history.append(accuracy_list)
            accuracy_list = []
            avg_forgetting = strategy.average_forgetting_metric(fogetting_dict)
            logging.info(f"Average Forgetting: {avg_forgetting}")
            logging.info(f"Average Accuracy: {np.mean(np.array(accuracy_history[-1]))}")
            wandb.log({"avg_forgetting": avg_forgetting,
                       "avg_accuracy": np.mean(np.array(accuracy_history[-1]))}, step=task_id)

            task_id += 1

        avg_forgetting = strategy.average_forgetting_metric(fogetting_dict)
        wandb.run.summary.update({"final_avg_forgetting": avg_forgetting,
                                  "final_avg_accuracy": float(np.mean(np.array(accuracy_history[-1])))})
