import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle

__all__ = ["get"]


def get(data_dir, seed=0, pc_valid=0.10):
    """Load Split CIFAR-100 as 10 tasks of 10 classes each.

    CIFAR-100 is downloaded to `data_dir`; the per-task tensors are cached under
    `data_dir/binary_cifar100` on first run and reloaded afterwards.
    """
    data_dir = os.path.expanduser(data_dir)
    file_dir = os.path.join(data_dir, 'binary_cifar100')
    data = {}
    taskcla = []
    size = [3, 32, 32]

    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        dat = {
            'train': datasets.CIFAR100(data_dir, train=True, download=True, transform=tfm),
            'test': datasets.CIFAR100(data_dir, train=False, download=True, transform=tfm),
        }
        for n in range(10):
            data[n] = {'name': 'cifar100', 'ncla': 10,
                       'train': {'x': [], 'y': []}, 'test': {'x': [], 'y': []}}
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                data[n // 10][s]['x'].append(image)
                data[n // 10][s]['y'].append(n % 10)

        # Unify and cache one binary per (task, split)
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(file_dir, 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(file_dir, 'data' + str(t) + s + 'y.bin'))

    # Load cached binaries
    data = {}
    ids = list(np.arange(10))
    for i in range(10):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(file_dir, 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(file_dir, 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100-' + str(ids[i])

    # Carve out a validation split per task
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Task/class bookkeeping
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size
