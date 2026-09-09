# Adapted from: https://github.com/joansj/hat/blob/master/src/dataloaders/mixture.py
import os
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import urllib.request
from PIL import Image
import pickle

__all__ = ["get_5datasets"]


def get_5datasets(data_dir, seed=1, pc_valid=0.05):
    """Load the 5-Datasets benchmark (CIFAR-10, MNIST, SVHN, Fashion-MNIST,
    notMNIST) as five tasks of 10 classes each.

    Each dataset is downloaded under `data_dir`; the per-task tensors are cached
    under `data_dir/binary_mixture_5_Data` on first run and reloaded afterwards.
    """
    data_dir = os.path.expanduser(data_dir)
    bin_dir = os.path.join(data_dir, 'binary_mixture_5_Data')
    data = {}
    taskcla = []
    size = [3, 32, 32]

    idata = np.arange(5)

    if not os.path.isdir(bin_dir):
        os.makedirs(bin_dir)
        for n, idx in enumerate(idata):
            if idx == 0:
                # CIFAR-10
                mean = [x / 255 for x in [125.3, 123.0, 113.9]]
                std = [x / 255 for x in [63.0, 62.1, 66.7]]
                tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
                dat = {
                    'train': datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm),
                    'test': datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm),
                }
                data[n] = {'name': 'cifar10', 'ncla': 10}
                for s in ['train', 'test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[n][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            elif idx == 1:
                # MNIST (padded to 32x32, expanded to 3 channels)
                mean = (0.1,)
                std = (0.2752,)
                tfm = transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
                dat = {
                    'train': datasets.MNIST(data_dir, train=True, download=True, transform=tfm),
                    'test': datasets.MNIST(data_dir, train=False, download=True, transform=tfm),
                }
                data[n] = {'name': 'mnist', 'ncla': 10}
                for s in ['train', 'test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[n][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        image = image.expand(1, 3, image.size(2), image.size(3))
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            elif idx == 2:
                # SVHN
                mean = [0.4377, 0.4438, 0.4728]
                std = [0.198, 0.201, 0.197]
                tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
                dat = {
                    'train': datasets.SVHN(data_dir, split='train', download=True, transform=tfm),
                    'test': datasets.SVHN(data_dir, split='test', download=True, transform=tfm),
                }
                data[n] = {'name': 'svhn', 'ncla': 10}
                for s in ['train', 'test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[n][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            elif idx == 3:
                # Fashion-MNIST (padded to 32x32, expanded to 3 channels)
                mean = (0.2190,)
                std = (0.3318,)
                tfm = transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
                fm_dir = os.path.join(data_dir, 'fashion_mnist')
                dat = {
                    'train': FashionMNIST(fm_dir, train=True, download=True, transform=tfm),
                    'test': FashionMNIST(fm_dir, train=False, download=True, transform=tfm),
                }
                data[n] = {'name': 'fashion-mnist', 'ncla': 10}
                for s in ['train', 'test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[n][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        image = image.expand(1, 3, image.size(2), image.size(3))
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            elif idx == 4:
                # notMNIST (A-J letter glyphs, expanded to 3 channels)
                mean = (0.4254,)
                std = (0.4501,)
                tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
                nm_dir = os.path.join(data_dir, 'notmnist')
                dat = {
                    'train': notMNIST(nm_dir, train=True, download=True, transform=tfm),
                    'test': notMNIST(nm_dir, train=False, download=True, transform=tfm),
                }
                data[n] = {'name': 'notmnist', 'ncla': 10}
                for s in ['train', 'test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[n][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        image = image.expand(1, 3, image.size(2), image.size(3))
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            # Unify and cache one binary per (task, split)
            for s in ['train', 'test']:
                data[n][s]['x'] = torch.stack(data[n][s]['x']).view(-1, size[0], size[1], size[2])
                data[n][s]['y'] = torch.LongTensor(np.array(data[n][s]['y'], dtype=int)).view(-1)
                torch.save(data[n][s]['x'], os.path.join(bin_dir, 'data' + str(idx) + s + 'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(bin_dir, 'data' + str(idx) + s + 'y.bin'))

    else:
        # Load cached binaries
        names = {0: 'cifar10', 1: 'mnist', 2: 'svhn', 3: 'fashion-mnist', 4: 'notmnist'}
        for n, idx in enumerate(idata):
            data[n] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[n]['name'] = names[idx]
            data[n]['ncla'] = 10
            for s in ['train', 'test']:
                data[n][s] = {'x': [], 'y': []}
                data[n][s]['x'] = torch.load(os.path.join(bin_dir, 'data' + str(idx) + s + 'x.bin'))
                data[n][s]['y'] = torch.load(os.path.join(bin_dir, 'data' + str(idx) + s + 'y.bin'))

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


class FashionMNIST(datasets.MNIST):
    """Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>."""
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]


class notMNIST(torch.utils.data.Dataset):
    """notMNIST: font glyphs for the letters A-J (MNIST-like)."""

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from ' + self.url)
                self.download()

        training_file = 'notmnist_train.pkl'
        testing_file = 'notmnist_test.pkl'
        split_file = training_file if train else testing_file
        with open(os.path.join(root, split_file), 'rb') as f:
            split = pickle.load(f)
        self.data = split['features'].astype(np.uint8)
        self.labels = split['labels'].astype(np.uint8)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        import zipfile
        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        urllib.request.urlretrieve(self.url, fpath)
        with zipfile.ZipFile(fpath, 'r') as zip_ref:
            zip_ref.extractall(root)
