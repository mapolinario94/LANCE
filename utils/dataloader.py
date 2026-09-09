import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode  # for bicubic resize/crop

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# -----------------------
# Datasets
# -----------------------
def get_num_classes(dataset_name: str):
    d = dataset_name.lower()
    return {
        "cifar10": 10,
        "cifar100": 100,
        "pets": 37,        # Oxford-IIIT Pet
        "flowers102": 102, # Oxford Flowers-102
        "cub200": 200,     # CUB-200-2011
        "imagenet": 1000,   # ImageNet-1k
        "imagenet1k": 1000, # alias
        "imagenet_splitb": 500,
    }[d]


def get_loaders(dataset: str, data_dir: str, image_size: int, batch_size: int, workers: int,
                val_split: float = 0.0, seed: int = 42):
    """
    Returns train_loader, val_loader.
    For datasets with official test/val split, we use that test (or val) as validation.
    For CIFARs we default to using the official 'test' as validation (val_split is ignored).
    """
    dataset = dataset.lower()
    if dataset in ["imagenet_splitb"]:
        from typing import Dict, List, Tuple, Optional
        import os, copy, math, random, time
        # ------------------------------
        # Utilities
        # ------------------------------
        # IMAGENET_MEAN = (0.485, 0.456, 0.406)
        # IMAGENET_STD  = (0.229, 0.224, 0.225)

        # ------------------------------
        # Class split + dataset builders
        # ------------------------------
        def _get_imagenet_train_folder(root: str) -> str:
            # Expect standard structure: <root>/train/<synset>/*.JPEG and <root>/val/...
            train_dir = os.path.join(root, "train")
            if not os.path.isdir(train_dir):
                raise FileNotFoundError(f"Could not find ImageNet 'train' folder at {train_dir}")
            return train_dir

        def _imagefolder_with_transform(
            root: str,
            t: transforms.Compose,
            target_t = None
        ) -> datasets.ImageFolder:
            ds = datasets.ImageFolder(root=root, transform=t, target_transform=target_t)
            return ds

        def make_class_splits(
            imagenet_root: str,
            mode: str = "alphabetical",  # or "random"
            seed: int = 0
        ) -> Tuple[List[str], List[str]]:
            """
            Returns (classes_A, classes_B) as lists of class folder names (synsets), each length 500.
            """
            train_dir = _get_imagenet_train_folder(imagenet_root)
            probe = datasets.ImageFolder(root=train_dir)  # reads classes from subdirs
            classes = list(probe.classes)  # synset folder names, len=1000

            if len(classes) != 1000:
                raise ValueError(f"Expected 1000 classes, found {len(classes)}. Is this ImageNet-1k?")

            if mode == "alphabetical":
                classes.sort()  # deterministic
            else:
                raise ValueError("mode must be 'alphabetical' or 'random'")

            classes_A = classes[:500]
            classes_B = classes[500:]
            return classes_A, classes_B

        def _filter_and_remap_imagefolder(
            ds: datasets.ImageFolder,
            keep_classnames: List[str]
        ) -> Tuple[Subset, Dict[int,int]]:
            """
            Filter ImageFolder 'ds' to only samples whose class is in keep_classnames.
            Remap original class indices -> [0..len(keep)-1] (contiguous).
            Returns (subset_dataset, mapping_dict).
            """
            # Build map original_idx -> new_idx
            class_to_idx = ds.class_to_idx  # e.g., {'n01440764': 0, ...}
            keep_orig = [class_to_idx[c] for c in keep_classnames]
            keep_orig_sorted = sorted(keep_orig)
            orig_to_new = {orig: new for new, orig in enumerate(keep_orig_sorted)}

            # Filter sample indices
            # ImageFolder stores: ds.samples = [(path, target), ...] and ds.targets
            indices = [i for i, (_, y) in enumerate(ds.samples) if y in orig_to_new]

            # IMPORTANT: give this dataset its own target_transform so B isn't affected
            ds_filtered = copy.deepcopy(ds)
            ds_filtered.target_transform = (lambda y: orig_to_new[y])

            return Subset(ds_filtered, indices), orig_to_new

        def build_transforms(img_size: int = 224, eval_resize: int = 256):
            train_t = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            val_t = transforms.Compose([
                transforms.Resize(eval_resize),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            return train_t, val_t

        def build_split_loaders(
            imagenet_root: str,
            which_half: str,              # "A" or "B"
            batch_size: int = 256,
            workers: int = 8,
            split_mode: str = "alphabetical",
            split_seed: int = 0,
            img_size: int = 224,
            eval_resize: int = 256,
            pin_memory: bool = True,
            persistent_workers: bool = True
        ):
            """
            Returns:
            train_loader, val_loader, num_classes, kept_classnames, orig_to_new_map
            """
            classes_A, classes_B = make_class_splits(imagenet_root, mode=split_mode, seed=split_seed)
            keep = classes_A if which_half.upper() == "A" else classes_B

            train_t, val_t = build_transforms(img_size=img_size, eval_resize=eval_resize)

            train_dir = os.path.join(imagenet_root, "train")
            val_dir   = os.path.join(imagenet_root, "val")

            base_train = _imagefolder_with_transform(train_dir, train_t)
            base_val   = _imagefolder_with_transform(val_dir,   val_t)

            train_subset, mapping_train = _filter_and_remap_imagefolder(base_train, keep)
            val_subset,   mapping_val   = _filter_and_remap_imagefolder(base_val,   keep)

            # Safety: mappings should be identical on keys
            if set(mapping_train.keys()) != set(mapping_val.keys()):
                raise RuntimeError("Train/Val class filtering mismatch. Check dataset structure.")

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers, 
                drop_last=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers, 
                drop_last=True
            )

            num_classes = 500
            return train_loader, val_loader, num_classes, keep, mapping_train
        
        imagenet_root= "/local/a/imagenet/imagenet2012"
        pretrain_half= "A"           # "A" or "B" model you trained on
        batch_size = batch_size
        workers = workers
        split_mode = "alphabetical"
        split_seed = 0
        img_size = image_size
        eval_resize = 256

        other = "B" if pretrain_half.upper() == "A" else "A"
        train_loader, val_loader, num_classes, keep, mapping_train =  build_split_loaders(
            imagenet_root=imagenet_root,
            which_half=other,
            batch_size=batch_size,
            workers=workers,
            split_mode=split_mode,
            split_seed=split_seed,
            img_size=img_size,
            eval_resize=eval_resize
        )
        return train_loader, val_loader


    # ---------------- ImageNet ----------------
    if dataset in ["imagenet", "imagenet1k"]:
        # Standard ImageNet recipe
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        # Resize short side to 256 for 224, scaled generally by image_size
        val_resize = int(round(image_size * (256.0 / 224.0)))
        test_tfms = transforms.Compose([
            transforms.Resize(val_resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        train_dir = os.path.join("/local/a/imagenet/imagenet2012", "train")
        val_dir   = os.path.join("/local/a/imagenet/imagenet2012", "val")

        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            tr = datasets.ImageFolder(train_dir, transform=train_tfms)
            te = datasets.ImageFolder(val_dir,   transform=test_tfms)
        else:
            # Fallback to torchvision's ImageNet class (expects ILSVRC layout)
            try:
                tr = datasets.ImageNet(root=data_dir, split="train", transform=train_tfms)
                te = datasets.ImageNet(root=data_dir, split="val",   transform=test_tfms)
            except Exception as e:
                raise FileNotFoundError(
                    "ImageNet not found. Expected:\n"
                    f"  - Folder layout: {data_dir}/train and {data_dir}/val with class subfolders\n"
                    "    OR a torchvision ImageNet installation at data_dir.\n"
                    f"Original error: {repr(e)}"
                )

        tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                               pin_memory=True, persistent_workers=workers > 0)
        te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=workers,
                               pin_memory=True, persistent_workers=workers > 0)
        return tr_loader, te_loader

    if dataset in ["cifar10", "cifar100"]:
        train_tfms = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.RandomCrop(image_size, padding=int(0.1 * image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_tfms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        if dataset == "cifar10":
            tr = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
            te = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)
        else:
            tr = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tfms)
            te = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tfms)

        tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                               pin_memory=True, persistent_workers=workers>0)
        te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=workers,
                               pin_memory=True, persistent_workers=workers>0)
        return tr_loader, te_loader

    # Generic transforms for larger natural images
    train_tfms = transforms.Compose([
        # transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tfms = transforms.Compose([
        # transforms.Resize(int(image_size * 1.15)),
        transforms.Resize((image_size, image_size)),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if dataset == "pets":
        # Oxford-IIIT Pet: use trainval for training and test for validation
        tr = datasets.OxfordIIITPet(root=data_dir, split="trainval", download=True, transform=train_tfms)
        te = datasets.OxfordIIITPet(root=data_dir, split="test",     download=True, transform=test_tfms)

    elif dataset == "flowers102":
        # Flowers102 provides 'train', 'val', 'test'. We'll train on 'train' and eval on 'val' by default.
        tr = datasets.Flowers102(root=data_dir, split="train", download=True, transform=train_tfms)
        te = datasets.Flowers102(root=data_dir, split="val",   download=True, transform=test_tfms)

    elif dataset == "cub200":
        # Try torchvision's CUB200 if present; else fallback to simple parser for the official split files.
        if hasattr(datasets, "CUB200"):
            tr = datasets.CUB200(root=data_dir, train=True, download=True, transform=train_tfms)
            te = datasets.CUB200(root=data_dir, train=False, download=True, transform=test_tfms)
        else:
            tr, te = _cub200_fallback_datasets(data_dir, train_tfms, test_tfms)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # (Optional) user override: carve out a small val split from training
    if val_split and 0.0 < val_split < 1.0:
        g = torch.Generator().manual_seed(seed)
        n_train = int((1 - val_split) * len(tr))
        n_val = len(tr) - n_train
        tr, extra_val = random_split(tr, [n_train, n_val], generator=g)
        # merge extra_val into existing val set by concatenating via Subset indices mechanism:
        # simplest: ignore original te and use extra_val for validation
        te = extra_val

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                           pin_memory=True, persistent_workers=workers>0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=workers,
                           pin_memory=True, persistent_workers=workers>0)
    return tr_loader, te_loader


def _cub200_fallback_datasets(root: str, train_tfms, test_tfms):
    """
    Minimal loader for CUB-200-2011 using the official split files.
    Expects the dataset extracted at: <root>/CUB_200_2011
    """
    base = os.path.join(root, "CUB_200_2011")
    images_txt  = os.path.join(base, "images.txt")
    labels_txt  = os.path.join(base, "image_class_labels.txt")
    split_txt   = os.path.join(base, "train_test_split.txt")
    images_dir  = os.path.join(base, "images")

    if not (os.path.exists(images_txt) and os.path.exists(labels_txt) and os.path.exists(split_txt)):
        raise FileNotFoundError(
            "CUB200 fallback could not find the split files. "
            "Expected under <data_dir>/CUB_200_2011/{images.txt,image_class_labels.txt,train_test_split.txt}"
        )

    # Parse
    with open(images_txt, "r") as f:
        id_to_rel = {int(line.split()[0]): line.split()[1] for line in f}
    with open(labels_txt, "r") as f:
        id_to_label = {int(line.split()[0]): int(line.split()[1]) - 1 for line in f}  # 1-indexed -> 0-indexed
    train_ids, test_ids = [], []
    with open(split_txt, "r") as f:
        for line in f:
            _id, is_train = line.split()
            (_id, is_train) = (int(_id), int(is_train))
            (train_ids if is_train == 1 else test_ids).append(_id)

    # Build datasets using ImageFolder-like access
    class CUBSubset(torch.utils.data.Dataset):
        def __init__(self, ids, transform):
            self.ids = ids
            self.transform = transform
        def __len__(self): return len(self.ids)
        def __getitem__(self, idx):
            img_id = self.ids[idx]
            rel = id_to_rel[img_id]
            path = os.path.join(images_dir, rel)
            img = datasets.folder.default_loader(path)
            y = id_to_label[img_id]
            if self.transform: img = self.transform(img)
            return img, y

    return CUBSubset(train_ids, train_tfms), CUBSubset(test_ids, test_tfms)
