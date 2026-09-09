# main.py
# Unified memory-efficient fine-tuning entry point.
#
# CNN backbones (mobilenetv2 / resnet18 / resnet34 / mcunet) replace the last N
# Conv2d layers; DeiT-Tiny (deit_tiny, via timm) replaces the last N body Linear
# layers. In both cases the replacement is one of:
#   - LANCE   (--use-lance)
#   - HOSVD   (--use-hosvd; variance-threshold, or fixed-rank with --load-lance-ranks)
#   - vanilla back-prop baseline (no flag)
# The classifier head is always trainable.

import argparse
import json
import os
import random
import time
from pathlib import Path

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm

from custom_layers.conv_hosvd import wrap_convHOSVD, Conv2d_HOSVD
from custom_layers.conv_lance import wrap_convLANCE, Conv2d_LANCE
from custom_layers.linear_lance import Linear_LANCEViT, wrap_linearLANCE
from custom_layers.linear_hosvd import (
    Linear_HOSVD_fixed_rank,
    wrap_linearHOSVD_var,
    wrap_linearHOSVD_fixed_rank,
)

from utils.training_eval import (
    train_one_epoch,
    evaluate,
    warm_up_convLANCE_layers,
    warm_up_linearLANCE_layers,
)
from utils.dataloader import get_loaders, get_num_classes
from utils.model_builder import (
    build_model_and_size,
    freeze_all,
    unfreeze_last_n_convs,
    unfreeze_last_linear,
    get_parent_and_child_name,
)

CNN_ARCHS = ("mobilenetv2", "resnet18", "resnet34", "mcunet")
VIT_ARCHS = ("deit_tiny",)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, num_classes: int):
    """Return (model, native_image_size) for the requested architecture."""
    if args.arch == "deit_tiny":
        model = timm.create_model(args.model_id, pretrained=True, num_classes=num_classes)
        return model, 224
    return build_model_and_size(
        args.arch, num_classes, net_id=args.net_id, pretrain_model_path=args.pretrain_model_path
    )


def set_norms_eval_for_frozen(model: nn.Module):
    """Keep normalization layers of frozen subgraphs in eval() so their running
    stats are not updated. Handles BatchNorm (CNN) and LayerNorm (ViT)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
            params = list(m.parameters())
            if len(params) == 0 or all((not p.requires_grad) for p in params):
                m.eval()
        if isinstance(m, nn.Dropout):
            m.eval()


# -----------------------
# Conv (CNN) replacement
# -----------------------
@torch.no_grad()
def replace_last_n_convs_with_lance(model, n, *, active=True, rank=8, threshold=0.9, unfreeze_replaced=True):
    conv_qnames = [
        qn for qn, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and getattr(m, "groups", 1) == 1  # skip depthwise
    ]
    if not conv_qnames:
        return 0
    n = max(0, min(n, len(conv_qnames)))
    for qn in conv_qnames[-n:]:
        conv = model.get_submodule(qn)
        parent, child = get_parent_and_child_name(model, qn)
        new_conv = wrap_convLANCE(conv, active=active, rank=rank, threshold=threshold)
        new_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)
        parent._modules[child] = new_conv
        if unfreeze_replaced:
            for p in new_conv.parameters():
                p.requires_grad = True
    unfreeze_last_linear(model)  # let head adapt
    return n


@torch.no_grad()
def replace_last_n_convs_with_hosvd(model, n, *, active=True, explained_var=0.9, unfreeze_replaced=True):
    conv_qnames = [qn for qn, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if not conv_qnames:
        return 0
    n = max(0, min(n, len(conv_qnames)))
    for qn in conv_qnames[-n:]:
        conv = model.get_submodule(qn)
        parent, child = get_parent_and_child_name(model, qn)
        new_conv = wrap_convHOSVD(conv, active=active, explained_var=explained_var)
        new_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)
        parent._modules[child] = new_conv
        if unfreeze_replaced:
            for p in new_conv.parameters():
                p.requires_grad = True
    unfreeze_last_linear(model)  # let head adapt
    return n


# -----------------------
# Linear (ViT) replacement
# -----------------------
def _find_head_qname(model: nn.Module):
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return "head"
    last_qn = None
    for qn, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_qn = qn
    return last_qn


def _body_linear_qnames(model: nn.Module):
    head_qn = _find_head_qname(model)
    return [qn for qn, m in model.named_modules() if isinstance(m, nn.Linear) and qn != head_qn]


@torch.no_grad()
def replace_last_n_linears_with_lance(model, n, *, active=True, rank=8, threshold=0.9, unfreeze_replaced=True):
    qnames = _body_linear_qnames(model)
    n = max(0, min(n, len(qnames)))
    for qn in qnames[-n:]:
        lin = model.get_submodule(qn)
        parent, child = get_parent_and_child_name(model, qn)
        new_lin = wrap_linearLANCE(lin, active=active, rank=rank, threshold=threshold)
        new_lin.to(device=lin.weight.device, dtype=lin.weight.dtype)
        parent._modules[child] = new_lin
        if unfreeze_replaced:
            for p in new_lin.parameters():
                p.requires_grad = True
    unfreeze_last_linear(model)
    return n


@torch.no_grad()
def replace_last_n_linears_with_hosvd(model, n, *, active=True, explained_var=0.9, fixed_ranks=None, unfreeze_replaced=True):
    """If `fixed_ranks` (qname -> per-mode ranks) is given, each layer uses a
    Linear_HOSVD_fixed_rank (iso-memory with LANCE); otherwise the variance-
    threshold HOSVD layer is used."""
    qnames = _body_linear_qnames(model)
    n = max(0, min(n, len(qnames)))
    for qn in qnames[-n:]:
        lin = model.get_submodule(qn)
        parent, child = get_parent_and_child_name(model, qn)
        if fixed_ranks is not None:
            if qn not in fixed_ranks:
                raise KeyError(f"--load-lance-ranks file is missing ranks for layer '{qn}'")
            new_lin = wrap_linearHOSVD_fixed_rank(lin, ranks=fixed_ranks[qn], active=active)
        else:
            new_lin = wrap_linearHOSVD_var(lin, active=active, SVD_var=explained_var, k_hosvd=None)
        new_lin.to(device=lin.weight.device, dtype=lin.weight.dtype)
        parent._modules[child] = new_lin
        if unfreeze_replaced:
            for p in new_lin.parameters():
                p.requires_grad = True
    unfreeze_last_linear(model)
    return n


def unfreeze_last_n_linears(model: nn.Module, n: int, include_head: bool = True):
    qnames = _body_linear_qnames(model)
    n = max(0, min(n, len(qnames)))
    for qn in qnames[-n:]:
        for p in model.get_submodule(qn).parameters():
            p.requires_grad = True
    if include_head:
        unfreeze_last_linear(model)


# -----------------------
# Metrics (ViT)
# -----------------------
def collect_lance_ranks(model: nn.Module) -> dict:
    """After LANCE warmup, extract per-layer per-mode ranks from u0/u1/u2 shapes."""
    out = {}
    for qn, m in model.named_modules():
        if isinstance(m, Linear_LANCEViT):
            ranks = []
            for i in range(3):
                u = getattr(m, f"u{i}", None)
                if u is None:
                    raise RuntimeError(f"Layer {qn} has u{i} == None; warmup did not run.")
                ranks.append(int(u.shape[1]))
            out[qn] = ranks
    return out


def measure_bp_metrics(model, loader, device, body_qnames):
    """Forward-hook one batch on the unfrozen body Linears to compute vanilla BP
    activation memory and forward FLOPs (matches Linear_LANCEViT accounting)."""
    shapes = {}
    handles = []

    def make_hook(qn):
        def _h(module, inputs, outputs):
            shapes[qn] = tuple(inputs[0].shape)
        return _h

    for qn in body_qnames:
        m = model.get_submodule(qn)
        handles.append(m.register_forward_hook(make_hook(qn)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            break
    if was_training:
        model.train()
    for h in handles:
        h.remove()

    total_mem = 0.0
    total_flops = 0.0
    for qn, shape in shapes.items():
        m = model.get_submodule(qn)
        if len(shape) == 3:
            B, L, F = shape
            n_act = B * L
        elif len(shape) == 2:
            B, F = shape
            n_act = B
            L = 1
        else:
            continue
        total_mem += B * L * m.in_features * 4 / (1024 ** 2)
        total_flops += 2 * m.in_features * m.out_features * n_act
    return total_mem, total_flops


def aggregate_compression_metrics(model):
    """Sum memory / FLOPs across LANCE and fixed-rank HOSVD layers."""
    out = {
        "compressed_memory_MB": 0.0,
        "vanilla_memory_MB": 0.0,
        "compressed_flops": 0.0,
        "vanilla_flops": 0.0,
        "n_layers": 0,
    }
    for _, m in model.named_modules():
        if isinstance(m, (Linear_LANCEViT, Linear_HOSVD_fixed_rank)):
            out["compressed_memory_MB"] += float(m.memory_MB)
            out["vanilla_memory_MB"] += float(m.memory_BP_MB)
            out["compressed_flops"] += float(m.flops)
            out["vanilla_flops"] += float(m.flops_BP)
            out["n_layers"] += 1
    return out


def log_conv_memory_flops(model, use_lance, use_hosvd, epoch):
    """Log conv-path activation memory / FLOPs to wandb (CNN backbones)."""
    if use_lance:
        tot_mem = tot_orig_mem = 0.0
        fw = bw = fw_o = bw_o = 0.0
        for _, m in model.named_modules():
            if isinstance(m, Conv2d_LANCE):
                if m.compressed_memory is not None and m.original_memory is not None:
                    tot_mem += m.compressed_memory
                    tot_orig_mem += m.original_memory
                if None not in (m.flops_fw, m.flops_bw, m.flops_fw_original, m.flops_bw_original):
                    fw += m.flops_fw
                    bw += m.flops_bw
                    fw_o += m.flops_fw_original
                    bw_o += m.flops_bw_original
        if tot_mem > 0.0 and tot_orig_mem > 0.0:
            wandb.log({
                "total_lance_memory_MB": tot_mem,
                "total_vanillabp_memory_MB": tot_orig_mem,
                "total_memory_ratio": tot_mem / tot_orig_mem,
            }, step=epoch)
        if fw > 0.0 and bw > 0.0 and fw_o > 0.0 and bw_o > 0.0:
            wandb.log({
                "total_lance_flops": fw + bw,
                "total_vanillabp_flops": fw_o + bw_o,
                "total_flops_ratio": (fw + bw) / (fw_o + bw_o),
            }, step=epoch)
    if use_hosvd:
        tot_mem = 0.0
        fw = bw = 0.0
        for _, m in model.named_modules():
            if isinstance(m, Conv2d_HOSVD):
                if m.compressed_memory is not None:
                    tot_mem += m.compressed_memory
                if m.flops_fw is not None and m.flops_bw is not None:
                    fw += m.flops_fw
                    bw += m.flops_bw
        if tot_mem > 0.0:
            wandb.log({"total_hosvd_memory_MB": tot_mem}, step=epoch)
        if fw > 0.0 and bw > 0.0:
            wandb.log({"total_hosvd_flops": fw + bw}, step=epoch)


# -----------------------
# Main
# -----------------------
def main(args):
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    is_vit = args.arch in VIT_ARCHS
    num_classes = get_num_classes(args.dataset)
    model, native_img_size = build_model(args, num_classes)
    img_size = args.image_size or native_img_size

    freeze_all(model)

    # Optional per-layer ranks (ViT HOSVD iso-memory comparison)
    fixed_ranks = None
    if is_vit and args.use_hosvd and args.load_lance_ranks:
        with open(os.path.expanduser(args.load_lance_ranks), "r") as f:
            fixed_ranks = json.load(f)
        print(f"Loaded LANCE ranks for {len(fixed_ranks)} layers from {args.load_lance_ranks}")

    # ---- replace / unfreeze last N layers ----
    if is_vit:
        if args.use_lance:
            replaced = replace_last_n_linears_with_lance(
                model, args.n_unfrozen, active=not args.lance_no_activate,
                rank=args.lance_rank, threshold=args.lance_threshold,
            )
            print(f"Replaced + unfroze last {replaced} body Linears with Linear_LANCEViT")
        elif args.use_hosvd:
            replaced = replace_last_n_linears_with_hosvd(
                model, args.n_unfrozen, explained_var=args.explained_var, fixed_ranks=fixed_ranks,
            )
            kind = "Linear_HOSVD_fixed_rank" if fixed_ranks is not None else "Linear_HOSVD_var"
            print(f"Replaced + unfroze last {replaced} body Linears with {kind}")
        else:
            unfreeze_last_n_linears(model, args.n_unfrozen, include_head=True)
            print(f"Unfroze last {args.n_unfrozen} body Linears (+ head)")
        body_qnames_unfrozen = _body_linear_qnames(model)[-args.n_unfrozen:] if args.n_unfrozen > 0 else []
    else:
        if args.use_lance:
            replaced = replace_last_n_convs_with_lance(
                model, args.n_unfrozen, active=not args.lance_no_activate,
                rank=args.lance_rank, threshold=args.lance_threshold,
            )
            print(f"Replaced + unfroze last {replaced} convs with Conv2d_LANCE")
        elif args.use_hosvd:
            replaced = replace_last_n_convs_with_hosvd(
                model, args.n_unfrozen, explained_var=args.explained_var,
            )
            print(f"Replaced + unfroze last {replaced} convs with Conv2d_HOSVD")
        else:
            unfreeze_last_n_convs(model, args.n_unfrozen, include_classifier=True)
            print(f"Unfroze last {args.n_unfrozen} convs (+ head)")
        body_qnames_unfrozen = []

    set_norms_eval_for_frozen(model)
    model.to(device)

    train_loader, val_loader = get_loaders(
        dataset=args.dataset, data_dir=args.data_dir, image_size=img_size,
        batch_size=args.batch_size, workers=args.workers,
        val_split=args.val_split, seed=args.seed,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No parameters are unfrozen; increase --n-unfrozen."

    if args.optim == "sgd":
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == "adam":
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = None  # AMP disabled

    # ---- LANCE warm-up ----
    if args.use_lance and args.lance_warmup_batches > 0:
        print(f"LANCE warmup for {args.lance_warmup_batches} batches...")
        if is_vit:
            warm_up_linearLANCE_layers(model, train_loader, device, max_batches=args.lance_warmup_batches)
            if args.save_lance_ranks:
                ranks = collect_lance_ranks(model)
                out_path = Path(os.path.expanduser(args.save_lance_ranks))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(ranks, f, indent=2)
                print(f"Saved per-layer LANCE ranks to {out_path}")
                wandb.config.update({"lance_ranks_path": str(out_path)})
        else:
            warm_up_convLANCE_layers(model, train_loader, device, max_batches=args.lance_warmup_batches)

    # ---- BP activation memory / FLOPs (ViT baseline) ----
    bp_metrics = None
    if is_vit and not (args.use_lance or args.use_hosvd) and body_qnames_unfrozen:
        bp_mem, bp_flops = measure_bp_metrics(model, train_loader, device, body_qnames_unfrozen)
        bp_metrics = {"vanilla_memory_MB": bp_mem, "vanilla_flops": bp_flops}

    # ---- train ----
    best = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args.log_interval,
            measure_grad_angle=args.measure_grad_angle,
        )
        va_loss, va_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"[Epoch {epoch:03d}] train {tr_loss:.4f}/{tr_acc:.2f}% | val {va_loss:.4f}/{va_acc:.2f}%")
        wandb.log({
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        if not is_vit and epoch == 1:
            log_conv_memory_flops(model, args.use_lance, args.use_hosvd, epoch)
        elif not is_vit and args.use_hosvd:
            log_conv_memory_flops(model, False, True, epoch)

        if va_acc > best:
            best = va_acc
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            ckpt = Path(args.out_dir) / f"{args.arch}_{args.dataset}_best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": va_acc}, ckpt)
            print(f"  >> Saved checkpoint to {ckpt} (best acc: {best:.2f}%)")
            wandb.log({"best_val_acc": best}, step=epoch)

    elapsed = (time.time() - start) / 60
    print(f"Done. Best val acc: {best:.2f}% | Time: {elapsed:.1f} min")

    # ---- end-of-run memory & FLOPs report (ViT) ----
    if is_vit:
        print("\n" + "=" * 60)
        print(f"Memory & FLOPs summary  (method={args.method}, n_unfrozen={args.n_unfrozen})")
        print("=" * 60)
        if args.method in ("lance", "hosvd_fixed", "hosvd_var"):
            agg = aggregate_compression_metrics(model)
            ratio = (agg["compressed_memory_MB"] / agg["vanilla_memory_MB"]) if agg["vanilla_memory_MB"] else 0.0
            print(f"  layers tracked        : {agg['n_layers']}")
            print(f"  vanilla activation MB : {agg['vanilla_memory_MB']:.4f}")
            print(f"  compressed memory MB  : {agg['compressed_memory_MB']:.4f}")
            print(f"  compression ratio     : {ratio:.4f}")
            print(f"  vanilla FLOPs         : {agg['vanilla_flops']:.3e}")
            print(f"  method  FLOPs         : {agg['compressed_flops']:.3e}")
            wandb.run.summary.update({
                "final_vanilla_memory_MB": agg["vanilla_memory_MB"],
                "final_compressed_memory_MB": agg["compressed_memory_MB"],
                "final_memory_ratio": ratio,
                "final_vanilla_flops": agg["vanilla_flops"],
                "final_compressed_flops": agg["compressed_flops"],
            })
        elif bp_metrics is not None:
            print(f"  vanilla activation MB : {bp_metrics['vanilla_memory_MB']:.4f}")
            print(f"  vanilla FLOPs         : {bp_metrics['vanilla_flops']:.3e}")
            wandb.run.summary.update({
                "final_vanilla_memory_MB": bp_metrics["vanilla_memory_MB"],
                "final_vanilla_flops": bp_metrics["vanilla_flops"],
            })
        else:
            print("  (no body Linears tracked)")
        print("=" * 60)


def build_argparser():
    p = argparse.ArgumentParser(
        description="LANCE / HOSVD memory-efficient fine-tuning "
                    "(MobileNetV2 / ResNet18/34 / MCUNet / DeiT-Tiny)."
    )
    # Data / model
    p.add_argument("--data-dir", type=str, default="~/Datasets", help="Root dataset directory")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["cifar10", "cifar100", "pets", "flowers102", "cub200", "imagenet_splitB"])
    p.add_argument("--arch", type=str, required=True,
                   choices=list(CNN_ARCHS) + list(VIT_ARCHS),
                   help="Backbone. deit_tiny uses the timm linear path (see --model-id); "
                        "the others use the conv path.")
    p.add_argument("--net-id", type=str, default="mcunet-in2", help="MCUNet model id (if --arch mcunet)")
    p.add_argument("--model-id", type=str, default="deit_tiny_patch16_224",
                   help="timm model id (if --arch deit_tiny)")
    p.add_argument("--image-size", type=int, default=None, help="Override model's native input size (e.g., 224)")
    p.add_argument("--pretrain-model-path", type=str, default=None,
                   help="Optional checkpoint for the CNN backbone (e.g. ImageNet-splitB pretraining).")

    # Train setup (single neutral default set; pass explicit flags per experiment)
    p.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)  # for SGD
    p.add_argument("--val-split", type=float, default=0.0,
                   help="Optional fraction carved from training set for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-interval", type=int, default=100)

    # Fine-tuning granularity
    p.add_argument("--n-unfrozen", type=int, default=2,
                   help="Number of last Conv2d (CNN) or body Linear (ViT) layers to unfreeze/replace")
    p.add_argument("--use-lance", action="store_true", help="Replace last N layers with the LANCE variant")
    p.add_argument("--lance-rank", type=int, default=8)
    p.add_argument("--lance-no-activate", action="store_true")
    p.add_argument("--lance-warmup-batches", type=int, default=100,
                   help="Batches used to warm up LANCE subspaces (0=skip)")
    p.add_argument("--lance-threshold", type=float, default=0.9,
                   help="Variance threshold for LANCE subspace initialization")
    p.add_argument("--measure-grad-angle", action="store_true", help="Measure gradient alignment")

    # HOSVD options
    p.add_argument("--use-hosvd", action="store_true", help="Replace last N layers with the HOSVD variant")
    p.add_argument("--explained-var", type=float, default=0.9,
                   help="Explained variance for HOSVD (ignored if --load-lance-ranks is set)")
    p.add_argument("--load-lance-ranks", type=str, default=None,
                   help="[deit_tiny] JSON with per-layer per-mode ranks (from a LANCE run). With --use-hosvd, "
                        "wraps each Linear at those fixed ranks for an iso-memory comparison.")
    p.add_argument("--save-lance-ranks", type=str, default=None,
                   help="[deit_tiny] With --use-lance, write per-layer per-mode ranks to this JSON after warmup.")

    # Logging
    p.add_argument("--experiment-name", type=str, default="lance_finetune", help="wandb project name")
    return p


def resolve_method(args):
    if args.use_lance:
        return "lance"
    if args.use_hosvd:
        return "hosvd_fixed" if args.load_lance_ranks else "hosvd_var"
    return "bp"


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if sum([bool(args.use_lance), bool(args.use_hosvd)]) > 1:
        raise ValueError("Pass at most one of --use-lance / --use-hosvd.")

    # Per-layer rank files are only supported on the ViT (linear) path.
    if args.arch not in VIT_ARCHS and (args.load_lance_ranks or args.save_lance_ranks):
        raise ValueError("--load-lance-ranks / --save-lance-ranks are only supported for --arch deit_tiny.")

    args.method = resolve_method(args)

    run = wandb.init(project=args.experiment_name, config=vars(args))
    print(f"wandb run: {run.name} ({run.id})")
    main(args)
    wandb.finish()
