import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from custom_layers.conv_lance import Conv2d_LANCE
from custom_layers.linear_lance import Linear_LANCEViT

__all__ = ["train_one_epoch", "evaluate", "warm_up_convLANCE_layers", "warm_up_linearLANCE_layers"]

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        bsz = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / bsz)).item())
        return res


@torch.no_grad()
def warm_up_convLANCE_layers(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 100):
    model.eval()
    for m in model.modules():
        if isinstance(m, Conv2d_LANCE):
            setattr(m, "record_feature", True)
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model(images)
        seen += 1
        if seen >= max_batches:
            break
    for m in model.modules():
        if isinstance(m, Conv2d_LANCE):
            setattr(m, "record_feature", False)
            if hasattr(m, "initialize_subspace"):
                m.initialize_subspace()

@torch.no_grad()
def warm_up_linearLANCE_layers(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 100):
    model.eval()
    for m in model.modules():
        if isinstance(m, Linear_LANCEViT):
            setattr(m, "record_feature", True)
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model(images)
        seen += 1
        if seen >= max_batches:
            break
    for m in model.modules():
        if isinstance(m, Linear_LANCEViT):
            setattr(m, "record_feature", False)
            if hasattr(m, "initialize_subspace"):
                m.initialize_subspace()

    for m in model.modules():
        if isinstance(m, Linear_LANCEViT):
            setattr(m, "record_hardware_metrics", True)
    
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model(images)
        seen += 1
        if seen >= 1:
            break
    
    total_memory = 0
    total_flops = 0
    total_memory_BP = 0
    total_flops_BP = 0

    for m in model.modules():
        if isinstance(m, Linear_LANCEViT):
            setattr(m, "record_hardware_metrics", False)
            total_flops += m.flops
            total_flops_BP += m.flops_BP
            total_memory += m.memory_MB
            total_memory_BP += m.memory_BP_MB
            wandb.log({
                "total_memory_MB": total_memory,
                "total_memory_BP": total_memory_BP,
                "total_flops": total_flops,
                "total_flops_BP": total_flops_BP,
            }, step=0)


def _param_owner_map(model: nn.Module):
    """
    Build a mapping from parameter object id -> (module_qualified_name, param_name),
    and a module->list[param tensors] map (own params only).
    """
    id_to_owner = {}
    module_to_params = {}
    for mod_qn, mod in model.named_modules():
        own_params = []
        for pname, p in mod.named_parameters(recurse=False):
            own_params.append(p)
            id_to_owner[id(p)] = (mod_qn, pname)
        if own_params:
            module_to_params[mod_qn] = own_params
    return id_to_owner, module_to_params


def _concat_grads(params):
    """Flatten & concat grads (skip None/frozen) into a 1-D tensor; return None if empty."""
    vecs = []
    for p in params:
        # if len(p.size())==1:
        #     continue
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g is None:
            continue
        vecs.append(g.reshape(-1))
    if not vecs:
        return None
    return torch.cat(vecs, dim=0)


def _angle_between(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12) -> float:
    """Return angle in degrees between v1 and v2."""
    n1 = torch.linalg.norm(v1)
    n2 = torch.linalg.norm(v2)
    if n1 <= eps or n2 <= eps:
        return float("nan")
    cos_sim = torch.clamp(torch.dot(v1, v2) / (n1 * n2 + eps), -1.0, 1.0)
    return float(torch.arccos(cos_sim) * (180.0 / math.pi))


def _toggle_lance(model: nn.Module, activate: bool):
    """Set Conv2d_LANCE.activate for all LANCE convs."""
    for m in model.modules():
        if isinstance(m, Conv2d_LANCE):
            m.activate = activate


def _log_grad_alignment_one_batch(model, x, y, device, epoch):
    """
    Compute true vs LANCE gradients on the same (x,y) and log one angle per layer.
    Returns (loss_lance, out_lance) with current grads set to the LANCE backward,
    so the caller can proceed to optimizer.step() without another backward.
    """
    ce = nn.CrossEntropyLoss()

    # Build owner maps once
    _, module_to_params = _param_owner_map(model)

    # -------- True gradients (LANCE deactivated) --------
    _toggle_lance(model, activate=False)
    model.zero_grad(set_to_none=True)
    out_true = model(x)
    loss_true = ce(out_true, y)
    loss_true.backward()

    # Snapshot grads per layer
    true_layer_grads = {}
    for mod_qn, params in module_to_params.items():
        g = _concat_grads(params)
        if g is not None:
            true_layer_grads[mod_qn] = g.clone()

    # -------- LANCE gradients (LANCE activated) --------
    _toggle_lance(model, activate=True)
    model.zero_grad(set_to_none=True)
    out_lance = model(x)
    loss_lance = ce(out_lance, y)
    loss_lance.backward()

    # Compute angles and log to wandb
    for mod_qn, params in module_to_params.items():
        g_lance = _concat_grads(params)
        g_true = true_layer_grads.get(mod_qn, None)
        if g_true is None or g_lance is None:
            continue
        angle_deg = _angle_between(g_true, g_lance)
        # Only log for meaningful layers (non-empty grads). One metric per layer per epoch.
        wandb.log({f"grad_angle_{mod_qn}_": angle_deg}, step=epoch)

    return loss_lance, out_lance


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, log_interval=100, measure_grad_angle=False):
    model.train()
    ce = nn.CrossEntropyLoss()
    run_loss = run_acc = n = 0
    # Keep frozen BNs in eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if all((not p.requires_grad) for p in m.parameters(recurse=True)):
                m.eval()
        if isinstance(m, nn.Dropout):
            m.eval()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if measure_grad_angle and i == 0:
            # Special path: do alignment probe and keep LANCE grads for this step
            loss, out = _log_grad_alignment_one_batch(model, x, y, device, epoch)
        else:
            out = model(x)
            loss = ce(out, y)
            loss.backward()
        # out = model(x)
        # loss = ce(out, y)

        # AMP-safe backward
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        # scaler.update()

        b = x.size(0)
        n += b
        run_loss += loss.item() * b
        run_acc  += (out.argmax(1) == y).float().sum().item()

        if (i + 1) % log_interval == 0:
            print(f"Epoch {epoch} [{i+1}/{len(loader)}] loss={run_loss/n:.4f} acc@1={100.0*run_acc/n:.2f}%")

    return run_loss / n, 100.0 * run_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = total_acc = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = ce(out, y)
        b = x.size(0)
        n += b
        total_loss += loss.item() * b
        total_acc  += (out.argmax(1) == y).float().sum().item()
    return total_loss / n, 100.0 * total_acc / n
