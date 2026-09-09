# ---------------------------------------------------------------------------
# Linear HOSVD activation-compression layers adapted from the ASI reference
# implementation: https://github.com/Le-TrungNguyen/ICML2025-ASI
# Copyright (c) 2025 Le-Trung Nguyen, MIT License (see THIRD_PARTY_LICENSES).
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Function

from .hosvd_var import hosvd_var, hosvd_fixed_rank
###### HOSVD based on explained variance threshold #############
class Linear_HOSVD_var_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, var, k_hosvd, bases, fixed_u, fixed_b = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Perform decomposition
        # if bases is not None:
        #     input = input - input @ bases.to(input.device) @ bases.to(input.device).T
        
        S, U_list = hosvd_var(input, var=var, fixed_u=fixed_u, fixed_b=fixed_b, fixed_mode=1 if fixed_u is not None else None)
        if k_hosvd is not None:
            # Log information for estimating activation memory
            for i, U in enumerate(U_list):
                k_hosvd[i].append(U.shape[1])
            k_hosvd[i + 1].append(input.shape)
            k_hosvd[i + 2].append(output.shape)

        # Save information for backpropagation
        ctx.save_for_backward(S, weight, bias)
        ctx.U_list = U_list
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        S, weight, bias = ctx.saved_tensors

        U_list = ctx.U_list

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if len(U_list) == 4:
                U1, U2, U3, U4 = U_list
                # grad_weight = torch.einsum('bhwc,bhwd->dc', restore_hosvd_4_mode(S, [U1, U2, U3, U4]), grad_output)
                ################ Low rank gradient calculation ################:
                Z1 = torch.einsum("Ba,BHWD->aHWD", U1, grad_output) # Shape: (B, K1) and (B, H, W, D) -> (K1, H, W, D)
                Z2 = torch.einsum("Hb,abcd->aHcd", U2, S) # Shape: (H, K2) and (K1, K2, K3, K4) -> (K1, H, K3, K4)
                Z3 = torch.einsum("Wc,aHWD->aHcD", U3, Z1) # Shape: (W, K3) and (K1, H, W, D) -> (K1, H, K3, D)
                Z4 = torch.einsum("Cd,aHcd->aHCc", U4, Z2) # Shape: (C, K4) and (K1, H, K3, K4) -> (K1, H, C, K3)
                grad_weight = torch.einsum("aHcD,aHCc->DC", Z3, Z4) # Shape: (K1, H, K3, D) and (K1, H, C, K3) -> (D, C)
            elif len(U_list) == 3:
                U1, U2, U3 = U_list
                Z1 = torch.einsum('blo,bk->lok', grad_output, U1) # Shape: B, L, O and B, K1 -> L, O, K1
                Z2 = torch.einsum('abc,lb->acl', S, U2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
                Z3 = torch.einsum('acl,ic->ail', Z2, U3) # Shape: K1, K3, L and I, K3 -> K1, I, L
                grad_weight = torch.einsum('lok,kil->oi', Z1, Z3) # Shape: L, O, K1 and K1, I, L -> O, I
            elif len(U_list) == 2:
                U1, U2 = U_list
                Z1 = torch.einsum('bo,bk->ok', grad_output, U1) # Shape: B, O and B, K1 -> O, K1
                Z2 = torch.einsum('kb,ib->ki', S, U2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
                grad_weight = torch.einsum('ok,ki->oi', Z1, Z2) # Shape: L, O, K1 and K1, I, L -> O, I


            # max_abs_diff = torch.max(torch.abs(grad_weight - grad_weight_)).item()
            # print(f"Maximum absolute difference: {max_abs_diff:.6f}")

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class Linear_HOSVD_var(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            var=0.9,
            k_hosvd = None):
        super(Linear_HOSVD_var, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.var = var
        self.k_hosvd = k_hosvd
        self.bases = None
        self.fixed_u = None
        self.fixed_b = None

    def forward(self, input):
        if self.activate and torch.is_grad_enabled(): # Training mode
            output = Linear_HOSVD_var_op.apply(input, self.weight, self.bias, self.var, self.k_hosvd, self.bases, self.fixed_u, self.fixed_b)
        else: # activate is False or Validation mode
            output = super().forward(input)
        return output
    

###### HOSVD with fixed (per-mode) truncation rank ##############
class Linear_HOSVD_fixed_rank_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, ranks = args

        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        S, U_list = hosvd_fixed_rank(input, ranks)

        ctx.save_for_backward(S, weight, bias)
        ctx.U_list = U_list
        return output

    @staticmethod
    def backward(ctx, grad_output):
        S, weight, bias = ctx.saved_tensors
        U_list = ctx.U_list

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if len(U_list) == 3:
                U1, U2, U3 = U_list
                Z1 = torch.einsum('blo,bk->lok', grad_output, U1)
                Z2 = torch.einsum('abc,lb->acl', S, U2)
                Z3 = torch.einsum('acl,ic->ail', Z2, U3)
                grad_weight = torch.einsum('lok,kil->oi', Z1, Z3)
            elif len(U_list) == 2:
                U1, U2 = U_list
                Z1 = torch.einsum('bo,bk->ok', grad_output, U1)
                Z2 = torch.einsum('kb,ib->ki', S, U2)
                grad_weight = torch.einsum('ok,ki->oi', Z1, Z2)
            elif len(U_list) == 4:
                U1, U2, U3, U4 = U_list
                Z1 = torch.einsum("Ba,BHWD->aHWD", U1, grad_output)
                Z2 = torch.einsum("Hb,abcd->aHcd", U2, S)
                Z3 = torch.einsum("Wc,aHWD->aHcD", U3, Z1)
                Z4 = torch.einsum("Cd,aHcd->aHCc", U4, Z2)
                grad_weight = torch.einsum("aHcD,aHCc->DC", Z3, Z4)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None


class Linear_HOSVD_fixed_rank(nn.Linear):
    """
    Fixed-rank HOSVD-compressed Linear. The per-mode ranks are supplied at
    construction time (typically copied from a LANCE warmup) so that the
    activation footprint exactly matches LANCE for an iso-memory comparison.
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            ranks=None):
        super(Linear_HOSVD_fixed_rank, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.activate = activate
        self.ranks = list(ranks) if ranks is not None else None

        # Hardware metric tracking (matches Linear_LANCEViT's accounting)
        self.memory_MB = 0.0
        self.memory_BP_MB = 0.0
        self.flops = 0.0
        self.flops_BP = 0.0

    def forward(self, input):
        if self.activate and torch.is_grad_enabled():
            if self.ranks is None:
                raise RuntimeError("Linear_HOSVD_fixed_rank.ranks is not set.")
            output = Linear_HOSVD_fixed_rank_op.apply(input, self.weight, self.bias, self.ranks)
            self._update_metrics(input)
        else:
            output = super().forward(input)
        return output

    def _update_metrics(self, input):
        if input.dim() != 3:
            return
        B, L, F = input.shape
        r0 = max(1, min(int(self.ranks[0]), B))
        r1 = max(1, min(int(self.ranks[1]), L))
        r2 = max(1, min(int(self.ranks[2]), F))
        # Memory: core S + U matrices (4 bytes per fp32 element)
        s_elems = r0 * r1 * r2
        u_elems = B * r0 + L * r1 + F * r2
        self.memory_MB = max(self.memory_MB, (s_elems + u_elems) * 4 / (1024 ** 2))
        self.memory_BP_MB = max(self.memory_BP_MB, B * L * F * 4 / (1024 ** 2))
        self.flops_BP = max(self.flops_BP, 2 * self.in_features * self.out_features * B * L)
        # ---- forward HOSVD: full per-mode SVD (conv_hosvd-style accounting) ----
        prod = B * L * F
        svd_cost = (
            max(B, prod // B) ** 2 * min(B, prod // B)
            + max(L, prod // L) ** 2 * min(L, prod // L)
            + max(F, prod // F) ** 2 * min(F, prod // F)
        )
        fw_matmul = B * L * self.in_features * self.out_features
        # ---- backward decomposed gradient chain (3-mode einsum chain) ----
        bw_cost = (
            B * L * self.out_features * r0
            + r0 * r1 * r2 * L
            + r0 * r2 * F * L
            + F * self.out_features * L * r0
        )
        self.flops = max(self.flops, svd_cost + fw_matmul + bw_cost)


def wrap_linearHOSVD_fixed_rank(linear, ranks, active=True):
    has_bias = (linear.bias is not None)
    new_linear = Linear_HOSVD_fixed_rank(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=has_bias,
        activate=active,
        ranks=ranks,
    )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear


def wrap_linearHOSVD_var(linear, active, SVD_var, k_hosvd):
    has_bias = (linear.bias is not None)
    new_linear = Linear_HOSVD_var(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        var=SVD_var,
                        k_hosvd = k_hosvd
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear