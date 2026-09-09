from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn
from .hosvd_var import unfolding
import torch
from torch.nn import functional as F
import logging

__all__ = ['Conv2d_LANCE_CL']


class Conv2d_LANCE_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, S, u0, u1, u2= args

        # Perform convolution
        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)
        B,C,H,W = input.shape
        # Save tensors for backward pass
        ctx.save_for_backward(S, u0, u1, u2, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.B, ctx.C, ctx.H, ctx.W = B,C,H,W

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # Retrieve saved tensors
        S, u0, u1, u2, weight, bias  = ctx.saved_tensors
        # B, O, L = u0.shape[0], u1.shape[0], u2.shape[0]
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        B, C, H, W = ctx.B, ctx.C, ctx.H, ctx.W 

        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs

        # Compute gradient with respect to the input
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input((B,C,H,W), weight, grad_output, stride, padding, dilation, groups)

        # Compute gradient with respect to the weights
        if ctx.needs_input_grad[1]:
            _, _, K_H, K_W = weight.shape # Shape: (C', C, K_H, K_W)
            _, C_prime, H_prime, W_prime = grad_output.shape # Shape: (B, C', H', W')

            S = S.permute(0, 2, 1)
            grad_output = grad_output.view(B, C_prime, -1)
            grad_output = grad_output.permute(0, 2, 1)

            Z1 = torch.einsum('blo,bk->lok', grad_output, u0) # Shape: B, L, O and B, K1 -> L, O, K1
            Z2 = torch.einsum('abc,lb->acl', S, u2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
            Z3 = torch.einsum('acl,ic->ail', Z2, u1) # Shape: K1, K3, L and I, K3 -> K1, I, L
            grad_weight = torch.einsum('lok,kil->oi', Z1, Z3) # Shape: L, O, K1 and K1, I, L -> O, I
            grad_weight = grad_weight.view_as(weight)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None

class Conv2d_LANCE_CL(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            rank=1,
            threshold=0.9,
            threshold_cl=0.9
    ) -> None:
        # Ensure kernel_size, padding, dilation are tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
            padding_mode='zeros',
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.rank = rank
        self.reuse_U = False
        self.record_feature = False
        self.u0 = None
        self.u1 = None
        self.u2 = None
        self.u3 = None
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_u2 = None
        self.feature_u3 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0
        self.feature_count_u2 = 0
        self.feature_count_u3 = 0
        self.threshold = threshold
        self.first_print_memory = True
        self.rank_mode = []
        self.original_shape = []
        self.compressed_memory = None
        self.original_memory = None

        self.first_print_flops = True
        self.flops_fw = None
        self.flops_bw = None
        self.flops_bw_original = None
        self.flops_fw_original = None

        self.original_act_memory = None

        # CL variables
        self.protected_features = None
        self.threshold_cl = threshold_cl

    def unfold_inputs(self, x):
        N, C_in, H, W = x.shape
        kH, kW       = self.kernel_size
        sH, sW       = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        dH, dW       = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        pH, pW       = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        G            = self.groups
        C_out        = self.out_channels
        Cg_in        = self.in_channels // G
        Cg_out       = self.out_channels // G
        device, dtype= x.device, x.dtype

        # Handle padding modes:
        # - If zeros (the usual), let F.unfold handle padding directly.
        # - Otherwise, pad explicitly first, then unfold with padding=0.
        if getattr(self, "padding_mode", "zeros") != "zeros":
            # torch.nn.Conv2d uses (left,right,top,bottom) for F.pad
            x_eff = F.pad(x, (pW, pW, pH, pH), mode=self.padding_mode)
            pad_for_unfold = (0, 0)
        else:
            x_eff = x
            pad_for_unfold = (pH, pW)

        # Unfold to (N, C_in * kH * kW, L) where L = H_out * W_out
        x_unfold = F.unfold(
            x_eff, kernel_size=(kH, kW),
            dilation=(dH, dW),
            padding=pad_for_unfold,
            stride=(sH, sW),
        )  # shape: (N, C_in * kH * kW, L)
        N, CK, L = x_unfold.shape
        return x_unfold

    def initialize_subspace(self):
        """Initialize the subspace using the recorded feature covariances."""
        compression_ratio = []
        for idx, feature_u in enumerate([self.feature_u0, self.feature_u1, self.feature_u2]):
            if feature_u is None:
                logging.info(f"Warning: feature_u{idx} is None, skipping SVD.")
                continue
            U, S, _ = torch.linalg.svd(feature_u, full_matrices=False)
            s_total = S.sum()
            if self.protected_features is not None and idx == 1:
                U, S, _ = torch.linalg.svd(feature_u.T - self.protected_features @ self.protected_features.T @ feature_u.T, full_matrices=False)
                s_residual = (S).sum()
                s_norm = (S) / s_total
                accumulated_s =(s_total - s_residual) / s_total
                r = 0
                for i in range(len(s_norm)):
                    if accumulated_s <= self.threshold_cl:
                        accumulated_s += s_norm[i]
                        r += 1
                    else:
                        break
                if self.protected_features.size(0) <= 80:
                    r = self.protected_features.size(0) - self.protected_features.size(1)
                r = max(r, 1)
                if r == 0:
                    continue
                else:
                    U_r = U[:, :r]
            else:
                s_norm = S / s_total
                r = (torch.cumsum(s_norm, dim=0) <= self.threshold).sum().item()
                r = max(r, 1)
                if r > 100:
                    print(f"Warning: large rank ({r}) in Conv2d_LANCE for mode {idx}!")
                U_r = U[:, :r]
            setattr(self, f'u{idx}', U_r)
            compression_ratio.append(r / feature_u.shape[0])
        self.rank_mode = [getattr(self, f'u{i}').shape[1] if getattr(self, f'u{i}') is not None else None for i in range(3)]
        self.original_shape = [getattr(self, f'feature_u{i}').shape[0] if getattr(self, f'feature_u{i}') is not None else None for i in range(3)]
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_u2 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0
        self.feature_count_u2 = 0

    def initialize_protected_subspace(self):
        """Initialize the subspace using the recorded feature covariances."""
        compression_ratio = []
        feature_u = self.feature_u1
        if feature_u is None:
            raise RuntimeError("feature_u1 is None; record activations before initializing the protected subspace.")
        U, S, _ = torch.linalg.svd(feature_u, full_matrices=False)
        s_total = (S).sum()
        if self.protected_features is not None:
            U, S, _ = torch.linalg.svd(feature_u.T - self.protected_features @ self.protected_features.T @ feature_u.T, full_matrices=False)
            s_residual = (S).sum()
            s_norm = (S) / s_total
            accumulated_s =(s_total - s_residual) / s_total
            r = 0
            for i in range(len(s_norm)):
                if accumulated_s <= self.threshold_cl:
                    accumulated_s += s_norm[i]
                    r += 1
                else:
                    break
            if r == 0:
                U_r = self.protected_features
            else:
                U_r = U[:, :r]
                U_r = torch.cat([self.protected_features, U_r], dim=1)
        else:
            s_norm = (S) / s_total
            r = (torch.cumsum(s_norm, dim=0) <= self.threshold_cl).sum().item() + 1
            if r / feature_u.shape[0] > 0.9:
                print(f"Warning: large rank ({r}) in Conv2d_LANCE_CL protected subspace!")
            U_r = U[:, :r]
        setattr(self, 'protected_features', U_r)
        compression_ratio.append(r / feature_u.shape[0])
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_u2 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0
        self.feature_count_u2 = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.record_feature:
            if self.original_act_memory is None:
                self.original_act_memory = (x.numel()*4)/(1024*1024)
            x_unfold = self.unfold_inputs(x)
            B, C, H, W = x.shape
            # Compute HOSVD of input feature map x
            for mode in range(3):
                flat = unfolding(mode, x_unfold)
                feature_u = getattr(self, f'feature_u{mode}')
                feature_count = getattr(self, f'feature_count_u{mode}')
                if feature_u is None:
                    feature_u = torch.matmul(flat, flat.T)
                else:
                    feature_u = (feature_count * feature_u + torch.matmul(flat, flat.T)) / (feature_count + flat.size(0))
                setattr(self, f'feature_u{mode}', feature_u)
                setattr(self, f'feature_count_u{mode}', feature_count + flat.size(0))
            y = super().forward(x)
            return y

        if self.activate and torch.is_grad_enabled():  # Training mode
            with torch.no_grad():
                self.original_act_memory = None
                x_unfold = self.unfold_inputs(x)
                S = x_unfold.clone()
                u_list = [self.u0, self.u1, self.u2]
                for i, u in enumerate(u_list):
                    if u is None:
                        raise RuntimeError(f"Subspace u{i} is not initialized. Call initialize_subspace() first.")
                    S = torch.tensordot(S, u.to(x.device), dims=([0], [0]))
            y = Conv2d_LANCE_op.apply(
                x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, S,
                self.u0.to(x.device), self.u1.to(x.device), self.u2.to(x.device)
            )
        else:  # activate is False or Inference mode
            y = super().forward(x)
        return y

    def extra_repr(self) -> str:
        base = super().extra_repr()
        extras = f"activate={self.activate}, rank={self.rank}, threshold={self.threshold}"
        return f"{base}, {extras}"
