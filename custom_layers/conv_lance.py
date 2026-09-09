from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn
from custom_layers.hosvd_var import unfolding
import torch

__all__ = ['Conv2d_LANCE', 'wrap_convLANCE']

class Conv2d_LANCE_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, S, u0, u1, u2, u3= args

        # Perform convolution
        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)

        # Save tensors for backward pass
        ctx.save_for_backward(S, u0, u1, u2, u3, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # Retrieve saved tensors
        S, u0, u1, u2, u3, weight, bias  = ctx.saved_tensors
        B, C, H, W = u0.shape[0], u1.shape[0], u2.shape[0], u3.shape[0]
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs

        # Compute gradient with respect to the input
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input((B,C,H,W), weight, grad_output, stride, padding, dilation, groups)

        # Compute gradient with respect to the weights
        if ctx.needs_input_grad[1]:
            _, _, K_H, K_W = weight.shape # Shape: (C', C, K_H, K_W)
            _, C_prime, H_prime, W_prime = grad_output.shape # Shape: (B, C', H', W')

            # Pad the input
            u2_padded = pad(u2, (0, 0, padding[0], padding[0])) # Shape: (H_padded, K2)
            u3_padded = pad(u3, (0, 0, padding[0], padding[0])) # Shape: (W_padded, K3)
            # Calculate Z1: (conv2d 1x1):
            Z1 = torch.einsum("bk,bchw->kchw", u0, grad_output) # Shape: (B, K0) einsum with (B, C', H', W') -> (B, K0, C', H', W') -> (K0, C', H', W')
            #______________________________________________________________________________________________________________
            # Calculate Z2: (conv2d 1x1):
            Z2 = torch.einsum("abcd,hc->abhd", S, u2_padded) # Shape: (K0, K1, K2, K3) einsum with (H_padded, K2) -> (K0, K1, H_padded, K2, K3) -> (K0, K1, H_padded, K3)
            #______________________________________________________________________________________________________________
            # Calculate Z3: (conv2d 1x1):
            Z3 = torch.einsum("abhd,wd->abhw", Z2, u3_padded) # Shape: (K0, K1, H_padded, K3) einsum with (W_padded, K3) -> (K0, K1, H_padded, W_padded, K3) -> (K0, K1, H_padded, W_padded)
            # ______________________________________________________________________________________________________________
            # Calculate Z4: (conv2d H'xW'):
            if stride == dilation:
                Z4 = conv2d(Z3.permute(1, 0, 2, 3), Z1.permute(1, 0, 2, 3)).permute(1, 0, 2, 3) # Shape: (K1, K0, H_padded, W_padded) conv with (C', K0, H', W') --> (K1, C', K_H, K_W) -> (C', K1, K_H, K_W)
            else:
                Z4 = nn.grad.conv2d_weight(Z3, (C_prime, u1.shape[1], K_H, K_W), Z1, stride=stride, dilation=dilation, groups=1) # Shape (C', K1, K_H, K_W)
            #______________________________________________________________________________________________________________
            # calculate grad_weight
            if groups == C == C_prime: # Depthwise
                grad_weight = torch.einsum("ckhw,ck->ckhw", Z4, u1).sum(dim=1, keepdim=True) # Shape: (C', 1, K_H, K_W)
            elif groups == 1:
                grad_weight = conv2d(Z4, u1.unsqueeze(-1).unsqueeze(-1)) # Shape: (C', K1, K_H, K_W) conv with (C, K1, 1, 1) -> (C', C, K_H, K_W)
            else:
                pass

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None

class Conv2d_LANCE(nn.Conv2d):
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
            threshold=0.9
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

    def initialize_subspace(self):
        """Initialize the subspace using the recorded feature covariances."""
        compression_ratio = []
        for idx, feature_u in enumerate([self.feature_u0, self.feature_u1, self.feature_u2, self.feature_u3]):
            if feature_u is None:
                print(f"Warning: feature_u{idx} is None, skipping SVD.")
                continue
            U, S, _ = torch.linalg.svd(feature_u, full_matrices=False)
            s_total = S.sum()
            s_norm = S / s_total
            r = (torch.cumsum(s_norm, dim=0) <= self.threshold).sum().item() + 1
            if r > 100:
                print(f"Warning: large rank ({r}) in Conv2d_LANCE for mode {idx}!")
            U_r = U[:, :r]
            setattr(self, f'u{idx}', U_r)
            compression_ratio.append(r / feature_u.shape[0])
        self.rank_mode = [getattr(self, f'u{i}').shape[1] if getattr(self, f'u{i}') is not None else None for i in range(4)]
        self.original_shape = [getattr(self, f'feature_u{i}').shape[0] if getattr(self, f'feature_u{i}') is not None else None for i in range(4)]
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_u2 = None
        self.feature_u3 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0
        self.feature_count_u2 = 0
        self.feature_count_u3 = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.record_feature:
            B, C, H, W = x.shape
            # Compute HOSVD of input feature map x
            for mode in range(4):
                flat = unfolding(mode, x)
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
            S = x.clone()
            u_list = [self.u0, self.u1, self.u2, self.u3]
            for i, u in enumerate(u_list):
                if u is None:
                    raise RuntimeError(f"Subspace u{i} is not initialized. Call initialize_subspace() first.")
                S = torch.tensordot(S, u.to(x.device), dims=([0], [0]))
            if self.first_print_memory:
                self.first_print_memory = False
                num_element_com = S.numel() + sum(u.numel() for u in u_list if u is not None)
                num_element_ori = int(x.numel())
                self.original_memory = (num_element_ori*4)/(1024*1024)
                self.compressed_memory = (num_element_com*4)/(1024*1024)
            y = Conv2d_LANCE_op.apply(
                x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, S,
                self.u0.to(x.device), self.u1.to(x.device), self.u2.to(x.device), self.u3.to(x.device)
            )

            if self.first_print_flops:
                self.first_print_flops = False
                # Forward FLOPs
                C_out, C_in, K_H, K_W = self.weight.shape
                B, C, H, W = x.shape
                H_out = y.shape[2]
                W_out = y.shape[3]
                flops_fw_original = B * C_out * H_out * W_out * C_in * K_H * K_W
                self.flops_fw_original = flops_fw_original

                r0 = self.u0.shape[1]
                r1 = self.u1.shape[1]
                r2 = self.u2.shape[1]
                r3 = self.u3.shape[1]
                flops_fw_compressed = B * C_in * H * W * (r0 + r1 + r2 + r3) + self.flops_fw_original
                self.flops_fw = flops_fw_compressed

                # Backward FLOPs
                flops_bw_original = B * C_out * H_out * W_out * C_in * K_H * K_W
                self.flops_bw_original = flops_bw_original

                flops_bw_compressed = r0*C_out*H_out*W_out*B + H*r0*r1*r2*r3 + H*W*r0*r1*r3 + C_out*r0*r1*K_H*K_W*H_out*W_out + C_out*C_in*K_H*K_W*r1
                self.flops_bw = flops_bw_compressed
        else:  # activate is False or Inference mode
            y = super().forward(x)
        return y

    def extra_repr(self) -> str:
        base = super().extra_repr()
        extras = f"activate={self.activate}, rank={self.rank}, threshold={self.threshold}"
        return f"{base}, {extras}"

def wrap_convLANCE(conv, active, rank, threshold=0.9):
    new_conv = Conv2d_LANCE(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         rank=rank,
                         threshold=threshold
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv