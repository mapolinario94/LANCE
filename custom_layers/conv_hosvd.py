# ---------------------------------------------------------------------------
# Conv2d HOSVD activation-compression layer adapted from the ASI reference
# implementation: https://github.com/Le-TrungNguyen/ICML2025-ASI
# Copyright (c) 2025 Le-Trung Nguyen, MIT License (see THIRD_PARTY_LICENSES).
# ---------------------------------------------------------------------------
import torch
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn
from .hosvd_var import hosvd_var

__all__ = ['Conv2d_HOSVD', 'wrap_convHOSVD_var']


###### HOSVD base on explained variance threshold #############
class Conv2d_HOSVD_var_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, S, u0, u1, u2, u3 = args

        # Perform convolution
        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)

        # Perform HOSVD decomposition on the input tensor
        # S, u_list = hosvd_var(input, var=var)
        # u0, u1, u2, u3 = u_list # B, C, H, W

        # if k_hosvd is not None:
        #     for idx in range(4):
        #         k_hosvd[idx].append(u_list[idx].shape[1])
        #     k_hosvd[4].append(input.shape)
        #     k_hosvd[5].append(output.shape)

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
    

class Conv2d_HOSVD(nn.Conv2d):
    """
    Custom Conv2D layer with HOSVD4-based decomposition.
    """
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
            explained_var=1
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        super(Conv2d_HOSVD, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.explained_var = explained_var
        self.first_print_memory = True
        self.compressed_memory = 0.0

        self.first_print_flops = True
        self.flops_fw = 0.0
        self.flops_bw = 0.0


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activate and torch.is_grad_enabled(): # Training mode
            S, u_list = hosvd_var(x, var=self.explained_var)
            u0, u1, u2, u3 = u_list
            y = Conv2d_HOSVD_var_op.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, S,
                u0.to(x.device), u1.to(x.device), u2.to(x.device), u3.to(x.device))
            self.estimate_memory(S, u_list)
            self.estimate_flops(S, x.shape, y)
        else: # activate is False or Inference mode
            y = super().forward(x)
        return y

    def estimate_memory(self, S, u_list):
        num_element_com = S.numel() + sum(u.numel() for u in u_list if u is not None)
        self.compressed_memory = max(self.compressed_memory, (num_element_com*4)/(1024*1024))
    
    def estimate_flops(self, S, x_shape, y):
        B, C, H, W = x_shape
        C_out, C_in, K_H, K_W = self.weight.shape
        H_out = y.shape[2]
        W_out = y.shape[3]

        r0, r1, r2, r3 = S.shape

        term1 = max(B, C * H * W) ** 2 * min(B, C * H * W)
        term2 = max(C, B * H * W) ** 2 * min(C, B * H * W)
        term3 = max(H, B * C * W) ** 2 * min(H, B * C * W)
        term4 = max(W, B * C * H) ** 2 * min(W, B * C * H)

        flops_fw_original = B * C_out * H_out * W_out * C_in * K_H * K_W

        self.flops_fw = max(self.flops_fw, term1 + term2 + term3 + term4 + flops_fw_original)

        flops_bw_compressed = r0*C_out*H_out*W_out*B + H*r0*r1*r2*r3 + H*W*r0*r1*r3 + C_out*r0*r1*K_H*K_W*H_out*W_out + C_out*C_in*K_H*K_W*r1
        self.flops_bw = max(self.flops_bw, flops_bw_compressed)
        

def wrap_convHOSVD(conv, active, explained_var):
    new_conv = Conv2d_HOSVD(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         explained_var=explained_var
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv