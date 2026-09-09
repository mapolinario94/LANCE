import torch
import torch.nn as nn
from torch.autograd import Function
from custom_layers.hosvd_var import unfolding

__all__ = ['Linear_LANCEViT', 'wrap_linearLANCE']


class Linear_LANCE3_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, S, U_list = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(S, U_list[0], U_list[1], U_list[2], weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        S, U1, U2, U3, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            Z1 = torch.einsum('blo,bk->lok', grad_output, U1) # Shape: B, L, O and B, K1 -> L, O, K1
            Z2 = torch.einsum('abc,lb->acl', S, U2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
            Z3 = torch.einsum('acl,ic->ail', Z2, U3) # Shape: K1, K3, L and I, K3 -> K1, I, L
            grad_weight = torch.einsum('lok,kil->oi', Z1, Z3) # Shape: L, O, K1 and K1, I, L -> O, I

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None


class Linear_LANCEViT(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            rank=0,
            activate=True,
            threshold=0.7):
        super(Linear_LANCEViT, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate

        self.record_feature = False
        self.u0 = None
        self.u1 = None
        self.u2 = None
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_u2 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0
        self.feature_count_u2 = 0

        self.threshold = threshold

        self.record_hardware_metrics = False
        self.memory_MB = 0
        self.memory_BP_MB = 0
        self.flops = 0
        self.flops_BP = 0

    def initialize_subspace(self):
        """Initialize the subspace using the recorded feature covariances."""
        for idx, feature_u in enumerate([self.feature_u0, self.feature_u1, self.feature_u2]):
            if feature_u is None:
                print(f"Warning: feature_u{idx} is None, skipping SVD.")
                continue
            U, S, _ = torch.linalg.svd(feature_u, full_matrices=False)
            s_total = S.sum()
            s_norm = S / s_total
            r = (torch.cumsum(s_norm, dim=0) <= self.threshold).sum().item() + 1
            if r > 100:
                print(f"Warning: large rank ({r}) in Linear_LANCEViT for mode {idx}!")
            U_r = U[:, :r]
            setattr(self, f'u{idx}', U_r)
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_u2 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0
        self.feature_count_u2 = 0

    def forward(self, input):
        if self.record_feature:
            B, L, N = input.shape
            # Accumulate the per-mode input covariances for HOSVD subspace init
            for mode in range(3):
                flat = unfolding(mode, input)
                feature_u = getattr(self, f'feature_u{mode}')
                feature_count = getattr(self, f'feature_count_u{mode}')
                if feature_u is None:
                    feature_u = torch.matmul(flat, flat.T)
                else:
                    feature_u = (feature_count * feature_u + torch.matmul(flat, flat.T)) / (feature_count + flat.size(0))
                setattr(self, f'feature_u{mode}', feature_u)
                setattr(self, f'feature_count_u{mode}', feature_count + flat.size(0))
            y = super().forward(input.detach())
            return y.detach()

        if self.record_hardware_metrics:
            B, N, F = input.shape
            r0 = self.u0.shape[1]
            r1 = self.u1.shape[1]
            r2 = self.u2.shape[1]
            u_list = [self.u0, self.u1, self.u2]
            self.memory_BP_MB = B*N*F*4/(1024**2) # Activation memory
            self.memory_MB = r0*r2*r1*4/(1024**2) + sum(u.numel() for u in u_list if u is not None)*4/(1024**2)
            self.flops_BP = 2 * self.in_features * self.out_features * B * N
            self.flops = 0
            for r in [r0, r1, r2]:
                self.flops += B*N*F*r
            self.flops += B * N * self.in_features * self.out_features + (B*N*self.out_features*r0 + r0*r1*r2*N + r0*r2*F*N + F*self.out_features*N*r0)
            y = super().forward(input.detach())
            return y.detach()

        if self.activate and torch.is_grad_enabled(): # Training mode
            S = input.clone()
            u_list = [self.u0, self.u1, self.u2]
            for i, u in enumerate(u_list):
                if u is None:
                    raise RuntimeError(f"Subspace u{i} is not initialized. Call initialize_subspace() first.")
                S = torch.tensordot(S, u.to(input.device), dims=([0], [0]))
            output = Linear_LANCE3_op.apply(input, self.weight, self.bias, S, u_list)

        else: # activate is False or Validation mode
            output = super().forward(input)
        return output


def wrap_linearLANCE(linear, active, rank, threshold=0.9):
    has_bias = (linear.bias is not None)
    new_linear = Linear_LANCEViT(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        rank=rank,
                        threshold=threshold
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear
