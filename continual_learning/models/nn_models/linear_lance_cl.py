from torch.autograd import Function
import torch.nn as nn
from .hosvd_var import unfolding
import torch
import logging

__all__ = ['Linear_LANCE_CL']


class Linear_LANCE2_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, S, U_list = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(S, U_list[0], U_list[1], weight, bias)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        S, U1, U2, weight, bias = ctx.saved_tensors
    
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            Z1 = torch.einsum('bo,bk->ok', grad_output, U1) # Shape: B, O and B, K1 -> O, K1
            Z2 = torch.einsum('kb,ib->ki', S, U2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
            grad_weight = torch.einsum('ok,ki->oi', Z1, Z2) # Shape: L, O, K1 and K1, I, L -> O, I

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None

class Linear_LANCE_CL(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            rank=0,
            activate=True,
            threshold=0.7,
            threshold_cl=0.95):
        super(Linear_LANCE_CL, self).__init__(
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
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0
        self.threshold = threshold

        self.original_act_memory = None

        # CL variables
        self.protected_features = None
        self.threshold_cl = threshold_cl

    def initialize_subspace(self):
        """Initialize the subspace using the recorded feature covariances."""
        compression_ratio = []
        for idx, feature_u in enumerate([self.feature_u0, self.feature_u1]):
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
                if r == 0:
                    continue
                else:
                    U_r = U[:, :r]
            else:
                s_norm = S / s_total
                r = (torch.cumsum(s_norm, dim=0) <= self.threshold).sum().item() + 1
                if r > 100:
                    logging.info(f"Warning: large rank ({r}) in Linear_LANCE for mode {idx}!")
                U_r = U[:, :r]
            setattr(self, f'u{idx}', U_r)
            compression_ratio.append(r / feature_u.shape[0])
        self.rank_mode = [getattr(self, f'u{i}').shape[1] if getattr(self, f'u{i}') is not None else None for i in range(2)]
        self.original_shape = [getattr(self, f'feature_u{i}').shape[0] if getattr(self, f'feature_u{i}') is not None else None for i in range(2)]
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0

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
                print(f"Warning: large rank ({r}) in Linear_LANCE_CL protected subspace!")
            U_r = U[:, :r]
        setattr(self, 'protected_features', U_r)
        compression_ratio.append(r / feature_u.shape[0])
        self.feature_u0 = None
        self.feature_u1 = None
        self.feature_count_u0 = 0
        self.feature_count_u1 = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.record_feature:
            if self.original_act_memory is None:
                self.original_act_memory = (x.numel()*4)/(1024*1024)
            # Accumulate the per-mode input covariances for HOSVD subspace init
            for mode in range(2):
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
            self.original_act_memory = None
            with torch.no_grad():
                S = x
                u_list = [self.u0, self.u1]
                for i, u in enumerate(u_list):
                    if u is None:
                        raise RuntimeError(f"Subspace u{i} is not initialized. Call initialize_subspace() first.")
                    S = torch.tensordot(S, u.to(x.device), dims=([0], [0]))
            y = Linear_LANCE2_op.apply(
                x, self.weight, self.bias,  S,
                [self.u0.to(x.device), self.u1.to(x.device)]
            )
        else:  # activate is False or Inference mode
            y = super().forward(x)
        return y

    def extra_repr(self) -> str:
        base = super().extra_repr()
        extras = f"activate={self.activate}, rank={self.rank}, threshold={self.threshold}"
        return f"{base}, {extras}"

