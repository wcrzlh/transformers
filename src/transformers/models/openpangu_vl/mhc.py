import torch
from torch import nn

class mHCModule(nn.Module):
    def __init__(
            self,
            config: OpenPanguV2Config,
            merge_layer_only_pre=False,
    ):
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.merge_layer_only_pre = merge_layer_only_pre

        if not self.merge_layer_only_pre:
            phi_output_hiden_size = (self.num_stream + 2) * self.num_stream
            self.branch_alpha_post = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            self.branch_alpha_res = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            self.branch_beta_post = nn.Parameter(torch.empty(self.num_stream, dtype=torch.bfloat16))
            self.branch_beta_res = nn.Parameter(torch.empty(self.num_stream * self.num_stream, dtype=torch.bfloat16))
        else:
            phi_output_hiden_size = self.num_stream

        self.branch_alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
        self.branch_beta_pre = nn.Parameter(torch.empty(self.num_stream, dtype=torch.bfloat16))
        self.phi = nn.Linear(
            self.hidden_size * self.num_stream,
            phi_output_hiden_size,
            bias=False,
            dtype=torch.bfloat16
        )
        self.mhc_use_gamma = config.mhc_use_gamma
        self.hc_eps = 1e-6
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        if self.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(torch.empty(self.hidden_size * self.num_stream, dtype=torch.bfloat16))


    def hc_pre(self, x):
        dtype = x.dtype
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        if self.mhc_use_gamma:
            weight = self.phi(x * rsqrt * self.norm_gamma.unsqueeze(0))
        else:
            weight = self.phi(x) * rsqrt

        h_pre, h_post, h_res = self.hc_split_sinkhorn_torch(weight)

        y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.num_stream, -1)), dim=2)
        return y.to(dtype), h_post, h_res

    def hc_post(self, x, residual, h_post, h_res):
        if self.merge_layer_only_pre:
            return x

        y = h_post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            h_res.unsqueeze(-1) * residual.unflatten(dim=-1, sizes=(self.num_stream, -1)).unsqueeze(-2), dim=-3
        )
        return y.view(residual.shape).type_as(x)

    def hc_split_sinkhorn_torch(self, weight):
        if not self.merge_layer_only_pre:
            h_pre, h_post, h_res = weight.split(
                [self.num_stream, self.num_stream, self.num_stream * self.num_stream], dim=-1
            )
            h_post = 2 * torch.sigmoid(h_post * self.branch_alpha_post + self.branch_beta_post)
            h_res = h_res.unflatten(-1, (self.num_stream, self.num_stream))
            h_res = h_res * self.branch_alpha_res + self.branch_beta_res.view(self.num_stream, self.num_stream)
            h_res = self.sinkhorn_knopps(h_res, self.mhc_recur_norm, self.hc_eps)
        else:
            h_pre = weight
            h_post = None
            h_res = None
        h_pre = torch.sigmoid(h_pre * self.branch_alpha_pre + self.branch_beta_pre)
        return h_pre, h_post, h_res

    def sinkhorn_knopps(self, h_res, sinkhorn_iters, eps):
        h_res = h_res.softmax(-1) + eps
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
        for _ in range(sinkhorn_iters - 1):
            row_sum = h_res.sum(-1, keepdim=True)
            h_res = h_res / (row_sum + eps)
            col_sum = h_res.sum(-2, keepdim=True)
            h_res = h_res / (col_sum + eps)
        return h_res