import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormSuper(torch.nn.LayerNorm):

    def __init__(self, super_embed_dim: int) -> None:
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return None


class LayerNormSub(torch.nn.Module):

    def __init__(self,
                 layer: torch.nn.LayerNorm,
                 embed_choice: int,
                 super_emb: int,
                 pre_norm: bool,
                 before: bool = False,
                 after: bool = False) -> None:
        super(LayerNormSub, self).__init__()
        self.sampled_in_dim = embed_choice
        self.super_embed_dim = super_emb
        self.normalize_before = pre_norm
        self.before = before
        self.after = after
        self.layer = layer

    def maybe_layer_norm(self,
                         norm: torch.nn.LayerNorm,
                         x: torch.Tensor,
                         before: bool = False,
                         after: bool = False) -> torch.Tensor:
        weight = norm.weight[:self.sampled_in_dim]
        bias = norm.bias[:self.sampled_in_dim]
        assert before ^ after
        if after ^ self.normalize_before:
            return F.layer_norm(x, (self.sampled_in_dim, ),
                                weight=weight,
                                bias=bias,
                                eps=norm.eps)
        else:
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maybe_layer_norm(self.layer,
                                  x[:, :, :self.sampled_in_dim],
                                  before=self.before,
                                  after=self.after)
        if x.shape[-1] != self.super_embed_dim:
            x = F.pad(x, (0, self.super_embed_dim - x.shape[-1]), "constant",
                      0)
        return x
