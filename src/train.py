"""
Training / adaptation utilities.
Implements:
  • SWIFTADAPT (one-pass LAS)
  • FLASHADAPT (two-pass)
  • TENT
  • CoTTA
  • OSHA (online SGD on all params)
  • Source (no adaptation)
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------------------------------------------------------#
#                              utils/helpers                                   #
# -----------------------------------------------------------------------------#

def _outer(u: torch.Tensor, v: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Rank-1 tensor with required shape (conv / linear)."""
    out_dim, in_dim = u.numel(), v.numel()
    base = u[:, None] * v[None, :]
    if len(target_shape) == 2:
        return base
    if len(target_shape) == 4:  # conv weight
        kH, kW = target_shape[2:]
        return base[:, :, None, None].expand(out_dim, in_dim, kH, kW)
    raise ValueError("unsupported weight shape")


# -----------------------------------------------------------------------------#
#                  Hyper-network producing rank-1 factors                       #
# -----------------------------------------------------------------------------#

class Rank1Hyper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stats_len: int = 3):
        super().__init__()
        hidden = max(16, stats_len * 4)
        self.mlp = nn.Sequential(
            nn.Linear(stats_len, hidden), nn.ReLU(True), nn.Linear(hidden, out_dim + in_dim)
        )

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vec = self.mlp(s)
        return vec[: -s.numel()], vec[-s.numel() :]


# -----------------------------------------------------------------------------#
#                           SWIFTADAPT main class                               #
# -----------------------------------------------------------------------------#

class SwiftAdapt(nn.Module):
    """Full SWIFTADAPT implementation with Look-Ahead Statistics."""

    def __init__(self, backbone: nn.Module, las_decay: float = 0.1, device: str = "cuda"):
        super().__init__()
        self.backbone = backbone.to(device).eval()
        self.blocks: List[nn.Module] = [
            m
            for m in self.backbone.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.requires_grad
        ]
        self.device = device
        self.state = [torch.zeros(3, device=device) for _ in self.blocks]
        self.hyper_nets = nn.ModuleList()
        for blk in self.blocks:
            w = blk.weight
            out_dim, in_dim = (w.shape[0], w.shape[1]) if w.dim() >= 2 else w.shape
            self.hyper_nets.append(Rank1Hyper(in_dim, out_dim))
        self.las_decay = las_decay
        self._delta_cache: List[torch.Tensor | None] = [None] * len(self.blocks)
        self._install_hooks()
        self.optimizer = torch.optim.Adam(self.hyper_nets.parameters(), lr=1e-4)

    # ------------------------------------------------------------------ #
    def _install_hooks(self):
        def _las_hook(_, __, out, idx: int):
            a = out.detach()
            mu, var, en = a.mean(), a.var(False), (a ** 2).mean()
            new = torch.tensor([mu, var, en], device=self.device)
            self.state[idx] = (1 - self.las_decay) * self.state[idx] + self.las_decay * new

        for i, blk in enumerate(self.blocks):
            blk.register_forward_hook(lambda m, x, y, j=i: _las_hook(m, x, y, j))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _apply(self):
        for i, blk in enumerate(self.blocks):
            u, v = self.hyper_nets[i](self.state[i])
            delta = _outer(u * 1e-3, v * 1e-3, blk.weight.shape)
            blk.weight.add_(delta)
            self._delta_cache[i] = delta

    @torch.no_grad()
    def _restore(self):
        for i, blk in enumerate(self.blocks):
            if self._delta_cache[i] is not None:
                blk.weight.sub_(self._delta_cache[i])
                self._delta_cache[i] = None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        self._apply()
        y = self.backbone(x)
        self._restore()
        return y

    # episodic fine-tuning (entropy)
    def unsupervised_finetune(self, imgs: torch.Tensor, steps: int = 4):
        self.train()
        for _ in range(steps):
            p = self(imgs)
            q = torch.softmax(p, 1)
            loss = (q * torch.log(q + 1e-8)).sum(1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.eval()


# -----------------------------------------------------------------------------#
#                               Baselines                                      #
# -----------------------------------------------------------------------------#

class SourceOnly:
    def __init__(self, model: nn.Module, device="cuda"):
        self.model = model.to(device).eval()

    @torch.no_grad()
    def predict(self, x):
        return self.model(x)


class FlashAdapt(SwiftAdapt):
    @torch.no_grad()
    def forward(self, x):
        _ = self.backbone(x)  # dry run
        self._apply()
        y = self.backbone(x)
        self._restore()
        return y


class Tent:
    def __init__(self, model: nn.Module, lr=3e-4, device="cuda"):
        self.model = model.to(device).train()
        self.opt = torch.optim.Adam(
            [p for n, p in model.named_parameters() if "bn" in n.lower()], lr=lr
        )

    def predict(self, x):
        p = self.model(x)
        q = torch.softmax(p, 1)
        loss = (q * torch.log(q + 1e-8)).sum(1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return p


class CoTTA(Tent):
    def __init__(self, model: nn.Module, lr=3e-4, ema=0.999, prob=0.01, device="cuda"):
        super().__init__(model, lr, device)
        self.ema, self.prob = ema, prob
        self.anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _update_ema(self):
        for k, v in self.model.state_dict().items():
            self.anchor[k].mul_(self.ema).add_(v * (1 - self.ema))

    def _restore_some(self):
        for n, p in self.model.named_parameters():
            if torch.rand(1).item() < self.prob:
                p.data.copy_(self.anchor[n])

    def predict(self, x):
        y = super().predict(x)
        self._update_ema()
        self._restore_some()
        return y


class OSHA:
    def __init__(self, model: nn.Module, lr=1e-5, device="cuda"):
        self.model = model.to(device).train()
        self.opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    def predict(self, x):
        p = self.model(x)
        q = torch.softmax(p, 1)
        loss = (q * torch.log(q + 1e-8)).sum(1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return p