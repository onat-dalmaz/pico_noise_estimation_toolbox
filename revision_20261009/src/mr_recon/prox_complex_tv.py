"""Complex-valued (vector) total-variation proximal operator.

``mr_recon.prox.TV`` applies TV to the magnitude of a complex image and
re-attaches the phase.  That map is not differentiable where the magnitude
is small (the phase derivative scales as 1/|x|), so a first-order noise
model of a reconstruction built on it is invalid at low-signal voxels once
the proximal operator actually acts (see scripts/cs_fista_stopping_check.py).

This class applies isotropic TV to the complex image directly, treating the
real and imaginary parts as two channels of one vector field (the standard
complex TV of Block et al. 2007 / Knoll et al. 2011):

    prox_{alpha*lamda*TV}(x) = argmin_u 0.5*||u - x||^2 + alpha*lamda*TV(u),
    TV(u) = sum_r sqrt( sum_{c in {Re,Im}} sum_{d in {x,y}} (D_d u_c)(r)^2 ),

solved by Chambolle's dual projection algorithm with a fixed number of
iterations (no data-dependent exit) and the classical step tau <= 1/8 on
the scaled problem.  The map is smooth wherever the gradient of u is
nonzero, and its Jacobian-vector products are well behaved.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grad2(x: torch.Tensor) -> torch.Tensor:
    """Forward differences with zero padding on the last two dims: (..., H, W) -> (2, ..., H, W)."""
    gx = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    gy = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    return torch.stack((gx, gy), dim=0)


def _div2(p: torch.Tensor) -> torch.Tensor:
    """Exact negative adjoint of _grad2 (so that <grad x, p> = -<x, div p>): (2, ..., H, W) -> (..., H, W)."""
    px, py = p[0], p[1]
    dx = torch.cat((px[..., :1, :], px[..., 1:-1, :] - px[..., :-2, :], -px[..., -2:-1, :]), dim=-2)
    dy = torch.cat((py[..., :, :1], py[..., :, 1:-1] - py[..., :, :-2], -py[..., :, -2:-1]), dim=-1)
    return dx + dy


class ComplexTV(nn.Module):
    def __init__(self, im_size: Tuple[int, int], lamda: float, max_iter: int = 50,
                 tol: float = 0.0, tau: Optional[float] = None):
        super().__init__()
        assert len(im_size) == 2, "ComplexTV: 2D only"
        self.im_size = tuple(im_size)
        self.lamda = float(lamda)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.tau = tau  # None: 1/8 (Chambolle's bound for the scaled dual problem)

    def forward(self, input: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        is_complex = torch.is_complex(input)
        x = torch.stack((input.real, input.imag), dim=0) if is_complex else input.unsqueeze(0)
        w = alpha * self.lamda
        # dual problem on the scaled image x / w: p <- P_ball( p + tau * grad( div p - x / w ) )
        tau = 0.125 if self.tau is None else float(self.tau)
        p = torch.zeros((2,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        for _ in range(self.max_iter):
            p_old = p
            g = _grad2(_div2(p) - x / w)
            q = p + tau * g
            nrm = torch.sqrt((q * q).sum(dim=(0, 1), keepdim=True))
            p = q / torch.clamp(nrm, min=1.0)
            if self.tol > 0 and (p - p_old).abs().max() < self.tol:
                break
        u = x - w * _div2(p)
        return torch.complex(u[0], u[1]) if is_complex else u[0]
