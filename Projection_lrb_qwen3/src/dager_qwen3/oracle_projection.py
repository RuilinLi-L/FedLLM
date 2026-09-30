"""Exact realized column-side Projection-LRB operators for Qwen nn.Linear.

No seed guesses, GPT-2 orientation rules, or hidden access to original features.
The signed operator is D U P D: downsample and upsample are generally NOT
orthogonal projections. Both operators are taken from the defense implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from utils.lrb_defense import apply_lrb_defense, _cached_rademacher, _adaptive_avg_pool_along_dim_manual
from utils.lrb_presets import apply_lrb_preset


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class DefendedQwenUpdate:
    gradients: tuple[torch.Tensor | None, ...]
    parameter_names: tuple[str, ...]
    layer_info: tuple[dict[str, Any], ...]

    def state_for(self, name: str) -> dict[str, Any]:
        index = self.parameter_names.index(name)
        state = self.layer_info[index]
        if state['idx'] != index or state['name'] != name or not state['active']:
            raise ValueError(f'Invalid canonical transform state for {name}')
        gradient = self.gradients[index]
        if gradient is None or tuple(state['shape']) != tuple(gradient.shape):
            raise ValueError(f'Gradient/state shape mismatch for {name}')
        return dict(state)


def defend_canonical_gradients(gradients, names, *, preset: str, rho: float, seed: int = 700001):
    gradients, names = tuple(gradients), tuple(names)
    if len(gradients) != len(names) or len(set(names)) != len(names):
        raise ValueError('Canonical names/gradient slots must be unique and aligned')
    if preset not in ('proj_uniform', 'proj_only', 'identity_lrb') or not 0 < rho <= 1:
        raise ValueError('v2 supports noise-free projection presets and 0 < rho <= 1 only')
    args = SimpleNamespace(defense='lrb', defense_lrb_preset=preset,
        defense_lrb_keep_ratio_sensitive=rho, defense_lrb_seed=seed,
        defense_lrb_seed_mode='static', rng_seed=seed)
    apply_lrb_preset(args)
    defended = apply_lrb_defense(gradients, args, layer_names=list(names))
    states = tuple(dict(s) for s in args.lrb_defense_layer_info)
    if len(defended) != len(gradients) or len(states) != len(names):
        raise ValueError('Defense changed canonical tuple length')
    for index, raw in enumerate(gradients):
        if (raw is None) != (defended[index] is None):
            raise ValueError('Defense changed canonical None slots')
    return DefendedQwenUpdate(tuple(defended), names, states)


class QwenColumnTransform:
    """Apply h R^T along the final feature axis, using the actual defense state."""
    def __init__(self, state: dict[str, Any]):
        self.state = dict(state)
        if not state.get('active') or len(state.get('shape', ())) != 2:
            raise ValueError('Oracle requires an active, two-dimensional canonical tensor')
        if state.get('noise_scale') != 0 or state.get('projection_mode') != 'signed_pool':
            raise ValueError('This oracle supports the registered noise-free signed_pool protocol only')
        self.rows, self.width = map(int, state['shape'])
        self.rho = float(state['keep_ratio'])
        if not 0 < self.rho <= 1:
            raise ValueError('Invalid realized keep ratio')
        self.device = torch.device(state['projection_device'])
        self.q = self.width if self.rho >= .999 else max(1, round(self.width * self.rho))
        self.signs = None
        if self.rho < .999:
            # Column signs are the SECOND RNG draw, after a (d_out, 1) draw.
            self.signs = _cached_rademacher((1, self.width), device=self.device,
                seed=int(state['projection_seed']), prior_shapes=((self.rows, 1),)).clone()

    def __call__(self, representations: torch.Tensor) -> torch.Tensor:
        if representations.shape[-1] != self.width:
            raise ValueError('Qwen feature transform received the wrong axis/width')
        if representations.device != self.device and not (
            representations.device.type == self.device.type == 'cuda'
            and self.device.index is None
        ):
            raise ValueError('Candidate device differs from realized defense device')
        x = representations.float()
        if self.rho >= .999:
            return x.clone()
        shape = x.shape
        signed = x.reshape(-1, self.width) * self.signs
        pooled = _adaptive_avg_pool_along_dim_manual(signed, self.q, dim=1)
        restored = F.interpolate(pooled[:, None, :], size=self.width,
                                mode='linear', align_corners=False)[:, 0, :]
        return (restored * self.signs).reshape(shape)

    def metadata(self, *, verify_rank: bool = False) -> dict[str, Any]:
        result = dict(self.state)
        result.update(feature_axis=1, feature_width=self.width, projected_q=self.q,
                      column_sign_sha256=None if self.signs is None else tensor_sha256(self.signs))
        if verify_rank:
            # Verify P and U separately, avoiding an enormous d x d SVD.
            if self.q == self.width:
                rank_p = rank_u = self.width
            else:
                eye = torch.eye(self.width, device=self.device, dtype=torch.float32)
                p = _adaptive_avg_pool_along_dim_manual(eye, self.q, dim=0)
                u = F.interpolate(torch.eye(self.q, device=self.device)[None],
                                  size=self.width, mode='linear', align_corners=False)[0].T
                rank_p = int(torch.linalg.matrix_rank(p, atol=1e-6, rtol=0))
                rank_u = int(torch.linalg.matrix_rank(u, atol=1e-6, rtol=0))
            result.update(pool_rank=rank_p, interpolation_rank=rank_u,
                          operator_rank=self.q if rank_p == rank_u == self.q else None)
            if result['operator_rank'] is None:
                raise ValueError('Feature operator rank is not the nominal q')
        return result
