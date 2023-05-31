import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = "cuda")

        # TODO (1.4): Sample points from z values
        N = ray_bundle.origins.shape[0]
        D = z_vals.shape[0]
        origins, directions = ray_bundle.origins.unsqueeze(1).repeat(1, D, 1), ray_bundle.directions.unsqueeze(1).repeat(1, D, 1)
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1)
        sample_points = origins + z_vals*directions

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}