import numpy as np
import torch
from ray_utils import RayBundle

def phong(
    normals,
    view_dirs, 
    light_dir,
    params,
    colors
):
    # TODO: Implement a simplified version Phong shading
    # Inputs:
    #   normals: (N x d, 3) tensor of surface normals
    #   view_dirs: (N x d, 3) tensor of view directions
    #   light_dir: (3,) tensor of light direction
    #   params: dict of Phong parameters
    #   colors: (N x d, 3) tensor of colors
    # Outputs:
    #   illumination: (N x d, 3) tensor of shaded colors
    #
    # Note: You can use torch.clamp to clamp the dot products to [0, 1]
    # Assume the ambient light (i_a) is of unit intensity 
    # While the general Phong model allows rerendering with multiple lights, 
    # here we only implement a single directional light source of unit intensity
    ka = params['ka']
    kd = params['kd']
    ks = params['ks']
    n = params['n']

    light_dir = light_dir / torch.norm(light_dir)
    # light_dir = light_dir.repeat(normals.shape[0], 1)

    # # h = torch.nn.functional.normalize(light_dir+view_dirs)

    # dot_prod_d, dot_prod_s = torch.ones((1, 3), device="cuda"), torch.ones((1, 3), device="cuda")
    # for i in range(3):
    #     dot_prod_d[:, i] = torch.clamp(torch.dot(light_dir[i], normals[i]), min=0, max=1)
    #     # dot_prod_s[:, i] = torch.clamp(torch.dot(h[i], normals[i]), min=0, max=1)
    #     h = 2 * torch.clamp(torch.dot(light_dir[i], normals[i]), min=0, max=1) * normals[i] - light_dir[i]
    #     dot_prod_s[:, i] = torch.clamp(torch.dot(h, view_dirs[i]), min=0, max=1)
    # illumination = ka * torch.ones((1, 3), device="cuda")+ kd * dot_prod_d * colors + ks * (dot_prod_s)**n * colors
    # light_dir, normals, view_dirs = light_dir.reshape((-1, )), normals.reshape((-1, )), view_dirs.reshape((-1, ))
    # # h = 2 * torch.clamp(torch.dot(light_dir, normals), min=0, max=1) * normals - light_dir
    # h = torch.nn.functional.normalize(light_dir+view_dirs)
    # dot_prod_d = torch.clamp(torch.dot(light_dir, normals), min=0, max=1)
    # dot_prod_s = torch.clamp(torch.dot(h, view_dirs), min=0, max=1)
    normals = normals/torch.norm(normals, dim=-1).unsqueeze(-1)
    view_dirs = view_dirs/torch.norm(view_dirs, dim=-1).unsqueeze(-1)

    R = 2 * torch.clamp(torch.matmul(normals.unsqueeze(-1).permute(0, 2, 1), light_dir.permute(1, 0)), min=0, max=1).squeeze(-1) * normals - light_dir
    R = R/torch.norm(R, dim=-1).unsqueeze(-1)
    dot_prod_d = torch.clamp(torch.matmul(light_dir.unsqueeze(-1).permute(0, 2, 1), normals.unsqueeze(-1)), min=0, max=1).squeeze(-1)
    dot_prod_s = torch.clamp(torch.matmul(R.unsqueeze(-1).permute(0, 2, 1), view_dirs.unsqueeze(-1)), min=0, max=1).squeeze(-1)

    illumination = ka * (colors-80)/torch.norm(colors-80) + kd * dot_prod_d * colors + ks * (dot_prod_s)**n * colors

    return illumination
    
relighting_dict = {
    'phong': phong
}
