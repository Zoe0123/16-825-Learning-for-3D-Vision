
import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from PIL import Image, ImageDraw
import imageio

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)

from pytorch3d.io import load_obj


def get_mesh_renderer(image_size=512, lights=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def render_360_mesh(mesh, image_size=256, output_path='images/q_1-1.gif', device=None, dist=3):
    renderer = get_mesh_renderer(image_size=image_size)


    num_views=12
    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=15,
        azim=angles[i],
    )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
        lights = pytorch3d.renderer.PointLights(location=[[1, 1, 3]], device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, fps=8)

def get_points_renderer(image_size=512, radius=0.01, background_color=(1, 1, 1)):
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def render_360_pc(point_cloud, image_size=256, output_path='images/q_5-1_pc1.gif', device=None):
    renderer = get_points_renderer(image_size=image_size)
 
    num_views = 12
    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=1,
        elev=0,
        azim=angles[i],
    )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
        
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, fps=15)

def render_vox(vox, output_path):
    device=vox.device
    mesh = pytorch3d.ops.cubify(vox, thresh=0.5, device=device)
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0))
    textures = textures * torch.tensor([0.7, 0.7, 1], device=device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_360_mesh(mesh, output_path=output_path, device=device)

def render_pc(points, output_path):
    device=points.device
    points = points.detach()[0]
    color = (points - points.min()) / (points.max() - points.min())
    pc = pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(device)
    render_360_pc(pc, output_path=output_path, device=device)

def render_mesh(mesh, output_path):
    device=mesh.device
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0), device = device)
    textures = textures * torch.tensor([0.7, 0.7, 1], device = device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_360_mesh(mesh.detach(), output_path=output_path, device=device, dist=1.5)



