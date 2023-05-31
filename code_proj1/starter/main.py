import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from PIL import Image, ImageDraw
from starter.utils import *

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
import imageio

device = get_device()

def get_cow_mesh(cow_path="data/cow.obj",image_size=256, color=[0.7, 0.7, 1]):
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    return mesh

def render_360(mesh, image_size=256, output_path='images/q_1-1.gif', light_loc=[[0, 0, -3]], num_views=12, fps=8, elev=0):
    renderer = get_mesh_renderer(image_size=image_size)

    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=3,
        elev=elev,
        azim=angles[i],
    )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
        lights = pytorch3d.renderer.PointLights(location=light_loc, device=device)
        
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, fps=fps)

def render_tetrah(image_size=512, output_path='images/q_2-1.gif', color=[0.7, 0.7, 1]):
    vertices = torch.Tensor([[0., 0., 0.], [1., 0.5, 1.], [0.5, 1., 1.], [1., 1., 0.5]], device = device)
    faces = torch.Tensor([[0, 3, 1], [0, 1, 2], [0, 2, 3], [3, 2, 1]], device = device)
    vertices = vertices.unsqueeze(0)  
    faces = faces.unsqueeze(0)  
    textures = torch.ones_like(vertices)  
    textures = textures * torch.tensor(color) 

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    render_360(mesh, image_size, output_path)

def render_cube(image_size=512, output_path='images/q_2-2.gif', color=[0.2, 0.2, 1]):
    vertices = torch.Tensor(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], 
                            [0, 1, 1], [1, 0, 0], [1, 0, 1], 
                            [1, 1, 0], [1, 1, 1]]), device = device)
    faces = torch.Tensor([[1, 5, 7], [1, 7, 3], [5, 4, 7], [7, 4, 6],
                          [0, 6, 2], [0, 4, 6], [3, 1, 0], [0, 2, 3],
                          [7, 6, 3], [6, 2, 3], [0, 1, 4], [1, 5, 4]], device = device)
    vertices = vertices.unsqueeze(0) 
    faces = faces.unsqueeze(0)  
    textures = torch.ones_like(vertices)  
    textures = textures * torch.tensor(color) 

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    render_360(mesh, image_size, output_path, light_loc=[[1, 1, 3]], num_views=24, elev=15)

def retexture(cow_mesh, image_size=512, output_path='images/q_3.gif', color1 = [0, 0, 1], color2 = [1, 0, 0]):
    vertices, _ = load_cow_mesh(path="data/cow.obj")
    texture_rgb = vertices.clone()
    color1, color2 = torch.tensor(color1), torch.tensor(color2)
    z_min, z_max = torch.amin(vertices[:, 2]).item(), torch.amax(vertices[:, 2]).item()
    
    N_v = vertices.shape[0]
    for i in range(N_v):
        z = vertices[i, 2]
        alpha = (z - z_min) / (z_max - z_min)
        texture_rgb[i] = alpha * color2 + (1 - alpha) * color1
    
    texture_rgb = texture_rgb.unsqueeze(0)

    textures_2c = pytorch3d.renderer.TexturesVertex(texture_rgb.to(device))
    cow_mesh.textures = textures_2c
    cow_mesh = cow_mesh.to(device)
    render_360(cow_mesh, image_size, output_path)  

def get_cow_mesh_1(cow_path="data/cow.obj", color=[0.7, 0.7, 1], translation=0):
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  
    faces = faces.unsqueeze(0)  
    textures = torch.ones_like(vertices) 
    textures = textures * torch.tensor(color)  
    mesh = pytorch3d.structures.Meshes(
        verts=vertices+torch.tensor([translation, 0, 0]).float(),
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    return mesh

def get_cow_mesh_2(cow_path="data/cow.obj", translation=0):
    vertices, face_props, text_props = pytorch3d.io.load_obj(cow_path)
    faces = face_props.verts_idx
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    verts_uvs = text_props.verts_uvs
    faces_uvs = face_props.textures_idx

    texture_map = plt.imread("data/cow_texture.png")

    textures_uv = pytorch3d.renderer.TexturesUV(
    maps=torch.tensor([texture_map]),
    faces_uvs=faces_uvs.unsqueeze(0),
    verts_uvs=verts_uvs.unsqueeze(0),
    ).to(device)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices+torch.tensor([translation, 0, 0]).float(),
        faces=faces,
        textures=textures_uv,
    )
    mesh = mesh.to(device)

    return mesh

def magic_cow(image_size=256, output_path='images/q_6.gif'):
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # mesh_join = pytorch3d.structures.join_meshes_as_batch([mesh1, mesh2])
    num_views=12
    translations1 = np.linspace(3.5, -3.5, num_views, endpoint=False)
    translations2 = np.linspace(-3.5, 3.5, num_views, endpoint=False)

    images = []
    for i in range(num_views):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device)

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        
        mesh = get_cow_mesh_1(translation=translations1[i])
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]

        image = Image.fromarray((rend * 255).astype(np.uint8))
        images.append(np.array(image))

    for i in range(num_views):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device)

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        
        mesh = get_cow_mesh_2(translation=translations2[i])
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]

        image = Image.fromarray((rend * 255).astype(np.uint8))
        images.append(np.array(image))

    imageio.mimsave(output_path, images, fps=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default='1.1')
    parser.add_argument("--image_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default='images/q_1-1.gif')
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    # q 1.1
    if args.question == '1.1':
        cow_mesh = get_cow_mesh(cow_path=args.image_path, image_size=args.image_size)
        render_360(cow_mesh, image_size=args.image_size, output_path=args.output_path)

    # q 1.2 run "python -m starter.dolly_zoom"

    # q 2.1
    if args.question == '2.1':
        render_tetrah(image_size=args.image_size, output_path=args.output_path)

    # q 2.2
    if args.question == '2.2':
        render_cube(image_size=args.image_size, output_path=args.output_path)

    # q 3 
    if args.question == '3':
        cow_mesh = get_cow_mesh(cow_path=args.image_path, image_size=args.image_size)
        retexture(cow_mesh, image_size=args.image_size, output_path=args.output_path)

    # q 4 run "python -m starter.camera_transforms"

    # q 5 run "python -m starter.render_generic (parameters)"

    # q 6
    if args.question == '6':
        magic_cow()



