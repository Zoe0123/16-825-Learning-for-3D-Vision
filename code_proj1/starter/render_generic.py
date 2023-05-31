"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image
from PIL import Image, ImageDraw
import imageio


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

# q 5
def render_360_pc(point_cloud, image_size=256, output_path='images/q_5-1_pc1.gif',
 num_views=12, fps=15, elev=10, dist=7, device=None, background_color=(1, 1, 1), rotate_R=False):
    if device is None:
        device = get_device()
    renderer = get_points_renderer(image_size=image_size, background_color=background_color)

    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=angles[i],
    )
        # rotate upside down
        if rotate_R:
            R = pytorch3d.transforms.euler_angles_to_matrix(torch.Tensor([0, 0, np.pi]), "XYZ") @ R

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
    imageio.mimsave(output_path, images, fps=fps)


def rgbd_2_pc(rgbd_path="data/rgbd_data.pkl", image_size=256, background_color=(1, 1, 1), device=None):
    data = load_rgbd_data(rgbd_path)
    img1, depth1, mask1 = torch.Tensor(data['rgb1']), torch.Tensor(data['depth1']), torch.Tensor(data['mask1'])
    camera1 = data['cameras1']
    img2, depth2, mask2 = torch.Tensor(data['rgb2']), torch.Tensor(data['depth2']), torch.Tensor(data['mask2'])
    camera2 = data['cameras2']

    points1, rgb1 = unproject_depth_image(img1, mask1, depth1, camera1)
    points2, rgb2 = unproject_depth_image(img2, mask2, depth2, camera2)
    
    points3, rgb3 = torch.cat((points1, points2), 0), torch.cat((rgb1, rgb2), 0)
    points1, rgb1 = points1.to(device).unsqueeze(0), rgb1.to(device).unsqueeze(0)
    points2, rgb2 = points2.to(device).unsqueeze(0), rgb2.to(device).unsqueeze(0)
    points3, rgb3 = points3.to(device).unsqueeze(0), rgb3.to(device).unsqueeze(0)

    point_cloud1 = pytorch3d.structures.Pointclouds(points=points1, features=rgb1)
    point_cloud2 = pytorch3d.structures.Pointclouds(points=points2, features=rgb2)
    point_cloud3 = pytorch3d.structures.Pointclouds(points=points3, features=rgb3)

    render_360_pc(point_cloud1, image_size=image_size, output_path='images/q_5-1_pc1.gif', fps=10, device=device, background_color=background_color, rotate_R=True)
    render_360_pc(point_cloud2, image_size=image_size, output_path='images/q_5-1_pc2.gif', fps=10, device=device, background_color=background_color, rotate_R=True)
    render_360_pc(point_cloud3, image_size=image_size, output_path='images/q_5-1_pc_union.gif', fps=10, device=device, background_color=background_color, rotate_R=True)

def render_torus_parametric(image_size=256, num_samples=200, device=None):
    if device is None:
        device = get_device()
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (2 + 1 * torch.cos(Theta)) * torch.cos(Phi)
    y = (2 + 1 * torch.cos(Theta)) * torch.sin(Phi)
    z = 1 * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    render_360_pc(torus_point_cloud, image_size=image_size, output_path='images/q_5-2.gif', fps=7, dist=7, device=device)

def render_360_mesh(mesh, image_size=256, output_path='images/q_5-3.gif', num_views=12, fps=7, elev=10, dist=7, device=None):
    if device is None:
        device = get_device()
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)

    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=angles[i],
    )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
        
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, fps=fps)

def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -4.1
    max_value = 4.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # implicit function for torus (R=4, r=2)
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - 2)**2 + Z ** 2 - 1**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    render_360_mesh(mesh, output_path='images/q_5-3.gif')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--question", type=str, default='5.1')
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)

    # q 5.1
    if args.question == '5.1':
        rgbd_2_pc()

    # q 5.2
    if args.question == '5.2':
        render_torus_parametric()

    # q 5.3
    if args.question == '5.3':
        render_torus_mesh()

    

