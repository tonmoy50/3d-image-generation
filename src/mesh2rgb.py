import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)
import trimesh


def show_mesh(mesh_path):
    mesh = trimesh.load(mesh_path)
    mesh.show()


def generate_gt_mv_img(mesh_path):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    mesh = load_objs_as_meshes([mesh_path], device=device)

    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    raster_settings = RasterizationSettings(image_size=512)

    # Define view angles using rotation matrices
    angles = {
        "front": torch.eye(3),
        # "back": torch.eye(3),
        "back": torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        "right": torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        "left": torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    }

    # Translation: move camera back along z
    T = torch.tensor([[0, 0, 0.5]], device=device)

    os.makedirs("gtoutputs", exist_ok=True)

    for view, R in angles.items():
        if view == "back":
            lights = PointLights(device=device, location=[[-2.0, 2.0, 2.0]])
            # R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            T = torch.tensor([[0, 0, 0.8]], device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R.unsqueeze(0), T=T)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
        )

        images = renderer(mesh)
        rgb_image = images[0, ..., :3].cpu().numpy()
        plt.imsave(f"gtoutputs/{view}.png", rgb_image)
        print(f"Saved gtoutputs/{view}.png")


def render_rgb_from_mesh(mesh_path):

    # Select CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the mesh (no textures)
    mesh = load_objs_as_meshes([mesh_path], device=device)

    # Manually assign vertex color (light gray)
    # verts_rgb = torch.ones_like(mesh.verts_padded()) * 0.8  # shape: (1, V, 3)
    # mesh.textures = TexturesVertex(verts_features=verts_rgb)

    # Setup renderer
    cameras = FoVPerspectiveCameras(
        device=device,
        R=torch.eye(3).unsqueeze(0),  # identity rotation
        T=torch.tensor([[0, 0, 0.5]], device=device),  # zoom out by increasing Z
    )
    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    raster_settings = RasterizationSettings(image_size=512)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )

    # Render
    images = renderer(mesh)
    rgb_image = images[0, ..., :3].cpu().numpy()

    # Save to file
    plt.imsave("rendered_img.png", rgb_image)

    # Optional: Show
    # plt.imshow(rgb_image)
    # plt.axis("off")
    # plt.show()


if __name__ == "__main__":
    mesh_path = "model.obj"
    # show_mesh(mesh_path)
    # render_rgb_from_mesh(mesh_path)
    # generate_gt_mv_img(mesh_path)
