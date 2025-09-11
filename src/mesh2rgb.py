import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
)
import trimesh

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_mesh_paths(mesh_root):
    print(os.listdir(mesh_root))
    texture_path = os.path.join(mesh_root, "materials", "textures", "texture.png")
    mesh_path = os.path.join(mesh_root, "meshes", "model.obj")
    material_path = os.path.join(mesh_root, "meshes", "model.mtl")

    assert os.path.exists(texture_path), f"Texture file not found: {texture_path}"
    assert os.path.exists(mesh_path), f"Mesh file not found: {mesh_path}"
    assert os.path.exists(material_path), f"Material file not found: {material_path}"

    files = [texture_path, mesh_path, material_path]
    for filename in files:
        shutil.copy2(filename, os.path.join(CUR_DIR, os.path.basename(filename)))

    return os.path.join(CUR_DIR, "model.obj")


def show_mesh(mesh_path):
    mesh = trimesh.load(mesh_path)
    mesh.show()


def test_fov_view(mesh_path):
    device = "cpu"
    mesh = load_objs_as_meshes([mesh_path], device=device)
    plt.figure(figsize=(7, 7))
    texture_image = mesh.textures.maps_padded()
    plt.imshow(texture_image.squeeze().cpu().numpy())
    plt.axis("off")


def normalize_mesh_torch(verts):
    min_xyz = verts.min(dim=0).values
    max_xyz = verts.max(dim=0).values
    center = 0.5 * (min_xyz + max_xyz)
    scale = (max_xyz - min_xyz).max()
    verts_normalized = (verts - center) / scale
    return verts_normalized


def generate_gt_mv_img(mesh_path):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    mesh = load_objs_as_meshes([mesh_path], device=device)
    verts = mesh.verts_packed()
    mesh = Meshes(
        verts=[normalize_mesh_torch(verts)],
        faces=[mesh.faces_packed()],
        textures=mesh.textures,
    )

    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    raster_settings = RasterizationSettings(image_size=1024)
    angles = {
        "front": torch.eye(3),
        "back": torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        "right": torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        "left": torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    }
    T = torch.tensor([[0, 0, 1]], device=device)

    os.makedirs("gtoutputs", exist_ok=True)
    for view, R in angles.items():
        if view == "back":
            lights = PointLights(device=device, location=[[-2.0, 2.0, 2.0]])
            # R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            # T = torch.tensor([[0, 0, 0.6]], device=device)
        elif view == "front":
            lights = PointLights(device=device, location=[[-2.0, 2.0, 2.0]])
            # T = torch.tensor([[0, 0, 0.5]], device=device)
        elif view == "right":
            lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
            T = torch.tensor([[0, 0, 1.2]], device=device)
        elif view == "left":
            lights = PointLights(device=device, location=[[-2.0, 2.0, -2.0]])
            T = torch.tensor([[0, 0, 1.2]], device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R.unsqueeze(0), T=T)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
        )

        images = renderer(mesh)
        rgb_image = images[0, ..., :3].cpu().numpy()
        plt.imsave(f"gtoutputs/{view}.png", rgb_image)
        # rgba_image_np = (images[0, ..., :].cpu().numpy() * 255).astype(np.uint8)
        # rgb_image = Image.fromarray(rgba_image_np, "RGBA")
        # rgb_image.save(f"gtoutputs/{view}.png")

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


def main():
    mesh_root = "/nobackup/nhaldert/data/Multiview Image Generation Experiment/Object 1/Threshold_Tray_Rectangle_Porcelain"
    mesh_path = "model.obj"
    # show_mesh(mesh_path)
    # render_rgb_from_mesh(mesh_path)
    # print(os.listdir("/nobackup/nhaldert/data"))
    generate_gt_mv_img(fetch_mesh_paths(mesh_root))
    # generate_gt_mv_img(mesh_path)
    # test_fov_view(mesh_path=mesh_path)


if __name__ == "__main__":
    main()
