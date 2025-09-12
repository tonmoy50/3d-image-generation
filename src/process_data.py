import torch

import shutil
import zipfile
import io
import os
import pandas as pd
import matplotlib.pyplot as plt

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

from mesh2rgb import normalize_mesh_torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "/nobackup/nhaldert/data"


def process_single_mesh(filename):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filename = os.path.splitext(filename)[0]
    device = "cpu"
    mesh = load_objs_as_meshes([os.path.join(CUR_DIR, "model.obj")], device=device)
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

    os.makedirs(filename, exist_ok=True)
    for view, R in angles.items():
        if view == "back":
            lights = PointLights(device=device, location=[[-2.0, 2.0, 2.0]])
        elif view == "front":
            lights = PointLights(device=device, location=[[-2.0, 2.0, 2.0]])
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

        output_dir = os.path.join(CUR_DIR, "data", "gtoutputs", filename)
        plt.imsave(os.path.join(output_dir, f"{view}.png"), rgb_image)

        print(f"Saved {filename}/{view}.png")


def prep_file_reference():
    df = pd.DataFrame(columns=["file_name", "path"])
    all_files = os.listdir(DATASET_DIR)

    for file in all_files:
        file_path = os.path.join(DATASET_DIR, file)
        if os.path.isfile(file_path):
            # file_size = os.path.getsize(file_path)
            df = df._append({"file_name": file, "path": file_path}, ignore_index=True)

    print(df)
    df.to_csv(os.path.join(CUR_DIR, "dataset_files.csv"), index=False)


def main():
    # prep_file_reference()
    dataset_df = pd.read_csv(os.path.join(CUR_DIR, "dataset_files.csv"))
    all_file_names = dataset_df["file_name"].tolist()
    all_file_paths = dataset_df["path"].tolist()

    os.makedirs(os.path.join(CUR_DIR, "data"), exist_ok=True)

    for filename, path in zip(all_file_names, all_file_paths):
        # file_path = os.path.join(DATASET_DIR, filename)
        if os.path.isfile(path):
            # compressed_file = io.BytesIO(os.path.join(DATASET_DIR, file))
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extract("meshes/model.obj", CUR_DIR)
                zip_ref.extract("meshes/model.mtl", CUR_DIR)
                zip_ref.extract("materials/textures/texture.png", CUR_DIR)

            shutil.move(
                os.path.join(CUR_DIR, "meshes", "model.obj"),
                os.path.join(CUR_DIR, "model.obj"),
            )
            shutil.move(
                os.path.join(CUR_DIR, "meshes", "model.mtl"),
                os.path.join(CUR_DIR, "model.mtl"),
            )
            shutil.move(
                os.path.join(CUR_DIR, "materials", "textures", "texture.png"),
                os.path.join(CUR_DIR, "texture.png"),
            )

            os.removedirs(os.path.join(CUR_DIR, "meshes"))
            os.removedirs(os.path.join(CUR_DIR, "materials", "textures"))
            # os.removedirs(os.path.join(CUR_DIR, "materials"))

        process_single_mesh(filename)
        print(f"Processed file: {filename}")


if __name__ == "__main__":
    main()
