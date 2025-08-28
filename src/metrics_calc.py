import math
import os
from PIL import Image

import numpy as np

import torch
from lpips import LPIPS
from clip import clip
from scipy.spatial import cKDTree, distance


def get_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def get_ssim(img1, img2):
    from skimage.metrics import structural_similarity as ssim

    # Ensure images are in grayscale if they are RGB
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
    if img2.ndim == 3 and img2.shape[2] == 3:
        img2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])

    ssim_value = ssim(img1, img2, win_size=3, data_range=img2.max() - img2.min())
    return ssim_value


def get_lpips(img1, img2):
    img1 = np.transpose(img1, (2, 0, 1))[:3, :, :]
    img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0)

    img2 = np.transpose(img2, (2, 0, 1))[:3, :, :]
    img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0)

    lpips_model = LPIPS(net="alex")  # Load the LPIPS model
    return lpips_model(img1, img2).item()


def get_clip_sim(img1, img2):
    # Ensure images are in the correct format
    # if img1.ndim == 3 and img1.shape[2] == 3:
    #     img1 = np.transpose(img1, (2, 0, 1))  # Convert to CxHxW
    # if img2.ndim == 3 and img2.shape[2] == 3:
    #     img2 = np.transpose(img2, (2, 0, 1))  # Convert to CxHxW

    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")  # Load the CLIP model
    with torch.no_grad():
        img1_features = clip_model.encode_image(
            preprocess(img1).unsqueeze(0).to("cuda")
        )
        img2_features = clip_model.encode_image(
            preprocess(img2).unsqueeze(0).to("cuda")
        )

    return (img1_features @ img2_features.T).item()


def get_chamfer_distance(img1, img2):
    img1_points = np.column_stack(
        np.where(img1 == 1)
    )  # non-zero (foreground) pixels of img1
    img2_points = np.column_stack(
        np.where(img2 == 1)
    )  # non-zero (foreground) pixels of img2

    tree = cKDTree(img1_points)
    dist_img2 = tree.query(img2_points)[0]
    tree = cKDTree(img2_points)
    dist_img1 = tree.query(img1_points)[0]

    return np.sum(dist_img1) + np.sum(dist_img2)


def get_volumeiou(img1, img2):
    intersection = np.sum((img1 > 0) & (img2 > 0))
    volume1 = np.sum(img1 > 0)
    volume2 = np.sum(img2 > 0)
    return (
        intersection / float(volume1 + volume2 - intersection)
        if (volume1 + volume2 - intersection) > 0
        else 0
    )


if __name__ == "__main__":
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {CUR_DIR}")

    print(
        "PSNR -",
        get_psnr(
            np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
            np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
        ),
    )

    print(
        "SSIM -",
        get_ssim(
            np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
            np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
        ),
    )
    print(
        "LPIPS -",
        get_lpips(
            np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
            np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
        ),
    )
    print(
        "CLIP Similarity -",
        get_clip_sim(
            Image.open(os.path.join(CUR_DIR, "rendered_img.png")),
            Image.open(os.path.join(CUR_DIR, "rendered_mesh.png")),
        ),
    )
    # print(
    #     "Chamfer Distance -",
    #     get_chamfer_distance(
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
    #     ),
    # )
