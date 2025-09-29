import math
import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from lpips import LPIPS
from clip import clip
from scipy.spatial import cKDTree, distance

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join("/nobackup/nhaldert", "data")


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


def get_clip_sim(clip_model, preprocess, img1, img2):
    # if img1.ndim == 3 and img1.shape[2] == 3:
    #     img1 = np.transpose(img1, (2, 0, 1))  # Convert to CxHxW
    # if img2.ndim == 3 and img2.shape[2] == 3:
    #     img2 = np.transpose(img2, (2, 0, 1))  # Convert to CxHxW

    # clip_model, preprocess = clip.load("ViT-B/32", device="cuda")  # Load the CLIP model
    with torch.no_grad():
        img1_features = clip_model.encode_image(
            preprocess(img1).unsqueeze(0).to("cuda")
        )
        img2_features = clip_model.encode_image(
            preprocess(img2).unsqueeze(0).to("cuda")
        )

    img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
    img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)

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


def view_metrics(metric_file_path: str):
    df_results = pd.DataFrame(
        columns=["Method", "view", "PSNR", "SSIM", "LPIPS", "CLIP Similarity"]
    )
    views = ["front", "left", "right", "back"]

    df_metrics = pd.read_csv(metric_file_path)
    # print(df_metrics.head())

    for view in views:
        df_psnr_unique3d = df_metrics[df_metrics["view"] == view]["psnr_unique3d"]
        df_ssim_unique3d = df_metrics[df_metrics["view"] == view]["ssim_unique3d"]
        df_lpips_unique3d = df_metrics[df_metrics["view"] == view]["lpips_unique3d"]
        df_clip_unique3d = df_metrics[df_metrics["view"] == view]["clip_sim_unique3d"]
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    [
                        {
                            "Method": "Unique3D",
                            "view": view,
                            "PSNR": df_psnr_unique3d.mean(),
                            "SSIM": df_ssim_unique3d.mean(),
                            "LPIPS": df_lpips_unique3d.mean(),
                            "CLIP Similarity": df_clip_unique3d.mean(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        df_psnr_zero123 = df_metrics[df_metrics["view"] == view]["psnr_zero123"]
        df_ssim_zero123 = df_metrics[df_metrics["view"] == view]["ssim_zero123"]
        df_lpips_zero123 = df_metrics[df_metrics["view"] == view]["lpips_zero123"]
        df_clip_zero123 = df_metrics[df_metrics["view"] == view]["clip_sim_zero123"]
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    [
                        {
                            "Method": "Zero123",
                            "view": view,
                            "PSNR": df_psnr_zero123.mean(),
                            "SSIM": df_ssim_zero123.mean(),
                            "LPIPS": df_lpips_zero123.mean(),
                            "CLIP Similarity": df_clip_zero123.mean(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # Overall metrics
    df_psnr_unique3d = df_metrics["psnr_unique3d"]
    df_ssim_unique3d = df_metrics["ssim_unique3d"]
    df_lpips_unique3d = df_metrics["lpips_unique3d"]
    df_clip_unique3d = df_metrics["clip_sim_unique3d"]
    df_results = pd.concat(
        [
            df_results,
            pd.DataFrame(
                [
                    {
                        "Method": "Unique3D",
                        "view": "All",
                        "PSNR": df_psnr_unique3d.mean(),
                        "SSIM": df_ssim_unique3d.mean(),
                        "LPIPS": df_lpips_unique3d.mean(),
                        "CLIP Similarity": df_clip_unique3d.mean(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df_psnr_zero123 = df_metrics["psnr_zero123"]
    df_ssim_zero123 = df_metrics["ssim_zero123"]
    df_lpips_zero123 = df_metrics["lpips_zero123"]
    df_clip_zero123 = df_metrics["clip_sim_zero123"]
    df_results = pd.concat(
        [
            df_results,
            pd.DataFrame(
                [
                    {
                        "Method": "Zero123",
                        "view": "All",
                        "PSNR": df_psnr_zero123.mean(),
                        "SSIM": df_ssim_zero123.mean(),
                        "LPIPS": df_lpips_zero123.mean(),
                        "CLIP Similarity": df_clip_zero123.mean(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    print("Final Results:")
    for i in range(0, len(df_results), 2):
        print(df_results.iloc[i : i + 2])
        print()

    return df_results


def main():
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

    df_dataset = pd.read_csv(os.path.join(CUR_DIR, "dataset_files.csv"))
    df_metrics = pd.DataFrame(
        columns=[
            "filename",
            "view",
            "psnr_unique3d",
            "ssim_unique3d",
            "lpips_unique3d",
            "clip_sim_unique3d",
            "psnr_zero123",
            "ssim_zero123",
            "lpips_zero123",
            "clip_sim_zero123",
        ]
    )

    all_filenames = df_dataset["file_name"].tolist()
    print(f"Total files to process: {len(all_filenames)}")
    views = ["front", "left", "right", "back"]
    for file_name in all_filenames:
        file_name = os.path.splitext(file_name)[0]
        print(f"Processing {file_name} ...")
        for view in views:
            gt_img_path = os.path.join(
                DATA_DIR, "outputs", "gt", file_name, f"{view}.png"
            )
            try:
                assert os.path.isfile(gt_img_path), f"File not found: {gt_img_path}"

                unique3d_img_path = os.path.join(
                    DATA_DIR, "outputs", "unique3d", file_name, f"{view}.png"
                )
                assert os.path.isfile(
                    unique3d_img_path
                ), f"File not found: {unique3d_img_path}"

                zero123_img_path = os.path.join(
                    DATA_DIR, "outputs", "zero123", file_name, f"{view}.png"
                )
                assert os.path.isfile(
                    zero123_img_path
                ), f"File not found: {zero123_img_path}"
            except AssertionError as e:
                continue
                print(e)

            print(f"Processing {file_name} - {view} ...")
            gt_img = (
                np.array(
                    Image.open(gt_img_path).convert("RGB").resize((256, 256))
                ).astype(np.float32)
                / 255.0
            )
            unique3d_img = (
                np.array(
                    Image.open(unique3d_img_path)
                    .convert("RGB")
                    .resize((256, 256), Image.LANCZOS)
                ).astype(np.float32)
                / 255.0
            )
            zero123_img = (
                np.array(
                    Image.open(zero123_img_path)
                    .convert("RGB")
                    .resize((256, 256), Image.LANCZOS)
                ).astype(np.float32)
                / 255.0
            )

            # psnr = get_psnr(gt_img, unique3d_img)
            # ssim = get_ssim(gt_img, unique3d_img)
            # lpips = get_lpips(gt_img, unique3d_img)
            # clip_sim = get_clip_sim(clip_model, preprocess, gt_img, unique3d_img)

            # print(f"Metrics for {file_name} - {view}:")
            # print(f"  PSNR: {psnr}")
            # print(f"  SSIM: {ssim}")
            # print(f"  LPIPS: {lpips}")
            # print(f"  CLIP Similarity: {clip_sim}")

            new_row = pd.DataFrame(
                [
                    {
                        "filename": file_name,
                        "view": view,
                        "psnr_unique3d": get_psnr(gt_img, unique3d_img),
                        "ssim_unique3d": get_ssim(gt_img, unique3d_img),
                        "lpips_unique3d": get_lpips(gt_img, unique3d_img),
                        "clip_sim_unique3d": get_clip_sim(
                            clip_model,
                            preprocess,
                            Image.open(gt_img_path),
                            Image.open(unique3d_img_path),
                        ),
                        "psnr_zero123": get_psnr(gt_img, zero123_img),
                        "ssim_zero123": get_ssim(gt_img, zero123_img),
                        "lpips_zero123": get_lpips(gt_img, zero123_img),
                        "clip_sim_zero123": get_clip_sim(
                            clip_model,
                            preprocess,
                            Image.open(gt_img_path),
                            Image.open(zero123_img_path),
                        ),
                    },
                ]
            )
            df_metrics = pd.concat([df_metrics, new_row], ignore_index=True)

    df_metrics.to_csv(os.path.join(CUR_DIR, "metrics_results.csv"), index=False)
    print("Metrics saved to metrics_results.csv")


if __name__ == "__main__":
    print(f"Current directory: {CUR_DIR}")
    main()
    view_metrics(os.path.join(CUR_DIR, "metrics_results.csv"))

    # print(
    #     "PSNR -",
    #     get_psnr(
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
    #     ),
    # )

    # print(
    #     "SSIM -",
    #     get_ssim(
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
    #     ),
    # )
    # print(
    #     "LPIPS -",
    #     get_lpips(
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
    #     ),
    # )
    # print(
    #     "CLIP Similarity -",
    #     get_clip_sim(
    #         Image.open(os.path.join(CUR_DIR, "rendered_img.png")),
    #         Image.open(os.path.join(CUR_DIR, "rendered_mesh.png")),
    #     ),
    # )
    # print(
    #     "Chamfer Distance -",
    #     get_chamfer_distance(
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_img.png"))),
    #         np.array(Image.open(os.path.join(CUR_DIR, "rendered_mesh.png"))),
    #     ),
    # )
