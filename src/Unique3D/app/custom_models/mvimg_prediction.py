import sys
import torch
import gradio as gr
from PIL import Image
import numpy as np
from rembg import remove
from app.utils import change_rgba_bg, rgba_to_rgb
from app.custom_models.utils import load_pipeline
from scripts.all_typing import *
from scripts.utils import session, simple_preprocess

training_config = "app/custom_models/image2mvimage.yaml"
# checkpoint_path = "ckpt/img2mvimg/unet_state_dict.pth"
checkpoint_path = "/nobackup/nhaldert/unet_state_dict.pth"
trainer, pipeline = load_pipeline(training_config, checkpoint_path)
# pipeline.enable_model_cpu_offload()


def predict(img_list: List[Image.Image], guidance_scale=2.0, **kwargs):
    if isinstance(img_list, Image.Image):
        img_list = [img_list]
    img_list = [rgba_to_rgb(i) if i.mode == "RGBA" else i for i in img_list]
    ret = []
    for img in img_list:
        images = trainer.pipeline_forward(
            pipeline=pipeline, image=img, guidance_scale=guidance_scale, **kwargs
        ).images
        ret.extend(images)
    return ret


def run_mvprediction(
    input_image: Image.Image, remove_bg=True, guidance_scale=1.5, seed=1145
):
    if input_image.mode == "RGB" or np.array(input_image)[..., -1].mean() == 255.0:
        # still do remove using rembg, since simple_preprocess requires RGBA image
        print("RGB image not RGBA! still remove bg!")
        remove_bg = True

    if remove_bg:
        input_image = remove(input_image, session=session)

    # make front_pil RGBA with white bg
    input_image = change_rgba_bg(input_image, "white")
    single_image = simple_preprocess(input_image)

    generator = (
        torch.Generator(device="cuda").manual_seed(int(seed)) if seed >= 0 else None
    )

    rgb_pils = predict(
        single_image,
        generator=generator,
        guidance_scale=guidance_scale,
        width=256,
        height=256,
        num_inference_steps=30,
    )

    return rgb_pils, single_image


from PIL import Image
import os
import pandas as pd

BASE_DIR = "/u/nhaldert/work_dir/3d-image-generation/src"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join("/nobackup/nhaldert", "data")


def main():
    # img = Image.open("rendered_mesh.png")  # your object image
    # mv_images, front_pil = run_mvprediction(img)

    # # Save results
    # for idx, view in enumerate(mv_images):
    #     view.save(f"view_{idx}.png")
    dataset_path = os.path.join(BASE_DIR, "dataset_files.csv")
    df = pd.read_csv(dataset_path)
    all_filenames = df["file_name"].tolist()
    for file_name in all_filenames:
        file_name = os.path.splitext(file_name)[0]
        raw_image_path = os.path.join(DATA_DIR, "outputs", "gt", file_name, "front.png")

        try:
            assert os.path.isfile(raw_image_path), f"File not found: {raw_image_path}"
        except AssertionError as e:
            print(e)
            continue

        print(f"Processing {file_name} ...")

        raw_im = Image.open(raw_image_path).convert("RGBA")
        output_dir = os.path.join(
            DATA_DIR,
            "outputs",
            "unique3d",
            file_name,
        )
        os.makedirs(output_dir, exist_ok=True)
        mv_images, front_pil = run_mvprediction(raw_im)
        # front_pil.save(os.path.join(output_dir, "front.png"))
        view_name = ["front", "right", "back", "left"]
        for idx, view in enumerate(mv_images):
            view.save(os.path.join(output_dir, f"{view_name[idx]}.png"))


if __name__ == "__main__":
    main()
