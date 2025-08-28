import os
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import time
import math

from lovely_numpy import lo
from contextlib import nullcontext
from einops import rearrange

import torch
from torchvision import transforms
from torch import autocast

from ldm.util import (
    create_carvekit_interface,
    load_and_preprocess,
    instantiate_from_config,
)
from ldm.models.diffusion.ddim import DDIMSampler


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess_image(models, input_im, preprocess):
    """
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    """

    print("old input_im:", input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models["carvekit"], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(
        f"Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s."
    )
    print("new input_im:", lo(input_im))

    return input_im


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def sample_model(
    input_im,
    model,
    sampler,
    precision,
    h,
    w,
    ddim_steps,
    n_samples,
    scale,
    ddim_eta,
    x,
    y,
    z,
):
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor(
                [
                    math.radians(x),
                    math.sin(math.radians(y)),
                    math.cos(math.radians(y)),
                    z,
                ]
            )
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond["c_crossattn"] = [c]
            cond["c_concat"] = [
                model.encode_first_stage((input_im.to(c.device)))
                .mode()
                .detach()
                .repeat(n_samples, 1, 1, 1)
            ]
            if scale != 1.0:
                uc = {}
                uc["c_concat"] = [
                    torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)
                ]
                uc["c_crossattn"] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None,
            )
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(raw_image_path: str):
    device_idx = 0
    config = "configs/sd-objaverse-finetune-c_concat-256.yaml"
    h = 256
    w = 256
    # ckpt = "/nobackup/nhaldert/weights/105000.ckpt"
    ckpt = "/nobackup/nhaldert/weights/zero123-xl.ckpt"

    raw_im = Image.open(raw_image_path).convert("RGBA")
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)

    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(config)

    models = dict()
    models["carvekit"] = create_carvekit_interface()
    models["turncam"] = load_model_from_config(config, ckpt, device=device)

    input_im = preprocess_image(models, raw_im, True)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(models["turncam"])
    used_x = [0.0, 0.0, 0.0, 0.0]
    used_y = [0.0, -90, 90, 180]
    used_z = [0.0, 0.0, 0.0, 0.0]
    ddim_steps = 30
    n_samples = 1
    scale = 3.0
    ddim_eta = 1.0
    precision = "fp32"

    i = 0
    for x, y, z in zip(used_x, used_y, used_z):
        print(f"Processing image {i} with x={x}, y={y}, z={z}")
        x_samples_ddim = sample_model(
            input_im,
            models["turncam"],
            sampler,
            precision,
            h,
            w,
            ddim_steps,
            n_samples,
            scale,
            ddim_eta,
            x,
            y,
            z,
        )
        x_sample = 255.0 * rearrange(x_samples_ddim[0].cpu().numpy(), "c h w -> h w c")
        output_im = Image.fromarray(x_sample.astype(np.uint8))
        output_im.save(os.path.join(CUR_DIR, "outputs", f"output_{i}.png"))
        print(
            f"Saved output image {i} to {os.path.join(CUR_DIR, 'outputs', f'output_{i}.png')}"
        )
        i += 1

    # output_ims = []
    # for x_sample in x_samples_ddim:
    #     x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
    #     output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    # for i, output_im in enumerate(output_ims):
    #     output_im.save(os.path.join(CUR_DIR, "outputs", f"output_{i}.png"))
    #     print(
    #         f"Saved output image {i} to {os.path.join(CUR_DIR, 'outputs', f'output_{i}.png')}"
    #     )
    # Image.fromarray((input_im).astype(np.uint8)).save(
    #     os.path.join(CUR_DIR, "outputs", "preprocessed_img.png")
    # )


if __name__ == "__main__":
    raw_image_path = "rendered_img.png"  # Path to the input image
    output_image_path = "output_image.png"  # Path to save the output image
    os.makedirs(os.path.join(CUR_DIR, "outputs"), exist_ok=True)
    main(raw_image_path)
    # print(f"Processed image saved to {output_image_path}")
