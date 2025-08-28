import os
import subprocess
from typing import List

import rembg
import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

# WEIGHTS_CACHE = "/src/weights/zero123plusplus"
WEIGHTS_PATH = "/nobackup/nhaldert/weights"
os.makedirs(WEIGHTS_PATH, exist_ok=True)
CHECKPOINT_URLS = [
    ("https://weights.replicate.delivery/default/zero123plusplus/zero123plusplus.tar", WEIGHTS_PATH),
]
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def download_model(url, dest):
    print("Downloading weights...")

    os.makedirs("/nobackup/nhaldert/tmp", exist_ok=True)
    tar_path = "/nobackup/nhaldert/tmp/zero123plusplus.tar"

    try:
        subprocess.check_call(["wget", url, "-O", tar_path])
        subprocess.check_call(["tar", "-xf", tar_path, "-C", dest])
        print("Download and extraction complete.")
    except subprocess.CalledProcessError as e:
        print("Error during download or extraction:", e)
        raise


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists("weights"):
            os.mkdir("weights")

        for (CKPT_URL, target_folder) in CHECKPOINT_URLS:
            if not os.path.exists(target_folder):
                download_model(CKPT_URL, target_folder)

        print("Setting up pipeline...")

        self.pipeline = DiffusionPipeline.from_pretrained(
            # "./weights/zero123plusplus",
            f"""{WEIGHTS_PATH}/zero123plusplus""",
            custom_pipeline="./diffusers-support/",
            torch_dtype=torch.float16,
            # local_files_only=True
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline.to('cuda:0')

    def predict(
        self,
        image: Path = Input(description="Input image. Aspect ratio should be 1:1. Recommended resolution is >= 320x320 pixels."),
        remove_background: bool = Input(description="Remove the background of the input image", default=False),
        return_intermediate_images: bool = Input(description="Return the intermediate images together with the output images", default=False),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        outputs = []

        cond = Image.open(str(image))
        image_filename = "original" + image.suffix

        # optional background removal step
        if remove_background:
            rembg_session = rembg.new_session()
            cond = rembg.remove(cond, session=rembg_session)
            # image should be a png after background removal
            image_filename += ".png"

        if return_intermediate_images:
            temp_original = f"/tmp/{image_filename}"
            cond.save(temp_original)
            outputs.append(temp_original)

        os.makedirs(os.path.join(CUR_DIR, "outputs"), exist_ok=True)

        # all_results = self.pipeline(cond, num_inference_steps=75)
        # for i, output_img in enumerate(all_results.images):
        #     filename = os.path.join(CUR_DIR, "outputs", f"output{i}.jpg")
        #     output_img.save(filename)
        #     outputs.append(filename)

        # return [Path(output) for output in outputs]

        angles = [0, 90, 180, 270]
        for angle in angles:
            result = predictor_obj.pipeline(
                cond,
                prompt=f"view from {angle} degree",
                num_inference_steps=75,
                num_images_per_prompt=1
            ).images[0]
            result.save(f"outputs/output_{angle}.jpg")


if __name__ == "__main__":
    predictor_obj = Predictor()
    predictor_obj.setup()
    predictor_obj.predict(
        image=Path("rendered_img.png"),
        remove_background=True,
        return_intermediate_images=False
    )
    print("Prediction complete. Check /tmp for output images.")