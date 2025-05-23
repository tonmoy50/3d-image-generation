import sys
from PIL import Image
from gradio_app.utils import rgba_to_rgb, simple_remove
from gradio_app.custom_models.utils import load_pipeline
from scripts.utils import rotate_normals_torch
from scripts.all_typing import *

training_config = "gradio_app/custom_models/image2normal.yaml"
checkpoint_path = "ckpt/image2normal/unet_state_dict.pth"
trainer, pipeline = load_pipeline(training_config, checkpoint_path)

def predict_normals(image: List[Image.Image], guidance_scale=2., do_rotate=True, num_inference_steps=30, **kwargs):    
    global pipeline
    pipeline = pipeline.to("cuda")
    
    img_list = image if isinstance(image, list) else [image]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    images = trainer.pipeline_forward(
        pipeline=pipeline,
        image=img_list,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, 
        **kwargs
    ).images
    images = simple_remove(images)
    if do_rotate and len(images) > 1:
        images = rotate_normals_torch(images, return_types='pil')
    return images