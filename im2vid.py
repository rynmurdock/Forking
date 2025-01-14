

# TODO unify/merge origin and this
# TODO save & restart from (if it exists) dataframe parquet
import torch

# lol
DEVICE = 'cuda'
STEPS = 8
output_hidden_state = False
device = "cuda"
dtype = torch.bfloat16

from transformers import T5EncoderModel, T5Tokenizer, BitsAndBytesConfig


from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from inference import load_unet, load_vae, load_scheduler, load_image_to_tensor_with_resize_and_crop

import spaces

import matplotlib.pyplot as plt
import matplotlib
import logging

import os
import imageio
import imageio.v3 as iio
import gradio as gr
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import sched
import threading

import random
import time
from PIL import Image
# from safety_checker_improved import maybe_nsfw


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import spaces
start_time = time.time()

####################### Setup Model

from PIL import Image
import uuid
import av

def write_video(file_name, images, fps=24):
    container = av.open(file_name, mode="w")

    # TODO h264?
    stream = container.add_stream("mpeg4", rate=fps)
    stream.options = {'preset': 'faster'}
    stream.thread_count = 0
    stream.width = 512
    stream.height = 512
    stream.pix_fmt = "yuv420p"

    for img in images:
        img = np.array(img) * 255
        img = np.round(img).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)
    # Close the file
    container.close()

def imio_write_video(file_name, images, fps=24):
    writer = imageio.get_writer(file_name, fps=fps)

    for im in images:
        writer.append_data(np.array(im))
    writer.close()




@spaces.GPU()
def generate_gpu(control_vector=None, negative_prompt='', prompt='a photo', media_items=None, inference_steps=40, num_frames=1, alpha=.5, frame_rate=24):
    if media_items is not None:
        media_items = load_image_to_tensor_with_resize_and_crop(media_items, 512, 512)

    assert control_vector is None or isinstance(control_vector, tuple)

    # Prepare input for the pipeline
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
    }

    # generator = torch.Generator(
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # ).manual_seed(8)
    images, activations = pipe(
        num_inference_steps=inference_steps, # TODO MAKE THIS FAST AGAIN
        num_images_per_prompt=1,
        guidance_scale=4,
        # generator=generator,
        output_type='pt',
        callback_on_step_end=None,
        height=512,
        width=512,
        num_frames=num_frames,
        frame_rate=frame_rate,
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(
            ConditioningMethod.FIRST_FRAME
            if media_items is not None
            else ConditioningMethod.UNCONDITIONAL
        ),
        mixed_precision=True,
        control_vector=control_vector,
        alpha=alpha
    )
                                            # COND vector timestep, channels
    return images[0]

def generate(t, nt, im=None):
    if im is not None:
      im = imio_write_video('i.mp4', np.repeat(np.array(im[None]), 2, axis=0))
      im = iio.imread("i.mp4", )
      im = im[0]
      print(im.shape)
      iio.imwrite('i.png', image=im)
    vid = generate_gpu(prompt=t, negative_prompt=nt, media_items='i.png' if im is not None else None)
    imio_write_video('o.mp4', vid.permute(1,2,3,0).to('cpu', torch.float32))
    return 'o.mp4'

with gr.Blocks() as demo:
  with gr.Row():
        img = gr.Video(
        label='Lightning',
        autoplay=True,
        interactive=False,
        height=512,
        width=512,
        include_audio=False,
        elem_id="video_output",
        #type='filepath',
       )
       # img.play(l, js='''document.querySelector('[data-testid="Lightning-player"]').loop = true''')

        b1 = gr.Text(value='', label='prompt', interactive=True, elem_id="prompt")
        b3 = gr.Text(value='', label='negative', interactive=True, elem_id="negative_prompt")
        b2 = gr.Button(value='Submit', interactive=True, elem_id="submit", visible=True, )
        im = gr.Image(interactive=True, elem_id="image", visible=True,)
        b2.click(generate, [b1, b3, im], img)


def load_pipeline():
    ckpt_dir = '/home/ryn_mote/Misc/ltx_video/ltx-weights/'
    unet_dir = ckpt_dir + "unet/"
    vae_dir = ckpt_dir + "vae/"
    scheduler_dir = ckpt_dir + "scheduler/"

    # Load models
    vae = load_vae(vae_dir)
    unet = load_unet(unet_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder",# quantization_config = BitsAndBytesConfig(load_in_8bit=True),
    ).requires_grad_(False)

    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )
    scheduler = load_scheduler(scheduler_dir)
    patchifier = SymmetricPatchifier(patch_size=1)
    
    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": unet,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    if torch.cuda.is_available():
        # pipeline.transformer = torch.compile(pipeline.transformer.to("cuda")).requires_grad_(False)
        pipeline.transformer = pipeline.transformer.to("cuda").to(torch.bfloat16).requires_grad_(False)
        pipeline.vae = pipeline.vae.to("cuda").to(torch.bfloat16).requires_grad_(False)
        pipeline.text_encoder = pipeline.text_encoder.to('cuda', torch.bfloat16).requires_grad_(False)
    # TODO compile model
    return pipeline

pipe = load_pipeline()



demo.launch(share=True,)


