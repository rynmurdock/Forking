
import torch
from tqdm import tqdm
from ip_data import dataloader

DEVICE = 'cuda'
DTYPE = torch.bfloat16

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import vae_encode, get_vae_size_scale_factor
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from inference import load_unet, load_vae, load_scheduler, load_image_to_tensor_with_resize_and_crop

import imageio
import numpy as np
import time


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

start_time = time.time()

####################### Setup Model

from PIL import Image
import uuid
import av

import clip

def get_clip(): # TODO give as input
    model, preprocess = clip.load("ViT-B/16", device=DEVICE)
    return model.requires_grad_(False)# doesn't cast the layernorm smh smh .to(DTYPE)

def write_video(file_name, images, fps=24):
    container = av.open(file_name, mode="w")

    # TODO h264?
    stream = container.add_stream("h264", rate=fps)
    stream.options = {'preset': 'faster'}
    stream.thread_count = 0
    stream.width = 768
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



def generate_gpu(control_vector, prompt='a photo', media_items=None, inference_steps=20, num_frames=57, alpha=.5, frame_rate=24):
    if media_items is not None:
        media_items = load_image_to_tensor_with_resize_and_crop(media_items, 512, 768)

    assert control_vector is None or isinstance(control_vector, tuple)

    # Prepare input for the pipeline
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": 'still low quality image',
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
    }

    # generator = torch.Generator(
    #     device=device if torch.cuda.is_available() else "cpu"
    # ).manual_seed(8)
    images, activations = pipe(
        num_inference_steps=inference_steps, # TODO MAKE THIS FAST AGAIN
        num_images_per_prompt=1,
        guidance_scale=4,
        # generator=generator,
        output_type='pt',
        callback_on_step_end=None,
        height=512,
        width=768,
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
    return images[0].permute(1, 2, 3, 0), activations.squeeze().to('cpu', torch.float32)

@torch.no_grad()
def generate(in_im_embs, prompt='the scene'):
    output, im_emb = generate_gpu(in_im_embs, prompt)

    # TODO this model may need the filter back!!! # TODO
    nsfw = False#maybe_nsfw(output.images[0])

    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.mp4"

    if nsfw:
        gr.Warning("NSFW content detected.")
        # TODO could return an automatic dislike of auto dislike on the backend for neither as well; just would need refactoring.
        return None, im_emb
    
    # output.images[0].save(path)
    imio_write_video(path, output.to('cpu', torch.float32))
    return path, im_emb


#######################

def get_grid(patchifier, latents, scale_grid):
    return patchifier.get_grid(
                    orig_num_frames=1,
                    orig_height=latents.shape[-2],
                    orig_width=latents.shape[-1],
                    batch_size=latents.shape[0],
                    scale_grid=scale_grid, # TODO this may matter
                    device=latents.device,
                )


ckpt_dir = '/home/ryn_mote/Misc/ltx_video/ltx-weights/'
unet_dir = ckpt_dir + "unet/"
vae_dir = ckpt_dir + "vae/"
scheduler_dir = ckpt_dir + "scheduler/"

def get_loss(sample, unet, scheduler, clip_embed, idx_grid):
    # sample = sample * scheduler.init_noise_sigma
    zeros_like_t5 = torch.zeros(sample.shape[0], 1, 4096, device=sample.device, dtype=sample.dtype)
    zeros_like_att = torch.zeros(sample.shape[0], 1, device=sample.device, dtype=sample.dtype)
    noise = torch.randn_like(sample)
    t = torch.randint(0, 1000, (sample.shape[0],)).to(sample.device)
    noised_sample = scheduler.add_noise(sample, noise, t)

    model_out = unet(noised_sample, idx_grid, timestep=t, 
                     encoder_hidden_states=zeros_like_t5, encoder_attention_mask=zeros_like_att,
                     cross_attention_kwargs={'clip_embed': clip_embed},
                     return_dict=False)

    true_out = noise - noised_sample
    loss = torch.nn.functional.mse_loss(model_out, true_out)
    return loss

def main():
    clip_model = get_clip()
    vae = load_vae(vae_dir).to(DEVICE).to(DTYPE).requires_grad_(False)
    # unet is a misnomer. oh well
    unet = load_unet(unet_dir).to(DEVICE)
    scheduler = load_scheduler(scheduler_dir)
    patchifier = SymmetricPatchifier(patch_size=1)

    params = [p for n, p in unet.named_parameters() if 'tha_ip' in n]
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(params=params, lr=1e-5) # TODO configs

    video_scale_factor, vae_scale_factor, _ = get_vae_size_scale_factor(vae)
    latent_frame_rate = 24 / video_scale_factor
    

    for epoch in range(1000):
        for ind, sample in tqdm(enumerate(dataloader)):
            if sample is None:
                continue
            sample = sample.to(DEVICE)
            clip_embed = clip_model.encode_image((torch.nn.functional.interpolate(sample, (224, 224)) - .45) / .26)

            sample = vae_encode(sample.unsqueeze(2).to(DTYPE)*2-1, vae) # my life is like a movie :)


            latent_frame_rates = (
                    torch.ones(
                        sample.shape[0], 1, device=sample.device
                    )
                    * latent_frame_rate
                )
            scale_grid = (
                    (
                        1 / latent_frame_rates ,
                        vae_scale_factor,
                        vae_scale_factor,
                    )
                    if unet.use_rope
                    else None
                )
                
            idx_grid = get_grid(patchifier, sample, scale_grid)
            sample = patchifier.patchify(latents=sample)

            with torch.cuda.amp.autocast(dtype=DTYPE):
                loss = get_loss(sample, unet, scheduler, clip_embed, idx_grid)
            print(loss.item())

            scaler.scale(loss).backward()
            optimizer.step()
            optimizer.zero_grad()

            if (ind) % 100 == 0:
                if ind > 0:
                    torch.save(unet.state_dict(), './latest.pt')
                #val(config.val_image_paths)

if __name__ == '__main__':
    main()
