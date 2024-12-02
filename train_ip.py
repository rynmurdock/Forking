
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
from ltx_video.models.transformer_patched import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from inference import load_vae, load_scheduler, load_image_to_tensor_with_resize_and_crop
from transformers import T5EncoderModel, T5Tokenizer


import safetensors.torch

def load_unet(unet_dir): # TODO don't hardcode -- use arg
    
    # unet_ckpt_path = './latest.pt'
    
    unet_ckpt_path = unet_dir + '/unet_diffusion_pytorch_model.safetensors'

    unet_config_path = unet_dir + "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    if unet_ckpt_path.endswith('.pt'):
        unet_state_dict = torch.load(unet_ckpt_path)
    else:
        unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=False)
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    return transformer


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


#######################

def get_grid(patchifier, latents, scale_grid):
    return patchifier.get_grid(
                    orig_num_frames=2,
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
    zeros_like_t5 = torch.zeros(sample.shape[0], 32, 4096, device=sample.device, dtype=sample.dtype)
    one_like_att = torch.ones(sample.shape[0], 32, device=sample.device, dtype=sample.dtype)
    noise = torch.randn_like(sample)
    # TODO test if t=1 is viable
    t = torch.randint(2, 100, (sample.shape[0],)).to(sample.device)
    t = scheduler.shift_timesteps(sample, t)
    noised_sample = scheduler.add_noise(sample, noise, t)

    model_out = unet(noised_sample, idx_grid, timestep=t, 
                     encoder_hidden_states=zeros_like_t5, encoder_attention_mask=one_like_att,
                     cross_attention_kwargs={'clip_embed': clip_embed, "ip_scale": 1},
                     return_dict=False)
    true_out = noise - noised_sample
    
    assert not torch.isnan(model_out).any(), 'model_out NaNs'
    assert not torch.isnan(true_out).any(), 'true_out NaNs'

    loss = torch.nn.functional.mse_loss(model_out, true_out)
    return loss

def val(unet, patchifier, vae, scheduler, clip_model, text_encoder, tokenizer, it=0):
    
    media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            'i.png', 512, 512
        )
    clip_media = (media_items_prepad + 1) / 2
    val_clip_embed = clip_model.encode_image((torch.nn.functional.interpolate(clip_media.squeeze().to('cuda')[None], (224, 224)) - .45) / .26).to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": unet,
        "patchifier": patchifier,
        "scheduler": scheduler,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(7)

    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline(prompt='', 
             num_frames=9, 
             num_inference_steps=40, 
             clip_embed=val_clip_embed.to(torch.float), 
             ip_scale=1,
             guidance_scale=9,
             vae_per_channel_normalize=True,
             height=512,
             width=512,
             is_video=True,
             frame_rate=24,
             generator=generator,
             )[0][0].save(f'outputs/mary_{it}_.png')
    
    media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            'assets/6o.png', 512, 512
        )
    clip_media = (media_items_prepad + 1) / 2
    val_clip_embed = clip_model.encode_image((torch.nn.functional.interpolate(clip_media.squeeze().to('cuda')[None], (224, 224)) - .45) / .26).to(torch.bfloat16)

    pipeline(prompt='', 
             num_frames=9, 
             num_inference_steps=40, 
             clip_embed=val_clip_embed.to(torch.float), 
             ip_scale=1,
             guidance_scale=9,
             vae_per_channel_normalize=True,
             height=512,
             width=512,
             is_video=True,
             frame_rate=24,
             generator=generator,
             )[0][0].save(f'outputs/cir_{it}_.png')
    pipeline(prompt='', 
             num_frames=9, 
             num_inference_steps=40, 
             clip_embed=val_clip_embed.to(torch.float), 
             ip_scale=1,
             guidance_scale=9,
             vae_per_channel_normalize=True,
             height=512,
             width=512,
             is_video=True,
             frame_rate=24,
            #  generator=generator,
             )[0][0].save(f'outputs/cir_rng_{it}_.png')


def main():
    clip_model = get_clip()
    vae = load_vae(vae_dir).to(DEVICE).to(DTYPE).requires_grad_(False)
    # unet is a misnomer. oh well
    unet = load_unet(unet_dir).to(DEVICE)
    unet.enable_gradient_checkpointing()
    scheduler = load_scheduler(scheduler_dir)
    patchifier = SymmetricPatchifier(patch_size=1)

    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder", 
    )
    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )

    params = [p for n, p in unet.named_parameters() if 'tha_ip' in n]
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(params=params, lr=1e-5) # TODO configs

    video_scale_factor, vae_scale_factor, _ = get_vae_size_scale_factor(vae)
    latent_frame_rate = 24 / video_scale_factor
    

    for epoch in range(1000):
        for ind, sample in tqdm(enumerate(dataloader)):
            if ind % 100 == 0:
                val(unet, patchifier, vae, scheduler, clip_model, text_encoder, tokenizer, ind)
            
            if sample is None:
                continue
            if torch.isnan(sample).any():
                print('NaN in sample')
                continue

            sample = sample.to(DEVICE)

            clip_embed = clip_model.encode_image((torch.nn.functional.interpolate(sample, (224, 224)) - .45) / .26)
            drop_or_nah = torch.rand((clip_embed.shape[0])) < .1
            clip_embed[drop_or_nah] = torch.zeros_like(clip_embed[drop_or_nah])

            sample = vae_encode(sample.unsqueeze(2).repeat(1, 1, 9, 1, 1).to(DTYPE)*2-1, 
                                vae, 
                                vae_per_channel_normalize=True, 
                                split_size=4) # my life is like a movie :)


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

            if (ind) % 200 == 0:
                if ind > 0:
                    torch.save(unet.state_dict(), './latest.pt')

if __name__ == '__main__':
    main()
