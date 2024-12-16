
import torch
import torchvision
from tqdm import tqdm
from ip_data import dataloader

DEVICE = 'cuda'
DTYPE = torch.bfloat16

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import vae_encode, get_vae_size_scale_factor, vae_decode
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformer_patched import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from inference import load_vae, load_scheduler, load_image_to_tensor_with_resize_and_crop
from transformers import T5EncoderModel, T5Tokenizer


import safetensors.torch

def load_unet(unet_dir): # TODO don't hardcode -- use arg
    
    unet_ckpt_path = 'latest.pt'
    
    # unet_ckpt_path = unet_dir + '/unet_diffusion_pytorch_model.safetensors'

    unet_config_path = unet_dir + "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    if unet_ckpt_path.endswith('.pt'):
        unet_state_dict = torch.load(unet_ckpt_path)
    else:
        unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    print(transformer.load_state_dict(unet_state_dict, strict=False))
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    transformer.training = True
    transformer.train = True
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
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
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
                    orig_num_frames=4,
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

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(np.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def get_loss(sample, unet, scheduler, clip_embed, idx_grid, prompt_embeds, prompt_att_masks):
    noise = torch.randn_like(sample)
    scheduler.set_timesteps(1000, sample, 'cuda')
    u = compute_density_for_timestep_sampling('logit_normal', sample.shape[0], 0, .5).to('cuda')

    # t = scheduler.timesteps[((u*torch.rand(sample.shape[0],))*1000).long()]
    # t = torch.rand((sample.shape[0],), device=sample.device)
    # TODO seems we need to sample reasonably & probs also shift.
    # print(sample.shape)
    t = scheduler.one_shift_timestep(sample, u)
    # t = u
    # t = scheduler.shift_timesteps(sample, u)
    assert all(t >= 0) and all (t <= 1), f'{t} has < 0 and > 1'

    print(t)
    assert not torch.isnan(t).any(), 'timestep NaNs'
    noised_sample = scheduler.add_noise(sample, noise, t)

    model_out = unet(noised_sample, idx_grid, timestep=t, 
                    encoder_hidden_states=None, encoder_attention_mask=None,
                    cross_attention_kwargs={'clip_embed': clip_embed, "ip_scale": 1},
                    return_dict=False)

    true_out = noise - sample
    
    assert not torch.isnan(noised_sample).any(), 'noised_sample NaNs'
    assert not torch.isnan(model_out).any(), 'model_out NaNs'
    assert not torch.isnan(true_out).any(), 'true_out NaNs'

    loss = torch.nn.functional.mse_loss(model_out[:, :512], true_out[:, :512])
    return loss

def callbackfn(pipe, i, t, noise_pred, latents):
    sigma = t
    pred_original_sample = latents / ((1.0 - sigma)) - sigma*noise_pred / (1.0 - sigma)
    _, sp, _ = get_vae_size_scale_factor(pipe.vae)
    latents = pipe.patchifier.unpatchify(
            latents=pred_original_sample,
            output_height=512//sp,
            output_width=512//sp,
            output_num_frames=4,
            out_channels=pipe.transformer.in_channels
            // np.prod(pipe.patchifier.patch_size),
        )
    im = vae_decode(latents, pipe.vae, is_video=True, vae_per_channel_normalize=True)
    print(im.shape)
    im = im[:, :, 0].to('cpu').float()
    # torchvision.transforms.ToPILImage()(im)
    pipe.image_processor.postprocess(im, output_type='pil')[0].save(f'latest/latest_{i}.png')

@torch.no_grad()
def val(unet, patchifier, vae, scheduler, clip_model, text_encoder, tokenizer, it=0):
    
    media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            'assets/5o.png', 512, 512
        )
    clip_media = (media_items_prepad + 1) / 2
    clip_media = (torch.nn.functional.interpolate(clip_media.squeeze().to('cuda')[None], (224, 224)) - .45) / .26
    # print(clip_media.min(), clip_media.max(), clip_media.shape)
    val_clip_embed = clip_model.encode_image(clip_media)

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
    )

    pipeline = LTXVideoPipeline(**submodel_dict)

    generator.manual_seed(7)
    pipeline(prompt='', 
             negative_prompt='',
             num_frames=25, 
             num_inference_steps=40, 
             clip_embed=val_clip_embed.to(torch.float), 
             ip_scale=1,
             guidance_scale=2,
             vae_per_channel_normalize=True,
             height=512,
             width=512,
             is_video=True,
             frame_rate=24,
             generator=generator,
             )[0][0].save(f'outputs/mary_{it}_.png')
    
    generator.manual_seed(7)
    pipeline(prompt='', 
             negative_prompt='',
             num_frames=25, 
             num_inference_steps=40, 
             clip_embed=val_clip_embed.to(torch.float), 
             ip_scale=1,
             guidance_scale=2,
             vae_per_channel_normalize=True,
             height=512,
             width=512,
             is_video=True,
             frame_rate=24,
            #  generator=generator,
             )[0][0].save(f'outputs/mary_lower{it}_.png')
    
    media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            'assets/6o.png', 512, 512
        )
    # print(media_items_prepad.min(), media_items_prepad.max(), 'should be -1 to 1')
    clip_media = (media_items_prepad + 1) / 2
    clip_media = (torch.nn.functional.interpolate(clip_media.squeeze().to('cuda')[None], (224, 224)) - .45) / .26
    print(clip_media.min(), clip_media.max(), clip_media.shape)
    val_clip_embed = clip_model.encode_image(clip_media)

    generator.manual_seed(7)
    pipeline(prompt='', 
             negative_prompt='',
             num_frames=25, 
             num_inference_steps=40, 
             clip_embed=val_clip_embed.to(torch.float), 
             ip_scale=1,
             guidance_scale=4,
             vae_per_channel_normalize=True,
             height=512,
             width=512,
             is_video=True,
             frame_rate=24,
             generator=generator,
             callback_on_step_end=callbackfn,
             )[0][0].save(f'outputs/cir_{it}_.png')
    pipeline(prompt='', 
             negative_prompt='',
             num_frames=25, 
             num_inference_steps=40, 
             clip_embed=val_clip_embed.to(torch.float), 
             ip_scale=1,
             guidance_scale=4,
             vae_per_channel_normalize=True,
             height=512,
             width=512,
             is_video=True,
             frame_rate=24,
            #  generator=generator,
             )[0][0].save(f'outputs/cir_rng_{it}_.png')

def get_p_embeds(tokenizer, text_encoder, prompt):
    text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=128,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.to('cuda')

    prompt_embeds = text_encoder(
        text_input_ids.to('cuda'), attention_mask=prompt_attention_mask
    )
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds, prompt_attention_mask


def main():
    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder", 
    )
    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )

    # could use empty or zeroed
    positive, positive_mask = torch.load('prompt_embed').to('cuda', torch.float32), torch.load('prompt_mask').to('cuda', torch.float32) #get_p_embeds(tokenizer, text_encoder, '')
    empty, empty_mask = torch.load('prompt_embed').to('cuda', torch.float32), torch.load('prompt_mask').to('cuda', torch.float32) #get_p_embeds(tokenizer, text_encoder, '')

    # torch.save(positive, 'prompt_embed')
    # torch.save(positive_mask, 'prompt_mask')

    # empty, empty_mask = torch.zeros_like(empty), torch.zeros_like(empty_mask)
    # empty, empty_mask = torch.zeros_like(positive), torch.zeros_like(positive_mask)



    text_encoder.to('cpu')
    torch.cuda.empty_cache()

    clip_model = get_clip()

    vae = load_vae(vae_dir).to(DEVICE).to(DTYPE).requires_grad_(False)
    # unet is a misnomer. oh well
    unet = load_unet(unet_dir).to(DEVICE)
    unet.enable_gradient_checkpointing()

    scheduler = load_scheduler(scheduler_dir)
    patchifier = SymmetricPatchifier(patch_size=1)

    params = [p for n, p in unet.named_parameters() if 'tha_ip' in n]# or 'to_q' in n]
    # scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(params=params, lr=4e-5, weight_decay=.0001) # TODO configs

    video_scale_factor, vae_scale_factor, _ = get_vae_size_scale_factor(vae)
    latent_frame_rate = (24) / video_scale_factor
    

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
            
            prompt_embeds = positive.repeat(sample.shape[0], 1, 1)
            prompt_att_masks = positive_mask.repeat(sample.shape[0], 1, 1)

            empty_embeds = empty.repeat(sample.shape[0], 1, 1)
            empty_att_mask = empty_mask.repeat(sample.shape[0], 1, 1)

            drop_or_nah = torch.rand((prompt_embeds.shape[0])) < .1
            prompt_embeds[drop_or_nah] = empty_embeds[drop_or_nah]
            prompt_att_masks[drop_or_nah] = empty_att_mask[drop_or_nah]

            # print(sample.min(), sample.max(), '0 to 1')
            clip_media = (torch.nn.functional.interpolate(sample, (224, 224)) - .45) / .26
            clip_embed = clip_model.encode_image(clip_media)
            # print(clip_media.min(), clip_media.max(), clip_media.shape)
            drop_or_nah = torch.rand((clip_embed.shape[0])) < .1
            clip_embed[drop_or_nah] = torch.zeros_like(clip_embed[drop_or_nah])


            sample = sample.unsqueeze(2).to(DTYPE) * 2 - 1
            # before_pad = np.random.randint(0, 25)
            # after_pad = 24 - before_pad
            # sample = torch.nn.functional.pad(sample, (0, 0, 0, 0, before_pad, after_pad), mode='constant', value=-1)
            sample = sample.repeat(1, 1, 25, 1, 1)
            assert sample.shape[2] == 25, f'{sample.shape}'

            # print(sample.min(), sample.max(), '-1 to 1')

            # TODO pad with -1?
            sample = vae_encode(sample, 
                                vae, 
                                vae_per_channel_normalize=True, 
                                split_size=sample.shape[0],
                                ) # my life is like a movie :)

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

            # with torch.cuda.amp.autocast(dtype=DTYPE):
            loss = get_loss(sample.float(), unet, scheduler, clip_embed.float(), idx_grid, prompt_embeds, prompt_att_masks)
            print(loss.item())
            # loss = scaler.scale(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if (ind) % 400 == 0:
                if ind > 0:
                    torch.save(unet.state_dict(), './latest.pt')

if __name__ == '__main__':
    main()
