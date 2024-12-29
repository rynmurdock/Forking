

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

from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.utils.conditioning_method import ConditioningMethod
from ip_inference import load_unet, load_vae, load_scheduler

import spaces
import torchvision

import os
import imageio
import gradio as gr
import numpy as np
import pandas as pd

from apscheduler.schedulers.background import BackgroundScheduler
import logging

import time
from PIL import Image
# from safety_checker_improved import maybe_nsfw

import clip



def get_clip(): # TODO give as input
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    return model.requires_grad_(False)# doesn't cast the layernorm smh smh .to(DTYPE)

clip_model = get_clip()

def embed_image(sample):
    # [0, 1]
    clip_media = (torch.nn.functional.interpolate(sample, (224, 224)) - .45) / .26
    clip_embed = clip_model.encode_image(clip_media)
    return clip_embed


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

prevs_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'from_user_id', 'text', 'gemb'])

start_time = time.time()

# prompt_list = [p for p in list(set(
#                 pd.read_csv('twitter_prompts.csv').iloc[:, 1].tolist())) if type(p) == str]


####################### Setup Model

from PIL import Image
import uuid
import av

def write_video(file_name, images, fps=24):
    container = av.open(file_name, mode="w")

    # TODO h264?
    stream = container.add_stream("h264", rate=fps)
    stream.options = {'preset': 'faster'}
    stream.thread_count = 0
    stream.width = 768
    stream.height = 768
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

def path_to_tensor(path):
    return torchvision.transforms.ToTensor()(Image.open(path)).unsqueeze(0).to('cuda')[:, :3]


global_ref = embed_image(path_to_tensor('assets/1o.png')).squeeze() # TODO a reference image

def proj(b, a):
    result = b * torch.dot(a, b) / torch.dot(b, b)
    return result

# ~from https://arxiv.org/abs/2411.09003
def solver(embs, ys, ref, alpha=1, alpha_n=1):
    pos = [e for e, y in zip(embs, ys) if y == 1]
    neg = [e for e, y in zip(embs, ys) if y == 0]
    pos_t = torch.stack(pos)
    neg_t = torch.stack(neg)
    r = pos_t.mean(0) - neg_t.mean(0)
    r_n = neg_t.mean(0) - pos_t.mean(0)
    v_reference = (global_ref) # may not want to use regular refs, because may not have typical magnitudes.
    v_prime = v_reference - proj(r_n, v_reference) + proj(r, neg_t.mean(0)) + alpha * r - alpha_n * r_n
    return v_prime.to('cpu', dtype=torch.float32).unsqueeze(0)


# num_frames up makes quality up. As you'd kinda expect
def generate_gpu(clip_embed, prompt='', inference_steps=30, num_frames=81, frame_rate=24):

    # Prepare input for the pipeline
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": None,
        "negative_prompt_attention_mask": None,
        'clip_embed': clip_embed.to('cuda'),
    }

    # generator = torch.Generator(
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # ).manual_seed(8)
    images = pipe(
        num_inference_steps=inference_steps, # TODO MAKE THIS FAST AGAIN
        num_images_per_prompt=1,
        guidance_scale=7,
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
            ConditioningMethod.UNCONDITIONAL
        ),
        mixed_precision=True,
    )[0]

    out_clip = embed_image(images[:, :, 3])
    
    # loop video
    # images = torch.cat([images, images.flip(2)], 2)

    return images[0].permute(1, 2, 3, 0), out_clip

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
    return path, im_emb.to('cpu')


#######################





def get_user_emb(embs, ys):
    # sample only as many negatives as there are positives
    indices = range(len(ys))                
    pos_indices = [i for i in indices if ys[i] > .5]
    neg_indices = [i for i in indices if ys[i] <= .5]
    
    mini = min(len(pos_indices), len(neg_indices))

    if len(ys) > 20: # drop earliest of whichever of neg or pos is most abundant
        if len(pos_indices) > len(neg_indices):
            ind = pos_indices[0]
        else:
            ind = neg_indices[0]
        ys.pop(ind)
        embs.pop(ind)
        print('Dropping at 20')
    
    if mini < 1:
        feature_embs = torch.cat([torch.randn(8, 768).to(torch.half), 
                                  torch.randn(8, 768).to(torch.half)]) # TODO verify shape is same as CLIP L image embed
        ys_t = [0, 1]
        print('Not enough ratings.')
    else:
        indices = range(len(ys))
        ys_t = [ys[i] for i in indices]
        feature_embs = torch.stack([embs[e].detach().cpu() for e in indices]).squeeze()
        
        # scaler = preprocessing.StandardScaler().fit(feature_embs)
        # feature_embs = scaler.transform(feature_embs)
        # ys_t = ys
        
        # print(np.array(feature_embs).shape, np.array(ys_t).shape)
    
    positives = [f for f, y in zip(feature_embs, ys) if y == 1]
    rand_positive = positives[np.random.randint(0, len(positives))]

    sol = solver(feature_embs.to('cuda'), torch.tensor(ys_t).to('cuda'), ref=rand_positive.to('cuda')) # TODO may not be opt.
    # TODO we should have clear i/o shapes & assert.

    # dif = torch.tensor(sol, dtype=dtype).to(device)
    
    # # could j have a base vector of a black image
    # latest_pos = (random.sample([feature_embs[i] for i in range(len(ys_t)) if ys_t[i] > .5], 1)[0]).to(device, dtype)

    # dif = ((dif / dif.std()) * latest_pos.std())

    # sol = (1*latest_pos + 3*dif)/4
    return sol


def pluck_img(user_id, user_emb):
    not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
    while len(not_rated_rows) == 0:
        not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
        time.sleep(.1)
    # TODO optimize this lol
    best_sim = -100000
    for i in not_rated_rows.iterrows():
        # TODO sloppy .to but it is 3am.
        a = i[1]['embeddings']
        assert a.shape == user_emb.shape and len(a.shape) == 2, f'{a.shape} must = {user_emb.shape} and shape must be 2 long'
        sim = torch.cosine_similarity(a.detach().to('cpu'), user_emb.detach().to('cpu'), 1)
        if sim > best_sim:
            best_sim = sim
            best_row = i[1]
    img = best_row['paths']
    return img


def background_next_image():
        global prevs_df
        # only let it get N (maybe 3) ahead of the user
        #not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
        rated_rows = prevs_df[[i[1]['user:rating'] != {' ': ' '} for i in prevs_df.iterrows()]]
        if len(rated_rows) < 4:
            time.sleep(.1)
        #    not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
            return

        user_id_list = set(rated_rows['latest_user_to_rate'].to_list())
        for uid in user_id_list:
            rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is not None for i in prevs_df.iterrows()]]
            not_rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is None for i in prevs_df.iterrows()]]
            
            # we need to intersect not_rated_rows from this user's embed > 7. Just add a new column on which user_id spawned the 
            #   media. 
            
            unrated_from_user = not_rated_rows[[i[1]['from_user_id'] == uid for i in not_rated_rows.iterrows()]]
            rated_from_user = rated_rows[[i[1]['from_user_id'] == uid for i in rated_rows.iterrows()]]

            # we pop previous ratings if there are > n
            if len(rated_from_user) >= 15:
                oldest = rated_from_user.iloc[0]['paths']
                prevs_df = prevs_df[prevs_df['paths'] != oldest]
            # we don't compute more after n are in the queue for them
            if len(unrated_from_user) >= 10:
                continue
            
            if len(rated_rows) < 5:
                continue
            
            embs, ys = pluck_embs_ys(uid)
            
            user_emb = get_user_emb(embs, [y[1] for y in ys])
            

            global glob_idx
            glob_idx += 1
            if glob_idx >= 1000:
                glob_idx = 0


            # if glob_idx % 2 == 0:
            #     text = prompt_list[glob_idx]
            # else:
            text = 'artistic video' # TODO unused
            print(text)
            img, embs = generate(user_emb, text)
            
            if img:
                tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'text', 'gemb'])
                tmp_df['paths'] = [img]
                tmp_df['embeddings'] = [embs]
                tmp_df['user:rating'] = [{' ': ' '}]
                tmp_df['from_user_id'] = [uid]
                tmp_df['text'] = [text]
                prevs_df = pd.concat((prevs_df, tmp_df))
                # we can free up storage by deleting the image
                if len(prevs_df) > 500:
                    oldest_path = prevs_df.iloc[6]['paths']
                    if os.path.isfile(oldest_path):
                        os.remove(oldest_path)
                    else:
                        # If it fails, inform the user.
                        print("Error: %s file not found" % oldest_path)
                    # only keep 50 images & embeddings & ips, then remove oldest besides calibrating
                    prevs_df = pd.concat((prevs_df.iloc[:6], prevs_df.iloc[7:]))
    

def pluck_embs_ys(user_id):
    rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) != None for i in prevs_df.iterrows()]]
    #not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) == None for i in prevs_df.iterrows()]]
    #while len(not_rated_rows) == 0:
    #    not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) == None for i in prevs_df.iterrows()]]
    #    rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) != None for i in prevs_df.iterrows()]]
    #    time.sleep(.01)
    #    print('current user has 0 not_rated_rows')
    
    embs = rated_rows['embeddings'].to_list()
    ys = [i[user_id] for i in rated_rows['user:rating'].to_list()]
    return embs, ys

def next_image(calibrate_prompts, user_id):
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            cal_video = calibrate_prompts.pop(0)
            image = prevs_df[prevs_df['paths'] == cal_video]['paths'].to_list()[0]
            return image, calibrate_prompts, 
        else:
            embs, ys = pluck_embs_ys(user_id)
            ys_here = [y[1] for y in ys]
            user_emb = get_user_emb(embs, ys_here)
            image = pluck_img(user_id, user_emb)
            return image, calibrate_prompts, 









def start(_, calibrate_prompts, user_id, request: gr.Request):
    user_id = int(str(time.time())[-7:].replace('.', ''))
    image, calibrate_prompts  = next_image(calibrate_prompts, user_id)
    return [
            gr.Button(value='👍', interactive=True), 
            # gr.Button(value='Neither (Space)', interactive=True, visible=False), 
            gr.Button(value='👎', interactive=True),
            gr.Button(value='Start', interactive=False),
            image,
            calibrate_prompts,
            user_id,
            
            ]


def choose(img, choice, calibrate_prompts, user_id, request: gr.Request):
    global prevs_df
    
    
    if choice == '👍':
        choice = [1, 1]
    elif choice == 'Neither (Space)':
        img, calibrate_prompts,  = next_image(calibrate_prompts, user_id)
        return img, calibrate_prompts, 
    elif choice == '👎':
        choice = [0, 0]
    elif choice == '👍 Style':
        choice = [0, 1]
    elif choice == '👍 Content':
        choice = [1, 0]
    else:
        assert False, f'choice is {choice}'
    
    # if we detected NSFW, leave that area of latent space regardless of how they rated chosen.
    # TODO skip allowing rating & just continue

    if img is None:
        print('NSFW -- choice is disliked')
        choice = [0, 0]
    
    row_mask = [p.split('/')[-1] in img for p in prevs_df['paths'].to_list()]
    # if it's still in the dataframe, add the choice
    if len(prevs_df.loc[row_mask, 'user:rating']) > 0:
        prevs_df.loc[row_mask, 'user:rating'][0][user_id] = choice
        # print(row_mask, prevs_df.loc[row_mask, 'latest_user_to_rate'], [user_id])
        prevs_df.loc[row_mask, 'latest_user_to_rate'] = [user_id]
    img, calibrate_prompts = next_image(calibrate_prompts, user_id)
    return img, calibrate_prompts, gr.update(interactive=False), gr.update(interactive=False)

css = '''.gradio-container{max-width: 700px !important}
#description{text-align: center}
#description h1, #description h3{display: block}
#description p{margin-top: 0}
.fade-in-out {animation: fadeInOut 3s forwards}
@keyframes fadeInOut {
    0% {
      background: var(--bg-color);
    }
    100% {
      background: var(--button-secondary-background-fill);
    }
}
'''
js_head = '''
<script>
document.addEventListener('keydown', function(event) {
    if (event.key === 'a' || event.key === 'A') {
        // Trigger click on 'dislike' if 'A' is pressed
        document.getElementById('dislike').click();
    } else if (event.key === ' ' || event.keyCode === 32) {
        // Trigger click on 'neither' if Spacebar is pressed
        document.getElementById('neither').click();
    } else if (event.key === 'l' || event.key === 'L') {
        // Trigger click on 'like' if 'L' is pressed
        document.getElementById('like').click();
    }
});
function fadeInOut(button, color) {
  button.style.setProperty('--bg-color', color);
  button.classList.remove('fade-in-out');
  void button.offsetWidth; // This line forces a repaint by accessing a DOM property
  
  button.classList.add('fade-in-out');
  button.addEventListener('animationend', () => {
    button.classList.remove('fade-in-out'); // Reset the animation state
  }, {once: true});
}
document.body.addEventListener('click', function(event) {
    const target = event.target;
    if (target.id === 'dislike') {
      fadeInOut(target, '#ff1717');
    } else if (target.id === 'like') {
      fadeInOut(target, '#006500');
    } else if (target.id === 'neither') {
      fadeInOut(target, '#cccccc');
    }
});

</script>
'''

with gr.Blocks(css=css, head=js_head) as demo:
    gr.Markdown('''# Zahir
### Generative Recommenders for Exporation of Possible Images

Explore the latent space without text prompts based on your preferences. Learn more on [the write-up](https://rynmurdock.github.io/posts/2024/3/generative_recomenders/).
    ''', elem_id="description")
    user_id = gr.State()
    # calibration videos -- this is a misnomer now :D
    calibrate_prompts = gr.State([ # TODO this is nonsense; only have one list lol
    './assets/1o.mp4',
    './assets/2o.mp4',
    './assets/3o.mp4',
    './assets/4o.mp4',
    './assets/5o.mp4',
    './assets/6o.mp4',
    './assets/7o.mp4',
    './assets/8o.mp4',
    './assets/9o.mp4',
    ])
    def l():
        return None

    with gr.Row(elem_id='output-image'):
        img = gr.Video(
        label='Lightning',
        autoplay=True,
        interactive=False,
#        height=512,
#        width=512,
        include_audio=False,
        elem_id="video_output",
        # type='filepath',
       )
        img.play(l, js='''document.querySelector('[data-testid="Lightning-player"]').loop = true''')
    
    def wait():
        time.sleep(2)
    
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='👎', interactive=False, elem_id="dislike")

        # b2 = gr.Button(value='Neither (Space)', interactive=False, elem_id="neither", visible=False)

        b1 = gr.Button(value='👍', interactive=False, elem_id="like")
    with gr.Row(equal_height=True):
        b1.click(
        choose, 
        [img, b1, calibrate_prompts, user_id],
        [img, calibrate_prompts, b1, b3],
        ).then(fn=wait).then(fn=lambda: [gr.update(interactive=True), gr.update(interactive=True)], inputs=None, outputs=[b1, b3])
        
        # b2.click(
        # choose, 
        # [img, b2, calibrate_prompts, user_id],
        # [img, calibrate_prompts, b1, b3],
        # )

        b3.click(
        choose, 
        [img, b3, calibrate_prompts, user_id],
        [img, calibrate_prompts, b1, b3],
        ).then(fn=wait).then(fn=lambda: [gr.update(interactive=True), gr.update(interactive=True)], inputs=None, outputs=[b1, b3])

    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4, calibrate_prompts, user_id],
                 [b1, b3, b4, img, calibrate_prompts, user_id, ]
                 )
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several images and then roam. </ div><br><br><br>

<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

scheduler = BackgroundScheduler()
scheduler.add_job(func=background_next_image, trigger="interval", seconds=.1)
scheduler.start()
logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)


def load_pipeline():
    ckpt_dir = './ltx-weights/'
    unet_dir = ckpt_dir + 'unet/'
    vae_dir = ckpt_dir + "vae/"
    scheduler_dir = ckpt_dir + "scheduler/"

    # Load models
    vae = load_vae(vae_dir)
    unet = load_unet(unet_dir).to(torch.bfloat16)

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
        
        pipeline.transformer = torch.compile(pipeline.transformer.to("cuda")).requires_grad_(False)

        # pipeline.transformer = pipeline.transformer.to("cuda").requires_grad_(False)

        pipeline.vae = pipeline.vae.to("cuda").requires_grad_(False)
        pipeline.text_encoder = pipeline.text_encoder
    return pipeline

pipe = load_pipeline()


# prep our calibration videos
for im, txt in [ # DO NOT NAME THESE JUST NUMBERS! apparently we assign images by number
    
    # TODO cache these
    ('./assets/1o.png', 'artistic vivid scene '),
    ('./assets/2o.png', 'artistic vivid scene '),
    ('./assets/3o.png', 'artistic vivid scene '),
    ('./assets/4o.png', 'artistic vivid scene '),
    ('./assets/5o.png', 'artistic vivid scene '),
    ('./assets/6o.png', 'vivid scene '),
    ('./assets/7o.png', 'vivid scene '),
    ('./assets/8o.png', 'vivid scene '),
    ('./assets/9o.png', 'vivid scene '), # TODO replace with .pt cache of activations & mp4s
    ]:
    tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'text', 'gemb'])
    tmp_df['paths'] = [im.replace('png', 'mp4')]
    image = Image.open(im).convert('RGB')

    ims, im_emb = generate_gpu(
                        clip_embed=embed_image(path_to_tensor(im))
                               )
    
    # TODO cache (this is in the repo at some commit iirc)
    imio_write_video(im.replace('png', 'mp4'), ims.to('cpu', torch.float32))
    
    
    tmp_df['embeddings'] = [im_emb.detach().to('cpu')]
    tmp_df['user:rating'] = [{' ': ' '}]
    tmp_df['text'] = [txt]
    prevs_df = pd.concat((prevs_df, tmp_df))



glob_idx = 0
demo.launch(share=True,)


