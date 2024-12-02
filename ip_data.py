import numpy as np
from datasets import load_dataset
import torch

def my_collate(batch):
    try:
      #raw_text = [item['prompt'] for item in batch]
      tensor_image = torch.cat([torch.nn.functional.interpolate(
                          torch.tensor(np.array(item['image'])).permute(2,0,1)[None] / 255, 
                                                                  (512, 512)) for item in batch])
    except Exception as e:
      print(e)
      return None
    return tensor_image

ds = load_dataset("stylebreeder/stylebreeder", split='2M_sample', streaming=True).shuffle(seed=7, buffer_size=1)
dataloader = torch.utils.data.DataLoader(ds, num_workers=32, collate_fn=my_collate, batch_size=4)
