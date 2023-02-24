import numpy as np
import torch
import getpass
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from random import randint
from accelerate import Accelerator


class Text_To_Image :
    
    def __init__(self, token, kr_prompt, en_prompt, seed):
        self.token = token
        self.kr_prompt = kr_prompt
        self.en_prompt = en_prompt
        self.seed = seed
        pass
    
    # Text To Image
    def sd_texttoimg_pipeline(self, token):
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision = 'fp16', 
            torch_dtype = torch.float16,
            use_auth_token=token
        ).to(device)

        return pipe
    
    def sd_texttoimg_function(self, pipe, kr_prompt, en_prompt, seed):
        device = "cuda"

        if seed == "":
            seed_no = randint(1, 999999999)
        else:
            seed_no = int(seed)

        generator = torch.Generator(device=device).manual_seed(seed_no)
        with autocast(device):
            image = pipe(prompt=en_prompt, generator=generator)['images'][0]

        print("kr_prompt : ", kr_prompt)
        print("en_prompt : ", en_prompt)
        print("seed : ", seed_no)
        return image
    
    
