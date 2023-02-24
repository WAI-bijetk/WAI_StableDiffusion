import numpy as np
import torch
import getpass
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from random import randint
from accelerate import Accelerator
import argparse
import os

class Image_To_Image :
    
    def __init__(self, token, file_name, prompt, strength, seed):
        self.token = token
        self.file_name = file_name
        self.prompt = prompt      
        self.strength = strength
        self.seed = seed
        
    
    # Text To Image
    def sd_imgtoimg_pipeline(self, token):
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device
        
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=token
        ).to(device)
        
        return pipe
    
    def sd_imgtoimg_function(self, pipe, prompt, file_name, strength, seed):
        image = Image.open(file_name).convert("RGB").resize((512,512), resample=Image.LANCZOS)

        device = "cuda"

        if seed == "" or seed == None:
            seed_no = randint(1, 999999999)
        else:
            seed_no = int(seed)

        generator = torch.Generator(device=device).manual_seed(seed_no)
        with autocast(device):
            image = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=7.5, generator=generator).images[0]
        
        print("kr_prompt : ", prompt)    
        print("seed : ", seed_no)
        
        output_path = os.getcwd() + "/ImagetoImage"
        print(output_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        image.save(output_path + f"/I2I_{prompt}_{seed_no}.png", "png")
        return image
    
def image_to_image(token, prompt, file_name, strength, seed):
    
    diffusion = Image_To_Image(token, file_name, prompt, strength, seed)
    
    try:
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
    except:
        pipe_i2i = diffusion.sd_imgtoimg_pipeline(token)
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
        
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        type=str
    )
    parser.add_argument(
        "--prompt",
        type=str
    )
    parser.add_argument(
        "--file_name",
        type=str
    )
    parser.add_argument(
        "--seed",
        default = None,
        type=str
    )
    parser.add_argument(
        "--strength",
        type=str,
        default="0.6"
    )
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    image = image_to_image(args.token, args.prompt, args.file_name, float(args.strength), args.seed)
    return image


if __name__ == "__main__":
    main()