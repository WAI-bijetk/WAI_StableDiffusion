import numpy as np
import torch
import getpass
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from random import randint
from accelerate import Accelerator


class Text_To_Image :
    
    def __init__(self, token, prompt, seed):
        self.token = token
        self.prompt = prompt
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
    
    def sd_texttoimg_function(self, pipe, prompt, seed):
        device = "cuda"

        if seed == "":
            seed_no = randint(1, 999999999)
        else:
            seed_no = int(seed)

        generator = torch.Generator(device=device).manual_seed(seed_no)
        with autocast(device):
            image = pipe(prompt=prompt, generator=generator)['images'][0]

        return image


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

        if seed == "":
            seed_no = randint(1, 999999999)
        else:
            seed_no = int(seed)

        generator = torch.Generator(device=device).manual_seed(seed_no)
        with autocast(device):
            image = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=7.5, generator=generator).images[0]

        return image
    
###################################################################################    

class Image_Extend:
    
    def __init__(self, token, prompt, file_name, a, b, output_name, seed):
        self.token = token
        self.prompt = prompt
        self.file_name = file_name
        self.a = a
        self.b = b
        self.output_name = output_name
        self.seed = seed
    
    def sd_extend_pipeline(self, token):
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device
        model_id = "runwayml/stable-diffusion-inpainting"

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=token
        ).to(device)
        
        return pipe
    
    def sd_extend_crop_mask(self, file_name, a, b):
        main_img = Image.open(file_name).convert("RGBA")

        main_width, main_height = main_img.size

        extend_width = main_width + (512 * 2)
        extend_height = main_height + (512 * 2)
        extend_square_w = np.full((extend_height, extend_width, 4), (255, 255, 255, 0), dtype=np.uint8)

        main_array = np.array(main_img)
        for width in range(0, main_width):
            for height in range(0, main_height):
                extend_square_w[height+512][width+512] = main_array[height][width]

        extend_main_img = Image.fromarray(extend_square_w)

        # crop extend_main_img
        extend_crop = extend_main_img.crop((a,b,a+512,b+512))
        extend_crop

        # a, b value 검증
        crop_array = np.array(extend_crop)
        zero_count = crop_array[:,:,3].reshape(-1).tolist().count(0)
        if zero_count == 0:
            print("a,b 값 다시 설정 필요.")
            return

        # 5. crop_array와 투명도를 이용하여 마스크 생성
        mask_array = crop_array.copy()
        for i in range(512):
            for j in range(512):
                if mask_array[i][j][3] == 255:
                    mask_array[i][j] = [0,0,0,255]
                else:
                    mask_array[i][j] = [255,255,255,255]
        mask = Image.fromarray(mask_array)

        return extend_main_img, extend_crop, mask
    
    def sd_extend_result_img(self, pipe, prompt, extend_img, image, mask_image, a, b, seed):
        num_samples = 1
        if seed == "":
            seed = randint(0,9999999999)
        else:
            seed = int(seed)
            
        device = "cuda"
        accelerator = Accelerator()
        device = accelerator.device
        generator = torch.Generator(device=device).manual_seed(seed) # change the seed to get different results

        images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=7.5,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images[0]

        extend_img_array = np.array(extend_img)
        images_array = np.array(images.convert("RGBA"))
        for i in range(512):
            for j in range(512):
                extend_img_array[b+i][a+j] = images_array[i][j]

        for_crop_h, for_crop_w = extend_img_array.shape[:2]

        w_list, h_list = [], []

        for h in range(for_crop_h):
            for w in range(for_crop_w):
                pixel = extend_img_array[h][w][3]
                if pixel == 255:
                    w_list.append(w)
                    h_list.append(h)

        result_img = Image.fromarray(extend_img_array)
        final_crop = result_img.crop((min(w_list),min(h_list),max(w_list),max(h_list)))
        return final_crop
    
    def sd_extend_function(self, pipe, file_name, prompt, a, b, output_name = "", seed = ""):
        
        extend_img, image, mask_image = self.sd_extend_crop_mask(file_name, a, b)
        

        final_result = self.sd_extend_result_img(pipe, prompt, extend_img, image, mask_image, a, b, seed)
            
        if output_name == "":
            return final_result
        else:
            final_result.save(output_name, output_name.split(".")[-1])
            return final_result

class FineTuning:
    
    def __init__(self, UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name):
        self.UNet_Training_Steps = UNet_Training_Steps, 
        self.UNet_Learning_Rate = UNet_Learning_Rate,
        self.Text_Encoder_Training_Steps = Text_Encoder_Training_Steps,
        self.Text_Encoder_Learning_Rate = Text_Encoder_Learning_Rate, 
        self.Session_Name = Session_Name
        self.WORKSPACE='/content/gdrive/MyDrive/Fast-Dreambooth'
        self.OUTPUT_DIR="/content/models/"+ Session_Name
        self.SESSION_DIR=self.WORKSPACE+'/Sessions/'+ Session_Name
        self.INSTANCE_DIR=self.SESSION_DIR+'/instance_images'
        self.CAPTIONS_DIR=self.SESSION_DIR+'/captions'
        self.MODEL_NAME="/content/stable-diffusion-v1-5"
        self.PT=""
        pass
    
    # Line Logging
    def line_logging(self, *messages):
        import datetime
        import sys
        today = datetime.datetime.today()
        log_time = today.strftime('[%Y/%m/%d %H:%M:%S]')
        log = []
        for message in messages:
            log.append(str(message))
        print(log_time + '::' + ' '.join(log) + '')
        sys.stdout.flush()
    
    
    # Environment Setting
    def sd_custom_environment(self):
        import os
        import subprocess
        import shutil
        import glob
        from distutils.dir_util import copy_tree
        import time
        self.line_logging("Start Env. Setting")
        os.chdir('/content/')
        subprocess.run(['pip', 'install', '-q', '--no-deps', 'accelerate==0.12.0'])
        subprocess.call (['wget', '-q', '-i', '/', "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/dbdeps.txt"])

        f = open('/content/dbdeps.txt')
        lines = f.readlines()
        for i in range(len(lines)):
            subprocess.call (['wget', '-q', '/', f"https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/deps.{i+1}"])
        for i in range(len(lines)):
            try:
                shutil.move(f"deps.{i+1}", f"deps.zip.00{i+1}")
            except:
                pass

        cmd = ['7z', 'x', 'deps.zip.001']
        sp = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        time.sleep(20)

        file_source = '/content/usr/local/lib/python3.8/dist-packages'
        file_destination = '/usr/local/lib/python3.8/dist-packages'
        copy_tree(file_source, file_destination)
        time.sleep(20)
        
        shutil.rmtree('/content/usr')

        file_list = []
        file_list.extend(glob.glob("*.00*"))
        file_list.extend(glob.glob("*.txt"))
        for file_name in file_list:
            os.remove(file_name)
        subprocess.run(["git", "clone",  "https://github.com/TheLastBen/diffusers", '--depth=1', '--branch=updt'])
        self.line_logging('Done, proceed')
        
    # create Session
    def sd_custom_create_session(self, MODEL_NAME, SESSION_DIR, INSTANCE_DIR):
        import os
        if os.path.exists(str(SESSION_DIR)):
            self.line_logging('Loading session with no previous model, using the original model or the custom downloaded model')
            if MODEL_NAME=="":
                self.line_logging('No model found, use the "Model Download" cell to download a model.')
            else:
                self.line_logging('Session Loaded, proceed to uploading instance images')


        elif not os.path.exists(str(SESSION_DIR)):
            # %mkdir -p "$INSTANCE_DIR"
            os.makedirs(INSTANCE_DIR)
            self.line_logging('Creating session...')
            if MODEL_NAME=="":
                self.line_logging('No model found, use the "Model Download" cell to download a model.')
            else:
                self.line_logging('Session created, proceed to uploading instance images')
                
    # upload_image_replace
    # 이미지, 캡션에 들어간 띄어쓰기를 "-" 로 바꿔주는 함수
    def sd_custom_upload_image_replace(self, directory):
        import shutil
        import glob

        inst_list = glob.glob(directory+"/*")
        for i in inst_list:
            old_name = i.split("/")[-1]
            new_name = old_name.replace(" ", "-")
            shutil.move(directory +"/"+old_name, directory + "/" + new_name)
            
    # Upload Image
    def sd_custom_upload_image(self, SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR):
        import shutil
        import os
        from tqdm import tqdm
        from google.colab import files
        from IPython.display import clear_output
        
        self.line_logging("Start : Upload Image...")
        
        if os.path.exists(str(INSTANCE_DIR)):
                shutil.rmtree(INSTANCE_DIR)
        if os.path.exists(str(CAPTIONS_DIR)):
            shutil.rmtree(CAPTIONS_DIR)

        if not os.path.exists(str(INSTANCE_DIR)):
            os.makedirs(INSTANCE_DIR)
        if not os.path.exists(str(CAPTIONS_DIR)):
            os.makedirs(CAPTIONS_DIR)

        if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
            shutil.rmtree(str(INSTANCE_DIR) + "/.ipynb_checkpoints")
            
        up=""  
        uploaded = files.upload()
        
        # 캡션과 이미지 파일 분리
        for filename in uploaded.keys():
            if filename.split(".")[-1]=="txt":
                shutil.move(filename, CAPTIONS_DIR)
            up=[filename for filename in uploaded.keys() if filename.split(".")[-1]!="txt"]
            
        # 이미지 파일들 INST_DIR로 이동, bar_Format은 막대기 모양인듯
        for filename in tqdm(uploaded.keys(), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
            shutil.move(filename, INSTANCE_DIR)
            clear_output()
        print('upload image Done, proceed to the next cell')
        
        #d 이미지, 캡션 파일 이름의 빈칸을 "-"로 바꿔줌
        for directory in [INSTANCE_DIR, CAPTIONS_DIR]:
            self.sd_custom_upload_image_replace(directory)  
        
        # 파일 압축  
        os.chdir(SESSION_DIR)
        if os.path.exists("instance_images.zip"):
            os.remove("instance_images.zip")
            
        if os.path.exists("captions.zip"):
            os.remove("captions.zip")
        
        shutil.make_archive('instance_images', 'zip', './instance_images')
        shutil.make_archive('captions', 'zip', './captions')
        
        self.line_logging("Done : Upload Image...")
        
    # Model Download (Ver 1.5)
    def sd_custom_model_download(self,):
        self.line_logging("Start : Model Download...")
        
        import shutil
        import os

        if os.path.exists('/content/stable-diffusion-v1-5'):
            shutil.rmtree('/content/stable-diffusion-v1-5')

        os.chdir('/content')
        os.mkdir('/content/stable-diffusion-v1-5')
        os.chdir('/content/stable-diffusion-v1-5')
        os.system('git init')
        os.system('git lfs install --system --skip-repo')
        os.system('''git remote add -f origin  "https://huggingface.co/runwayml/stable-diffusion-v1-5"''')
        os.system("git config core.sparsecheckout true")
        os.system('''echo -e "\nscheduler\ntext_encoder\ntokenizer\nunet\nfeature_extractor\nsafety_checker\nmodel_index.json\n!*.safetensors" > .git/info/sparse-checkout''')
        os.system("git pull origin main")
        if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
            os.system('''git clone "https://huggingface.co/stabilityai/sd-vae-ft-mse"''')
            os.system('''mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae''')
            os.system("rm -r /content/stable-diffusion-v1-5/.git")
            os.chdir("/content/stable-diffusion-v1-5")
            os.system('''sed -i 's@"clip_sample": false@@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json''')
            os.system('''sed -i 's@"trained_betas": null,@"trained_betas": null@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json''')
            os.system('''sed -i 's@"sample_size": 256,@"sample_size": 512,@g' /content/stable-diffusion-v1-5/vae/config.json''')
            os.chdir("/content")

            self.line_logging('DONE : Model Download...')
        else:
            while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
                self.line_logging('Model Download : Something went wrong')
                
    # TextEnc, UNet Training
    def sd_custom_training(self, UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, MODEL_NAME, SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR, OUTPUT_DIR, Session_Name, PT):    
        self.line_logging("Start : Fine Tuning")
        import random
        import os
        import shutil

        MODELT_NAME = MODEL_NAME
        
        # UNet
        UNet_Training_Steps=UNet_Training_Steps 
        UNet_Learning_Rate = UNet_Learning_Rate
        untlr=UNet_Learning_Rate

        # Text_Encoder
        Enable_text_encoder_training= True
        Text_Encoder_Training_Steps=Text_Encoder_Training_Steps
        Text_Encoder_Learning_Rate = Text_Encoder_Learning_Rate #param ["2e-6", "1e-6","8e-7","6e-7","5e-7","4e-7"] {type:"raw"}
        stptxt=Text_Encoder_Training_Steps
        txlr=Text_Encoder_Learning_Rate

        # Seed
        Seed=""
        if Seed =='' or Seed=='0':
            Seed=random.randint(1, 999999)
        else:
            Seed=int(Seed)
            
        trnonltxt=""
        extrnlcptn=""
        Style=""
        Res = 512

        prec="fp16"
        precision=prec
        GC="--gradient_checkpointing"

        stp=0
        Start_saving_from_the_step=0
        stpsv=Start_saving_from_the_step


        dump_only_textenc = f"""accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
            {trnonltxt} \
            --image_captions_filename \
            --train_text_encoder \
            --dump_only_text_encoder \
            --pretrained_model_name_or_path="{MODELT_NAME}" \
            --instance_data_dir="{INSTANCE_DIR}" \
            --output_dir="{OUTPUT_DIR}" \
            --instance_prompt="{PT}" \
            --seed={Seed} \
            --resolution=512 \
            --mixed_precision={precision} \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 {GC} \
            --use_8bit_adam \
            --learning_rate={txlr} \
            --lr_scheduler="polynomial" \
            --lr_warmup_steps=0 \
            --max_train_steps={stptxt}
            """

        train_only_unet = f"""accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
            {Style} \
            {extrnlcptn} \
            --stop_text_encoder_training={stptxt} \
            --image_captions_filename \
            --train_only_unet \
            --save_starting_step={stpsv} \
            --save_n_steps={stp} \
            --Session_dir="{SESSION_DIR}" \
            --pretrained_model_name_or_path="{MODELT_NAME}" \
            --instance_data_dir="{INSTANCE_DIR}" \
            --output_dir="{OUTPUT_DIR}" \
            --captions_dir="{CAPTIONS_DIR}" \
            --instance_prompt={PT} \
            --seed={Seed} \
            --resolution={Res} \
            --mixed_precision={precision} \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 {GC} \
            --use_8bit_adam \
            --learning_rate={untlr} \
            --lr_scheduler="polynomial" \
            --lr_warmup_steps=0 \
            --max_train_steps={UNet_Training_Steps}
            """
        os.chdir('/content')
        # Text Encoder Training
        if Enable_text_encoder_training :
            self.line_logging('Training the text encoder...')
            if os.path.exists(OUTPUT_DIR+'/'+'text_encoder_trained'):
                shutil.rmtree(OUTPUT_DIR+'/'+'text_encoder_trained')
            os.system(dump_only_textenc)

        # UNet Training
        if UNet_Training_Steps!=0:
            self.line_logging('Training the UNet...')
            os.system(train_only_unet)

        # Copy feature_extractor, safety_checker, model_index.json 슈틸 3형제
        try:
            shutil.copytree("/content/stable-diffusion-v1-5/feature_extractor", OUTPUT_DIR + "/feature_extractor")
        except:
            print(f"File exists: '/content/models/{Session_Name}/feature_extractor'")
        try:    
            shutil.copytree("/content/stable-diffusion-v1-5/safety_checker", OUTPUT_DIR + "/safety_checker")
        except:
            print(f"File exists: '/content/models/{Session_Name}/safety_checker'")

        shutil.copyfile('/content/stable-diffusion-v1-5/model_index.json', OUTPUT_DIR + "/model_index.json")
        self.line_logging("Done : FineTuning...")
    # Total Function
    
    def sd_custom_function(self, UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name):
        import os
        WORKSPACE='/content/gdrive/MyDrive/Fast-Dreambooth'
        OUTPUT_DIR="/content/models/"+ Session_Name
        SESSION_DIR=WORKSPACE+'/Sessions/'+ Session_Name
        INSTANCE_DIR=SESSION_DIR+'/instance_images'
        CAPTIONS_DIR=SESSION_DIR+'/captions'
        MODEL_NAME="/content/stable-diffusion-v1-5"
        PT=""
        ### 1. Environment Setting
        try:
            import wget
        except:
            self.sd_custom_environment()

        ### 2. Create Session
        self.sd_custom_create_session(MODEL_NAME, SESSION_DIR, INSTANCE_DIR)

        ### 3. Image Upload
        self.sd_custom_upload_image(SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR)

        ### 4. Model Download (진행중)
        if not os.path.exists('/content/stable-diffusion-v1-5'):
            self.sd_custom_model_download()
        else:
            print("The v1.5 model already exists, using this model.") 

        ### 5. Training
        self.sd_custom_training(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, MODEL_NAME, SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR, OUTPUT_DIR, Session_Name, PT)


def text_to_image():
    
    print('Input the Huggingface Token: ')
    Huggingface_Token = getpass.getpass('')
    token=Huggingface_Token

    # Session Name
    print('Input the prompt: ')
    prompt = input('')

    print('Input the prompt: ')
    print('If you want Random Seed, input Nothing.')
    seed = input('')
    
    diffusion = Text_To_Image(token, prompt, seed)
    
    try:
        image = diffusion.sd_texttoimg_function(pipe_t2i, prompt, seed)
    except:
        pipe_t2i = diffusion.sd_texttoimg_pipeline(token)
        image = diffusion.sd_texttoimg_function(pipe_t2i, prompt, seed)
        
    return image


def image_to_image():
    
    print('Input the Huggingface Token: ')
    Huggingface_Token = getpass.getpass('')
    token=Huggingface_Token

    print('Input the file_name(or file_path) of image: ') 
    file_name = input('')

    print('Input the prompt: ')
    prompt = input('')
    
    print('Input the strength: ')
    print('Strength is recommended between 0.4 and 0.6.')
    strength = float(input(''))
    
    print('Input the seed: ')
    print('If you want Random Seed, input Nothing.')
    seed = input('')
    
    diffusion = Image_To_Image(token, file_name, prompt, strength, seed)
    
    try:
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
    except:
        pipe_i2i = diffusion.sd_imgtoimg_pipeline(token)
        image = diffusion.sd_imgtoimg_function(pipe_i2i, prompt, file_name, strength, seed)
        
    return image


def image_extend():
    print('Input the Huggingface Token: ')
    Huggingface_Token = getpass.getpass('')
    token=Huggingface_Token

    print('Input the file_name(or file_path) of image: ') 
    file_name = input('')

    print('Input the prompt: ')
    prompt = input('')
    
    print('Input the the x,y coordinates of the upper left vertex (ex. 325 410): ')
    num_list = list(map(int,input('').split( )))
    a, b = num_list[0], num_list[1]
    
    print('Input the Output Name: ')
    print("If you don't want save the result image, input Nothing.")
    output_name = input('')
    
    print('Input the prompt: ')
    print('If you want Random Seed, input Nothing.')
    seed = input('')
    
    diffusion = Image_Extend(token, file_name, prompt, a, b, output_name, seed)
    
    try:
        image = diffusion.sd_extend_function(pipe_ie, file_name, prompt, a, b, output_name, seed)
    except:
        pipe_ie = diffusion.sd_extend_pipeline(token)
        image = diffusion.sd_extend_function(pipe_ie, file_name, prompt, a, b, output_name, seed)
        
    return image


def fine_tuning():
    print('Input the Session_Name: ')
    Session_Name = input('')

    print('Input the UNet Training Steps: ')
    UNet_Training_Steps = input('')
    
    print('Input the UNet Learning Rate: ')
    print('Parameter : 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6')
    UNet_Learning_Rate = input('')
    
    print('Input the Text Encoder Training Steps: ')
    print("200-450 steps is enough for a small dataset.")
    print("keep this number small to avoid overfitting")
    Text_Encoder_Training_Steps = input('')

    print('Input the Text Encoder Learning Rate: ')
    print('Parameter : 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6')
    Text_Encoder_Learning_Rate = input('')

    diffusion = FineTuning(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name)
    
    diffusion.sd_custom_function(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, Session_Name)