from IPython.utils import capture
from subprocess import getoutput
import time
import os
from google.colab import files


print('Installing dependencies...')

with capture.capture_output() as cap:
    %cd /content/
    !pip install -q --no-deps accelerate==0.12.0
    !wget -q -i "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/dbdeps.txt"
    for i in range(1,8):
        !mv "deps.{i}" "deps.7z.00{i}"
    !7z x -y -o/ deps.7z.001
    !rm *.00* *.txt
    !git clone --depth 1 --branch updt https://github.com/TheLastBen/diffusers
    s = getoutput('nvidia-smi')
    if "A100" in s:
        !wget -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/A100
        !rm -r /usr/local/lib/python3.8/dist-packages/xformers
        !7z x -y -o/usr/local/lib/python3.8/dist-packages/ /content/A100
        !rm /content/A100
print('Done, proceed')

from IPython.display import clear_output
import wget
import shutil
from os import listdir
from os.path import isfile
from PIL import Image
from tqdm import tqdm
import ipywidgets as widgets
from io import BytesIO
import random
import shutil

def downloadmodel(token):
    token=token
    
    if os.path.exists('/content/stable-diffusion-v1-5'):
        !rm -r /content/stable-diffusion-v1-5
    clear_output()

    %cd /content/
    clear_output()
    !mkdir /content/stable-diffusion-v1-5
    %cd /content/stable-diffusion-v1-5
    !git init
    !git lfs install --system --skip-repo
    !git remote add -f origin  "https://USER:{token}@huggingface.co/runwayml/stable-diffusion-v1-5"
    !git config core.sparsecheckout true
    !echo -e "scheduler\ntext_encoder\ntokenizer\nunet\nfeature_extractor\nsafety_checker\nmodel_index.json\n!*.safetensors" > .git/info/sparse-checkout
    !git pull origin main
    if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
        !git clone "https://USER:{token}@huggingface.co/stabilityai/sd-vae-ft-mse"
        !mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae
        !rm -r /content/stable-diffusion-v1-5/.git
        %cd /content/stable-diffusion-v1-5
        !sed -i 's@"clip_sample": false@@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json
        !sed -i 's@"trained_betas": null,@"trained_betas": null@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json
        !sed -i 's@"sample_size": 256,@"sample_size": 512,@g' /content/stable-diffusion-v1-5/vae/config.json  
        %cd /content/    
        clear_output()
        print('DONE !')
    else:
        while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
            print('Make sure you accepted the terms in https://huggingface.co/runwayml/stable-diffusion-v1-5')
            time.sleep(5)

def sd_custom_create_session(MODEL_NAME, SESSION_DIR, MDLPTH, OUTPUT_DIR):
    if os.path.exists(str(SESSION_DIR)) and not os.path.exists(MDLPTH):
        print('Loading session with no previous model, using the original model or the custom downloaded model')
        if MODEL_NAME=="":
            print('No model found, use the "Model Download" cell to download a model.')
        else:
            print('Session Loaded, proceed to uploading instance images')

    elif os.path.exists(MDLPTH):
        print('Session found, loading the trained model ...')
        !wget -q -O refmdlz https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/refmdlz
        !unzip -o -q refmdlz
        !rm -f refmdlz
        !wget -q -O convertodiff.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/convertodiffv1.py
        clear_output()
        print('Session found, loading the trained model ...')
        !python /content/convertodiff.py "$MDLPTH" "$OUTPUT_DIR" --v1
        !rm -r /content/refmdl
        !rm /content/convertodiff.py  
        if os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
            resume=True    
            clear_output()
            print('Session loaded.')
        else:     
            if not os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
                print('Conversion error, if the error persists, remove the CKPT file from the current session folder')

    elif not os.path.exists(str(SESSION_DIR)):
        %mkdir -p "$INSTANCE_DIR"
        print('Creating session...')
        if MODEL_NAME=="":
            print('No model found, use the "Model Download" cell to download a model.')
        else:
            print('Session created, proceed to uploading instance images')


def sd_custom_uploaded_image(CAPTIONS_DIR, INSTANCE_DIR):

    if os.path.exists(CAPTIONS_DIR+"off"):
        !mv $CAPTIONS_DIR"off" $CAPTIONS_DIR
        time.sleep(3)

    if os.path.exists(str(INSTANCE_DIR)):
        !rm -r "$INSTANCE_DIR"
    if os.path.exists(str(CAPTIONS_DIR)):
        !rm -r "$CAPTIONS_DIR"

    if not os.path.exists(str(INSTANCE_DIR)):
        %mkdir -p "$INSTANCE_DIR"
    if not os.path.exists(str(CAPTIONS_DIR)):
        %mkdir -p "$CAPTIONS_DIR"

    if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
        %rm -r $INSTANCE_DIR"/.ipynb_checkpoints"


    for filename in uploaded.keys():
        if filename.split(".")[-1]=="txt":
            shutil.move(filename, CAPTIONS_DIR)
        up=[filename for filename in uploaded.keys() if filename.split(".")[-1]!="txt"]

    for filename in tqdm(uploaded.keys(), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
        shutil.move(filename, INSTANCE_DIR)
        clear_output()
    print('upload image Done, proceed to the next cell')

    with capture.capture_output() as cap:
        %cd "$INSTANCE_DIR"
        !find . -name "* *" -type f | rename 's/ /-/g'
        %cd "$CAPTIONS_DIR"
        !find . -name "* *" -type f | rename 's/ /-/g'
        
        %cd $SESSION_DIR
        !rm instance_images.zip captions.zip
        !zip -r instance_images instance_images
        !zip -r captions captions
        %cd /content


def dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps):
    
    !accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
    $trnonltxt \
    --image_captions_filename \
    --train_text_encoder \
    --dump_only_text_encoder \
    --pretrained_model_name_or_path="$MODELT_NAME" \
    --instance_data_dir="$INSTANCE_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --instance_prompt="$PT" \
    --seed=$Seed \
    --resolution=512 \
    --mixed_precision=$precision \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 $GC \
    --use_8bit_adam \
    --learning_rate=$txlr \
    --lr_scheduler="polynomial" \
    --lr_warmup_steps=0 \
    --max_train_steps=$Training_Steps

def train_only_unet(stptxt, stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps):
    clear_output()
    print('Training the UNet...')
    !accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
    $Style \
    $extrnlcptn \
    --stop_text_encoder_training=$stptxt \
    --image_captions_filename \
    --train_only_unet \
    --save_starting_step=$stpsv \
    --save_n_steps=$stp \
    --Session_dir=$SESSION_DIR \
    --pretrained_model_name_or_path="$MODELT_NAME" \
    --instance_data_dir="$INSTANCE_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --captions_dir="$CAPTIONS_DIR" \
    --instance_prompt="$PT" \
    --seed=$Seed \
    --resolution=$Res \
    --mixed_precision=$precision \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 $GC \
    --use_8bit_adam \
    --learning_rate=$untlr \
    --lr_scheduler="polynomial" \
    --lr_warmup_steps=0 \
    --max_train_steps=$Training_Steps
    
    
def sd_custom_function(token, Session_Name, UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate):
    import os
    from subprocess import getoutput
    from IPython.display import clear_output
    from google.colab import runtime
    import time
    import random
    # 디렉토리 선언
    WORKSPACE='/content/gdrive/MyDrive/Fast-Dreambooth'
    INSTANCE_NAME=Session_Name
    OUTPUT_DIR="/content/models/"+ Session_Name
    SESSION_DIR=WORKSPACE+'/Sessions/'+ Session_Name
    INSTANCE_DIR=SESSION_DIR+'/instance_images'
    CONCEPT_DIR=SESSION_DIR+'/concept_images'
    CAPTIONS_DIR=SESSION_DIR+'/captions'
    MDLPTH=str(SESSION_DIR+"/"+ Session_Name+'.ckpt')
    PT=""

    # Model Download
    if not os.path.exists('/content/stable-diffusion-v1-5'):
        downloadmodel(token)
        MODEL_NAME="/content/stable-diffusion-v1-5"
    else:
        MODEL_NAME="/content/stable-diffusion-v1-5"
        print("The v1.5 model already exists, using this model.")  
        
    # create/load session
    with capture.capture_output() as cap:
        %cd /content
    sd_custom_create_session(MODEL_NAME, SESSION_DIR, MDLPTH, OUTPUT_DIR)

    # Instant_image
    sd_custom_prepare_image_upload(CAPTIONS_DIR, INSTANCE_DIR)

    up=""  
    uploaded = files.upload()

    sd_custom_uploaded_image(CAPTIONS_DIR, INSTANCE_DIR)
    with capture.capture_output() as cap:
        %cd "$INSTANCE_DIR"
        !find . -name "* *" -type f | rename 's/ /-/g'
        %cd "$CAPTIONS_DIR"
        !find . -name "* *" -type f | rename 's/ /-/g'
        
        %cd $SESSION_DIR
        !rm instance_images.zip captions.zip
        !zip -r instance_images instance_images
        !zip -r captions captions
        %cd /content
        
    # training
    

    Resume_Training = False

    MODELT_NAME=MODEL_NAME
    print(MODELT_NAME)
    print(OUTPUT_DIR)
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

    Enable_Text_Encoder_Concept_Training= False
    Text_Encoder_Concept_Training_Steps=0

    # Seed
    Seed=""
    if Seed =='' or Seed=='0':
        Seed=random.randint(1, 999999)
    else:
        Seed=int(Seed)
    print(type(Seed))
    trnonltxt=""

    extrnlcptn=""
    Style=""

    Resolution = "512" #param ["512", "576", "640", "704", "768", "832", "896", "960", "1024"]
    Res=int(Resolution)

    fp16 = True
    prec="fp16"
    precision=prec

    GC="--gradient_checkpointing"
    s = getoutput('nvidia-smi')
    if 'A100' in s:
        GC=""
        
    External_Captions = False

    resuming=""

    stp=0
    Start_saving_from_the_step=0
    stpsv=Start_saving_from_the_step

    Disconnect_after_training=False


    # 텍스트 인코더 트레이닝
    if Enable_text_encoder_training :
        print('Training the text encoder...')
        if os.path.exists(OUTPUT_DIR+'/'+'text_encoder_trained'):
            %rm -r $OUTPUT_DIR"/text_encoder_trained"
        dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxt)

    # 유넷 트레이닝
    if UNet_Training_Steps!=0:
        train_only_unet(stptxt, stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps=UNet_Training_Steps)

    # feature_extractor, safety_checker, model_index.json 커스텀 모델 파일에 추가
    try:
        shutil.copytree("/content/stable-diffusion-v1-5/feature_extractor", OUTPUT_DIR + "/feature_extractor")
    except:
        print(f"File exists: '/content/models/{Session_Name}/feature_extractor'")
    try:    
        shutil.copytree("/content/stable-diffusion-v1-5/safety_checker", OUTPUT_DIR + "/safety_checker")
    except:
        print(f"File exists: '/content/models/{Session_Name}/safety_checker'")

    shutil.copyfile('/content/stable-diffusion-v1-5/model_index.json', OUTPUT_DIR + "/model_index.json")
