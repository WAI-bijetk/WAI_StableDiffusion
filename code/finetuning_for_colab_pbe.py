def sd_custom_environment():
    import os
    import subprocess
    import shutil
    import glob
    from distutils.dir_util import copy_tree
    import time
    print("Start Env. Setting")
    os.chdir('/content/')
    subprocess.run(['pip', 'install', '-q', '--no-deps', 'accelerate==0.12.0'])
    # !pip install -q --no-deps accelerate==0.12.0
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
    print('Done, proceed')
    

def sd_custom_create_session(MODEL_NAME, SESSION_DIR, MDLPTH):
    import os
    if os.path.exists(str(SESSION_DIR)) and not os.path.exists(MDLPTH):
        print('Loading session with no previous model, using the original model or the custom downloaded model')
        if MODEL_NAME=="":
            print('No model found, use the "Model Download" cell to download a model.')
        else:
            print('Session Loaded, proceed to uploading instance images')


    elif not os.path.exists(str(SESSION_DIR)):
        # %mkdir -p "$INSTANCE_DIR"
        os.makedirs(INSTANCE_DIR)
        print('Creating session...')
        if MODEL_NAME=="":
            print('No model found, use the "Model Download" cell to download a model.')
        else:
            print('Session created, proceed to uploading instance images')
            

def sd_custom_upload_image_replace(directory):
    import shutil
    # 이미지, 캡션에 들어간 띄어쓰기를 "-" 로 바꿔주는 함수
    import glob

    inst_list = glob.glob(directory+"/*")
    for i in inst_list:
        old_name = i.split("/")[-1]
        new_name = old_name.replace(" ", "-")
        shutil.move(directory +"/"+old_name, directory + "/" + new_name)
        
        
def sd_custom_upload_image(SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR):
    import shutil
    import os
    from tqdm import tqdm
    from google.colab import files
    from IPython.display import clear_output
    
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
        sd_custom_upload_image_replace(directory)  
    
    # 파일 압축  
    os.chdir(SESSION_DIR)
    if os.path.exists("instance_images.zip"):
        os.remove("instance_images.zip")
        
    if os.path.exists("captions.zip"):
        os.remove("captions.zip")
    
    shutil.make_archive('instance_images', 'zip', './instance_images')
    shutil.make_archive('captions', 'zip', './captions')
    
    
### Training Text Encoder
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

### Training UNet
def train_only_unet(stptxt, stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps):
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
    

def sd_custom_training(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, MODEL_NAME, INSTANCE_DIR, OUTPUT_DIR, Session_Name):
    import random
    import os
    import shutil
    os.chdir("/content")
    Resume_Training = False

    MODELT_NAME=MODEL_NAME
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
        
    trnonltxt=""
    extrnlcptn=""
    Style=""
    Resolution = 512

    fp16 = True
    prec="fp16"
    precision=prec
    GC="--gradient_checkpointing"
        
    External_Captions = False
    resuming=""

    stp=0
    Start_saving_from_the_step=0
    stpsv=Start_saving_from_the_step
    Disconnect_after_training=False


    # Text Encoder Training
    if Enable_text_encoder_training :
        print('Training the text encoder...')
        if os.path.exists(OUTPUT_DIR+'/'+'text_encoder_trained'):
            %rm -r $OUTPUT_DIR"/text_encoder_trained"
        os.chdir("/content")
        dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, stptxt)

    # UNet Training
    if UNet_Training_Steps!=0:
        print('Training the UNet...')
        train_only_unet(stptxt, stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, UNet_Training_Steps)

    # Copy feature_extractor, safety_checker, model_index.json
    try:
        shutil.copytree("/content/stable-diffusion-v1-5/feature_extractor", OUTPUT_DIR + "/feature_extractor")
    except:
        print(f"File exists: '/content/models/{Session_Name}/feature_extractor'")
    try:    
        shutil.copytree("/content/stable-diffusion-v1-5/safety_checker", OUTPUT_DIR + "/safety_checker")
    except:
        print(f"File exists: '/content/models/{Session_Name}/safety_checker'")

    shutil.copyfile('/content/stable-diffusion-v1-5/model_index.json', OUTPUT_DIR + "/model_index.json")


def sd_custom_function(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, MODEL_NAME, INSTANCE_DIR, OUTPUT_DIR, Session_Name, MDLPH, CAPTIONS_DIR):
    import os
    import shutil
    ### 1. Environment Setting
    sd_custom_environment()

    ### 2. Create Session
    sd_custom_create_session(MODEL_NAME, SESSION_DIR, MDLPTH)

    ### 3. Image Upload
    sd_custom_upload_image(SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR)

    ### 4. Model Download (진행중)
    if not os.path.exists('/content/stable-diffusion-v1-5'):
        shutil.copytree("/content/gdrive/MyDrive/stable-diffusion-v1-5", "/content/stable-diffusion-v1-5")
    else:
        print("The v1.5 model already exists, using this model.") 

    ### 5. Training
    sd_custom_training(UNet_Training_Steps, UNet_Learning_Rate, Text_Encoder_Training_Steps, Text_Encoder_Learning_Rate, MODEL_NAME, INSTANCE_DIR, OUTPUT_DIR, Session_Name)

