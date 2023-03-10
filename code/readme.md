# ▶Text to Image

<br>

## ◎ __stable_diffusion_total.py__

- __sd_texttoimg_pipeline( )__
  - StableDiffusionPipeline이 선언이 안되어 있을 경우, pretrained 된 pipeline을 불러오는 함수
  
  - model : [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  
    <br>
  
- __sd_texttoimg_function( )__ : 
  - 작성한 prompt를 이미지로 바꿔주는 함수

    <br>
  
- __사용예시__

  ```python
  !pip install -r requirements.txt -q
  
  from stable_diffusion_total import text_to_image
  
  text_to_image()
  
  >>>> Input the Huggingface Token: 
  >>>> ··········
  >>>> Input the prompt: 
  >>>> SF Movie, Zootopia, space
  >>>> Input the prompt: 
  >>>> If you want Random Seed, input Nothing.
  >>>> 213
  ```

---

# ▶Image to Image

<br>

업로드한 이미지를 prompt에 맞는 이미지로 변환시켜주는 모듈

## ◎ __ImageToImage.py__

- __sd_imgtoimg_pipeline(token)__ : 
  - StableDiffusionImg2ImgPipeline이 선언이 안되어 있을 경우, pretrained 된 pipeline을 불러오는 함수

  - model : [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

    <br>

- __sd_imgtoimg_function(prompt, pipe, file_name, seed)__ :
  - file_name의 이미지를 불러 온 뒤, 그 이미지를 prompt에 맞게 바꿔주는 함수

    <br>
  
- __사용예시__

  ```python
  !pip install -r requirements.txt -q
  
  from stable_diffusion_total import image_to_image
  
  image_to_image()
  
  >>>> Input the Huggingface Token: 
  >>>> ··········
  >>>> Input the file_name(or file_path) of image: 
  >>>> /content/12344.jpg
  >>>> Input the prompt: 
  >>>> Harry Poter wearing a suit
  >>>> Input the strength: 
  >>>> Strength is recommended between 0.4 and 0.6.
  >>>> 0.5
  >>>> Input the seed: 
  >>>> If you want Random Seed, input Nothing.
  >>>>  
  ```

---

# ▶ImageExtend

<br>

## ◎ImageExtend.py

- __sd_extend_crop_mask(file_name, a, b)__ :

  - file_name에 있는 이미지를 불러온 뒤, (a, b) 좌표를 좌상단으로 하는 512*512 사이즈의 크롭이미지와 마스크를 만드는 함수

    <br>

- __sd_extend_pipeline(pipe, token)__ : 

  - StableDiffusionInpaintPipeline이 선언 안되어 있을 경우, pretrained 된 pipeline을 불러오는 함수

  - model : [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)

    <br>

- __sd_extend_result_img( )__ :

  - crop된 이미지와 마스크를 기반으로 inpaint를 실행한 뒤, 원본 이미지와 inpaint된 이미지를 합치는 함수
  
    <br>

- __sd_extend_function( )__ :

  - file_name의 이미지를 불러온 뒤, (a, b)좌표를 기준으로 image Extend 후 output_name으로 결과물을 저장하는 함수

    <br>

- __사용예시__

  ```python
  !pip install -r requirements.txt -q
  
  from stable_diffusion_total import image_extend
  
  image_extend()
  
  >>>> Input the Huggingface Token: 
  >>>> ··········
  >>>> Input the file_name(or file_path) of image: 
  >>>> /content/9984444A5BB495A00E.jfif
  >>>> Input the prompt: 
  >>>> Star Wars, meteor strike
  >>>> Input the the x,y coordinates of the upper left vertex (ex. 325 410): 
  >>>> 250 250
  >>>> Input the Output Name: 
  >>>> If you don't want save the result image, input Nothing.
  >>>> 
  >>>> Input the prompt: 
  >>>> If you want Random Seed, input Nothing.
  >>>> 
  ```

# ▶Diffusion Custom Train (Fine Tuning)

<br>

기존 Stable Diffusion 모델에 원하는 이미지와 prompt를 학습시켜주는 모듈

> basecode : [How to Use DreamBooth to Fine-Tune Stable Diffusion (Colab)](https://bytexd.com/how-to-use-dreambooth-to-fine-tune-stable-diffusion-colab/)
>
> [basecode-colab](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb) 

## ◎ __finetuning_stable_diffusion_for_Colab.ipynb__

- __dowoloadmodel( )__ :

  - Stable Diffusion 모델을 다운로드 하는 함수, 현재 1.5버전만 지원

  - 경로 : /content/stable-diffusion-v1-5

  - 모델 : [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

    <br>

- __sd_custom_create_session( )__ :

  - input으로 넣어준 Session_Name에 맞춰 폴더를 생성해주는 함수

    <br>

- __sd_custom_uploaded_image( )__ :

  - 이미지를 업로드하고, 이미지와 캡션을 분리해주는 함수

    <br>

- __dump_only_textenc( )__ :

  - TextEncoder를 학습시켜주는 함수

    <br>

- __train_only_unet ( )__ :

  - UNet을 학습시켜주는 함수

    <br>

- __sd_custom_function ( )__ :

  - Fine Tuning을 해주는 함수

  - token과 Session_Name을 입력해줘야 한다.

  - 업로드하는 사진의 크기는  512*512 크기여야 하고, 이름은 prompt (1), prompt (2), prompt (3) ... 이런식으로 통일되어야 한다.

    <br>

- __사용예시__

  ```python
  print('Input the Huggingface Token: ')
  Huggingface_Token = input('')
  token=Huggingface_Token
  
  # Session Name
  print('Input the Session Name:') 
  Session_Name=input('')
  
  sd_custom_function(token, Session_Name, 800, 1e-5, 350, 1e-6)
  ```

---

# 
