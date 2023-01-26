# ▶1. Diffusion Custom Train (Fine Tuning)

기존 Stable Diffusion 모델에 원하는 이미지와 prompt를 학습시켜주는 모듈

basecode : [How to Use DreamBooth to Fine-Tune Stable Diffusion (Colab)](https://bytexd.com/how-to-use-dreambooth-to-fine-tune-stable-diffusion-colab/)

[basecode-colab](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb) 

## ◎ __finetuning_stable_diffusion_for_Colab.ipynb__

- __dowoloadmodel( )__ :

  - Stable Diffusion 모델을 다운로드 하는 함수, 1.5버전만 지원

  - 경로 : /content/stable-diffusion-v1-5

  - 모델 : [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

    

- __sd_custom_create_session( )__ :

  - input으로 넣어준 Session_Name에 맞춰 폴더를 생성해주는 함수

    

- __sd_custom_uploaded_image( )__ :

  - 이미지를 업로드하고, 이미지와 캡션을 분리해주는 함수

    

- __dump_only_textenc( )__ :

  - TextEncoder를 학습시켜주는 함수

    

- __train_only_unet ()__ :

  - UNet을 학습시켜주는 함수

    

- __sd_custom_function ()__ :

  - 종합

---

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

# ▶2. Text to Image



---

# ▶3. Image to Image

업로드한 이미지를 prompt에 맞는 이미지로 변환시켜주는 모듈

## ◎ __ImageToImage.py__

- __sd_imgtoimg_pipeline(pipe, token)__ : 
  - StableDiffusionImg2ImgPipeline이 선언이 안되어 있을 경우, pretrained 된 pipeline을 불러오는 함수

  - model : [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

    

- __sd_imgtoimg_function(prompt, pipe, file_name, seed)__ :
  - 원하는 이미지를 업로드하고, prompt에 맞게 이미지를 바꿔주는 함수

---

- __사용예시__

  ```python
  prompt = str(input("프롬프트를 입력해주세요 : "))
  img_to_img("1.5", prompt)
  ```

---

# ▶4. ImageExtend

## ◎ImageExtend.py

- __sd_extend_crop_mask(file_name, a, b)__ :

  - file_name에 있는 이미지를 불러온 뒤, (a, b) 좌표를 좌상단으로 하는 512*512 사이즈의 크롭이미지와 마스크를 만드는 함수

    

- __sd_extend_pipeline(pipe, token)__ : 

  - StableDiffusionInpaintPipeline이 선언 안되어 있을 경우, pretrained 된 pipeline을 불러오는 함수

  - model : [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)

    

- __sd_extend_result_img( )__ :

  - crop된 이미지와 마스크를 기반으로 inpaint를 실행한 뒤, 원본 이미지와 inpaint된 이미지를 합치는 함수

