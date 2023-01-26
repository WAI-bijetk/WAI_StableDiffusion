# ▶1. Diffusion Custom Train (Fine Tuning)



# ▶2. Text to Image



# ▶3. Image to Image

업로드한 이미지를 prompt에 맞는 이미지로 변환시켜주는 모듈

## ◎ __ImageToImage.py__

- __sd_imgtoimg_pipeline(pipe, token)__ : 
  - StableDiffusionImg2ImgPipeline이 선언이 안되어 있을 경우, pretrained 된 pipeline을 불러오는 함수
  - model : [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

- __sd_imgtoimg_function(prompt, pipe, file_name, seed)__ :
  - 원하는 이미지를 업로드하고, prompt에 맞게 이미지를 바꿔주는 함수



# ▶4. ImageExtend

## ◎ImageExtend.py

- __sd_extend_crop_mask(file_name, a, b)__ :

  - file_name에 있는 이미지를 불러온 뒤, (a, b) 좌표를 좌상단으로 하는 512*512 사이즈의 크롭이미지와 마스크를 만드는 함수

    

- __sd_extend_pipeline(pipe, token)__ : 

  - StableDiffusionInpaintPipeline이 선언 안되어 있을 경우, pretrained 된 pipeline을 불러오는 함수

  - model : [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)

    

- __sd_extend_result_img()__ :

  - crop된 이미지와 마스크를 기반으로 inpaint를 실행한 뒤, 원본 이미지와 inpaint된 이미지를 합치는 함수

