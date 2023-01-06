Stable Diffusion - Image Extend.py

- 실행방법
: Python 환경에서 
1. pip install -r requirements.txt 

2. from ImageExtend import sd_extend_pipeline,sd_extend_function

3. try~except 선언을 통해 pipe 변수 선언 여부를 판단하고, 변수가 선언되지 않았을때만 pipe 파일을 다운로드.
   필요 변수 : pipe, token (Hugging Face 토큰)
    try:
        pipe
    except:
        pipe = None
    pipe = sd_extend_pipeline(pipe, token)

4. sd_extend_function(pipe, file_name, prompt, negative_prompt, a, b, output_name,) 실행
- 필요변수 
   pipe : sd_extend_pipeline을 통해 다운로드 받은 pipeline
   file_name : 원본 이미지 파일 이름 (경로)
   promprt : 사용할 prompt
   negative_prompt = 사용할 negative prompt
   a, b : 새로운 이미지를 생성할 공간의 좌상단 x, y 좌표
   output_name = 이미지를 저장할 경우, 이름 (경로)



- 실행 전 필독사항

sd_extend_pipeline(pipe, token) : pipe = None 일 때, pretarined pipeline을 가져오는 함수

sd_extend_crop_mask(file_name, a, b) : file_name의 원본 이미지를 불러온 뒤, a,b 좌표를 좌상단 꼭지점 좌표로 하는 512*512 크기의 사각형으로 이미지를 크롭하는 함수

sd_extend_result_img(pipe, prompt, negative_prompt, guidance_scale, extend_img, image, mask_image, a, b, seed) : 

sd_extend_function : 