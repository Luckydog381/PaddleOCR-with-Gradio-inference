from paddleocr import PaddleOCR, draw_ocr

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
PAGE_NUM = 10 # Set the recognition page number
pdf_path = 'default.pdf'
ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=PAGE_NUM)  # need to run only once to download and load model into memory
# ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=PAGE_NUM,use_gpu=0) # To Use GPU,uncomment this line and comment the above one.
result = ocr.ocr(pdf_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    if res == None: # Skip when empty result detected to avoid TypeError:NoneType
        print(f"[DEBUG] Empty page {idx+1} detected, skip it.")
        continue
    for line in res:
        print(line)
