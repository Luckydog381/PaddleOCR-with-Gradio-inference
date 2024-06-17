import gradio as gr
from PIL import Image,ImageDraw
import os
import json
from OCR_Process import OCR  # Giả định bạn đã có sẵn class này
from dotenv import load_dotenv

from dotenv import load_dotenv

load_dotenv(".venv")

def test_image_display():
    # Create a simple image to test Gradio's display capability
    image = Image.new('RGB', (100, 100), (255, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Test", fill=(255, 255, 255))
    
    dummy_data = {"dummy": "data"}  # Placeholder for the JSON output
    return image, dummy_data

def process_image(pdf_file, url, sheet_name, lang,  use_angle_cls, det_db_thresh,
                  det_db_box_thresh, det_db_unclip_ratio, max_batch_size, det_limit_side_len, 
                  det_db_score_mode, dilation, ocr_version):
    """Xử lý hình ảnh và trích xuất dữ liệu."""
    # Chuyển đổi Gradio Image sang PIL Image nếu cần
    
    det_limit_side_len = int(det_limit_side_len)
    
    if not det_db_score_mode in ["slow", "fast"]:
        det_db_score_mode = "slow"

    if pdf_file is None:
        raise ValueError("No image provided. Please upload an image.")

    # Giả định bạn đã có sẵn hàm perform_ocr để xử lý OCR
    ocr = OCR(pdf_file, lang=lang, use_angle_cls=use_angle_cls, det_db_thresh=det_db_thresh, 
          det_db_box_thresh=det_db_box_thresh, det_db_unclip_ratio=det_db_unclip_ratio, 
          max_batch_size=max_batch_size, det_limit_side_len=det_limit_side_len, 
          det_db_score_mode=det_db_score_mode, dilation=dilation, ocr_version=ocr_version)
    result, time_taken = ocr.perform_ocr()
    extracted_text = ocr.extract_text(result)
    processed_image = ocr.draw_and_save_results_pdf(result)  # Hàm này giờ trả về list PIL images
    
    image_obj = Image.new('RGB', (100, 100))
    if isinstance(processed_image, Image.Image):
      print("Đối tượng là một instance của PIL Image.")
    else:
      print("Đối tượng không phải là một instance của PIL Image.")

    return processed_image, extracted_text

# Tạo Gradio interface
iface = gr.Interface(
    process_image,
   [
    gr.File(label="Upload Image or Take a Picture"),
    gr.Text(label="Google Sheet URL"),
    gr.Text(label="Sheet Name"),
    gr.Dropdown(choices=['en', 'japan', 'ch'], label="Language", value='en'),
    gr.Checkbox(label="Use Angle Classification", value=True),
    gr.Slider(minimum=0.1, maximum=0.9, step=0.1, value=0.4, label="Detection DB Threshold"),
    gr.Slider(minimum=0.1, maximum=0.9, step=0.1, value=0.5, label="Detection DB Box Threshold"),
    gr.Slider(minimum=1.0, maximum=2.0, step=0.1, value=1.4, label="Detection DB Unclip Ratio"),
    gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Max Batch Size"),
    gr.Slider(minimum=100, maximum=1024, step=10, value=340, label="Det Limit Side Len"),
    gr.Dropdown(choices=["slow", "fast"], label="Detection DB Score Mode", value="slow"),
    gr.Checkbox(label="Use Dilation", value=False),
    gr.Dropdown(choices=['PP-OCRv3', 'PP-OCRv4'], label="OCR Version", value='PP-OCRv4'),
    ],
    outputs=[gr.Gallery(type="pil", label="Processed Image"), gr.Text(label="Extracted Data")],
    title="OCR and Data Extraction Demo",
    description="Upload an image or take a picture to extract data and update it to Google Sheets."
)

# Chạy ứng dụng 
iface.launch(share=True)