from paddleocr import PaddleOCR, draw_ocr
from VisualizeOCR import OCRVisualization
from PIL import Image
import cv2
import numpy as np
import fitz
import time
import os

class OCR:
    def __init__(self, file_path, lang='japan', use_angle_cls=True, det_db_thresh=0.4, det_db_box_thresh=0.5,
                 det_db_unclip_ratio=1.5, max_batch_size=10, det_limit_side_len=960, det_db_score_mode="slow",
                 dilation=False, ocr_version="PP-OCRv4"):
        self.file_path = file_path
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.max_batch_size = max_batch_size
        self.det_limit_side_len = det_limit_side_len
        self.det_db_score_mode = det_db_score_mode
        self.dilation = dilation
        self.ocr_version = ocr_version
        # Initialize PaddleOCR with keyword arguments
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls, 
            lang=lang, 
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh, 
            det_db_unclip_ratio=det_db_unclip_ratio,
            max_batch_size=max_batch_size, 
            det_limit_side_len=det_limit_side_len,
            det_db_score_mode=det_db_score_mode, 
            version=ocr_version,
            dilation=dilation
        )
        self.visualizer = OCRVisualization()


    def perform_ocr(self):
        starting_time = time.time()
        result = self.ocr.ocr(self.file_path, cls=True)
        ending_time = time.time()
        return result, ending_time - starting_time

    def extract_text(self, result):
        text = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line[1][0])
                text.append(line[1][0])
        return ' '.join(text)

    def draw_and_save_results(self, result):
        result = result[0]
        image = Image.open(self.file_path).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = self.visualizer.draw_ocr(image, boxes, txts, scores, font_path='./japan.ttc')
        im_show = Image.fromarray(im_show)

        filename = os.path.basename(self.file_path)
        new_filename = "result_" + filename
        output_folder = './outputs/'
        output_path = os.path.join(output_folder, new_filename)
        os.makedirs(output_folder, exist_ok=True)
        im_show.save(output_path)
        return im_show
    
    def draw_and_save_results_pdf(self, result):

        """
        Draw and save the OCR results to a PDF file.
        Args:result: list of OCR results
             PAGE_NUM: number of pages to process
        Returns: list of PIL images 
        """
        
        # Get the file name and create output file name
        filename = os.path.basename(self.file_path)
        new_filename = "result_" + filename
        output_folder = './outputs/'
        output_path = os.path.join(output_folder, new_filename)
        
        # Get the pdf page numbers
        with fitz.open(self.file_path) as pdf:
            PAGE_NUM = pdf.page_count

        # Create a list to store the images
        imgs = []
        im_show_list = []

        # Open the PDF file
        with fitz.open(self.file_path) as pdf:
            for pg in range(0, PAGE_NUM):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
        
        # Draw and save the OCR results
        for idx in range(len(result)):
            res = result[idx]
            if res == None:
                continue
            image = imgs[idx]
            boxes = [line[0] for line in res]
            txts = [line[1][0] for line in res]
            scores = [line[1][1] for line in res]
            im_show = draw_ocr(image, boxes, txts, scores, font_path='./japan.ttc')
            im_show = Image.fromarray(im_show)
            im_show_list.append(im_show)
            im_show.save(output_path + '_page_{}.jpg'.format(idx))

        return im_show_list