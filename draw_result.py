from PIL import Image
from paddleocr import draw_ocr
import cv2
import numpy as np
import fitz
import os


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