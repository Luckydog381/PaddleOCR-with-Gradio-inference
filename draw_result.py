import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf",
             font_size=12):
    
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
        font_size: the size of font
    return(array):
        the visualized img
    """

    # Ensure image is a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert image to RGB if it's not already
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Load your custom font
    font = ImageFont.truetype(font_path, font_size)

    for i, box in enumerate(boxes):
        if scores[i] < drop_score or math.isnan(scores[i]):
            continue

        # Draw the bounding box
        box = np.array(box).astype(np.int32)
        image = cv2.polylines(image, [box], True, (255, 0, 0), 2)

        # Convert OpenCV image to PIL Image for drawing text
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Calculate the position for the text (above the bounding box)
        text_position = (int(box[0][0]), int(box[0][1]) - 10)  # Adjust Y-offset as needed

        # Draw the text on the image using Pillow
        draw.text(text_position, str(txts[i]), (0, 0, 0), font=font)

        # Convert PIL Image back to OpenCV image
        image = np.array(pil_image)

    return image

# Example usage:
# image = cv2.imread('path_to_your_image.jpg')  # Load your image
# visualize_boxes(image, boxes, txts, scores, drop_score, font_path)