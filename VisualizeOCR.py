import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string

class OCRVisualization:
    def __init__(self, input_size=600):
        self.input_size = input_size

    def resize_img(self, img):
        img = np.array(img)
        im_shape = img.shape
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.input_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
        return img

    def draw_ocr(self, image, boxes, txts=None, scores=None, drop_score=0.5, font_path="./doc/fonts/simfang.ttf"):
        if scores is None:
            scores = [1] * len(boxes)
        box_num = len(boxes)
        for i in range(box_num):
            if scores is not None and (scores[i] < drop_score or math.isnan(scores[i])):
                continue
            box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
            image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
            image = cv2.putText(image, str(i+1), (box[0][0][0], box[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        if txts is not None:
            img = np.array(self.resize_img(image))
            txt_img = self.text_visual(txts, scores, img_h=img.shape[0], img_w=600, threshold=drop_score, font_path=font_path)
            img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
            return img
        return image

    def text_visual(self, texts, scores, img_h=400, img_w=600, threshold=0., font_path="./doc/fonts/simfang.ttf"):
        if scores is not None:
            assert len(texts) == len(scores), "The number of texts and corresponding scores must match"
        
        blank_img, draw_txt = self.create_blank_img(img_h, img_w)

        font_size = 20
        txt_color = (0, 0, 0)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

        gap = font_size + 5
        txt_img_list = []
        count, index = 1, 0
        for idx, txt in enumerate(texts):
            index += 1
            if scores[idx] < threshold or math.isnan(scores[idx]):
                index -= 1
                continue
            first_line = True
            while self.str_count(txt) >= img_w // font_size - 4:
                tmp = txt
                txt = tmp[:img_w // font_size - 4]
                if first_line:
                    new_txt = str(index) + ': ' + txt
                    first_line = False
                else:
                    new_txt = '    ' + txt
                draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
                txt = tmp[img_w // font_size - 4:]
                if count >= img_h // gap - 1:
                    txt_img_list.append(np.array(blank_img))
                    blank_img, draw_txt = self.create_blank_img(img_h, img_w)
                    count = 0
                count += 1
            if first_line:
                new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
            else:
                new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            if count >= img_h // gap - 1 and idx + 1 < len(texts):
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = self.create_blank_img(img_h, img_w)
                count = 0
            count += 1
        txt_img_list.append(np.array(blank_img))
        if len(txt_img_list) == 1:
            blank_img = np.array(txt_img_list[0])
        else:
            blank_img = np.concatenate(txt_img_list, axis=1)
        return np.array(blank_img)

    def create_blank_img(self, img_h, img_w):
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    @staticmethod
    def str_count(s):
        count_zh = count_pu = 0
        s_len = len(s)
        en_dg_count = 0
        for c in s:
            if c in string.ascii_letters or c.isdigit() or c.isspace():
                en_dg_count += 1
            elif c.isalpha():
                count_zh += 1
            else:
                count_pu += 1
        return s_len - math.ceil(en_dg_count / 2)
