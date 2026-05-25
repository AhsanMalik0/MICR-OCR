import platform
import pathlib
from ultralytics import YOLO
import numpy as np
import pytesseract
from pytesseract import Output
import cv2
from PIL import Image, ImageOps, ImageEnhance
import re
from cheque_args import arguments

class ChequeProcessor:
    def __init__(self):
        self.args = arguments().get_args()
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath
        self.model = YOLO(self.args.Model_Path)

    # ========== Utilities ==========
    def detect_angle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return 0.0
        angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0]]
        return np.median(angles)

    def rotate(self, image):
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        angle = self.detect_angle(image)
        if abs(angle) < 1.0:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def pad_image(self, image, padding):
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        if image.ndim == 2:
            return np.pad(image, pad_width=padding, mode='reflect')
        elif image.ndim == 3:
            return np.pad(image, pad_width=((padding, padding), (padding, padding), (0, 0)), mode='edge')
        else:
            raise ValueError("Unsupported image format")

    # ========== MICR Detection ==========
    def micr_prediction(self, image, psm=[6, 11, 8], padding=[0, 2, 9, 3]):
        padded_images = [self.pad_image(image, p) for p in padding]
        results = []

        for i in psm:
            for num, j in enumerate(padded_images):
                words_with_conf, words = self.tesseract_prediction(Image.fromarray(j), i)
                raw = " ".join(words)
                cleaned = self.clean_micr_text(raw)
                results.append({
                    "psm": i,
                    "padding": padding[num],
                    "raw_text": raw,
                    "cleaned_text": cleaned,
                    "words_with_conf": words_with_conf
                })
        return results

    def tesseract_prediction(self, image, psm, osm=None, conf_score=None, custom_con=None):
        image = image.convert('L')
        conf_score = conf_score or self.args.conf_score
        osm = osm or self.args.OSM
        accurate_words = []
        accurate_words_with_conf = []

        config = custom_con or f'--tessdata-dir ./train/train_tiff -l micr_ocr --oem {osm} --psm {psm}'
        data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)

        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            try:
                conf = int(data['conf'][i])
            except:
                continue
            if word != "" and conf >= conf_score:
                accurate_words.append(word)
                accurate_words_with_conf.append((word, conf))
        return accurate_words_with_conf, accurate_words

    def clean_micr_text(self, text):
        text = "".join(text)
        return ''.join(['⑆' if i == 'T' else "⑈" if i == 'O' else i for i in text])

    def extract_micr_fields(self, text):
        text = text.replace('\n', ' ').replace('D', '').strip()
        cleaned = re.sub(self.args.MICR_Char, '', text).replace(" ", '')
        rest = cleaned

        routing = re.search(self.args.Routing_Pattern, cleaned)
        routing_number = routing.group(0) if routing else None
        rest = rest.replace(routing_number, '') if routing else rest
        routing_number = re.sub(self.args.Clean_result, '', routing_number) if routing else None

        account = re.search(self.args.Account_Pattern, rest)
        account_number = account.group(1) if account else None
        rest = rest.replace(account.group(0), '') if account else rest
        account_number = re.sub(self.args.Clean_result, '', account_number) if account else None

        cheque = None
        for pattern in [self.args.Cheque_Pattern_I, self.args.Cheque_Pattern_II, self.args.Cheque_Pattern_III]:
            match = re.search(pattern, rest)
            if match:
                cheque = match.group(1)
                break

        return {
            "Cheuqe Number": cheque,
            "Account_Number": account_number,
            "Routing_number": routing_number
        }

    # ========== Simple OCR ==========
    def simple_ocr(self, image):
        image = Image.fromarray(image)
        image = ImageEnhance.Contrast(image).enhance(1.1)
        config = r'--oem 3 --psm 6 -l eng --tessdata-dir "/usr/share/tesseract-ocr/5/tessdata/"'
        _, words = self.tesseract_prediction(image, psm=None, custom_con=config, conf_score=35)
        text = " ".join(words)
        return re.sub(r'[^0-9a-zA-Z$.,]+', ' ', text)

    # ========== Main Prediction Pipeline ==========
    def predict(self, image):
        image = self.rotate(np.array(image))
        results = self.model(Image.fromarray(image))
        micr_result = None
        owner_result = None

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"Class: {cls}, Confidence: {conf}")

                arr = image[y1:y2, x1:x2]

                if cls == 0 and conf > 0.60:  # MICR
                    img = Image.fromarray(arr).convert('L')
                    width, height = img.size
                    colors = (
                        img.getpixel((0, height // 2)),
                        img.getpixel((width - 1, height // 2)),
                        img.getpixel((width // 2, 0)),
                        img.getpixel((width // 2, height - 1)),
                    )
                    img = ImageOps.expand(img, border=(4, 0, 0, 0), fill=colors[0])
                    img = ImageOps.expand(img, border=(0, 2, 0, 0), fill=colors[2])
                    img = ImageOps.expand(img, border=(0, 0, 4, 0), fill=colors[1])
                    img = ImageOps.expand(img, border=(0, 0, 0, 2), fill=colors[3])
                    micr_runs = self.micr_prediction(np.array(img), psm=self.args.PSM, padding=self.args.pading)
                    for r in micr_runs:
                        micr_fields = self.extract_micr_fields(r["cleaned_text"])
                        if all(micr_fields.values()):
                            micr_result = micr_fields
                            break

                elif cls == 3 and conf > 0.60:  # Owner
                    owner_result = self.simple_ocr(arr)

        return micr_result, owner_result
