"""
Authorizer.py
Cleaned + Optimized OCR Module
"""

import logging
import re

import numpy as np
from PIL import Image

from classes import MICR_Detection

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# BASE OCR CLASS
class BaseOCR:
    def __init__(
        self,
        custom_config: str,
        confidence_threshold: int = 35,
        grayscale: bool = False,
    ):

        self.detector = MICR_Detection()

        self.custom_config = custom_config

        self.confidence_threshold = confidence_threshold

        self.grayscale = grayscale

    # PREPROCESSING
    def preprocess(self, image):
        if image is None:
            raise ValueError("Image is None")

        image = np.array(image)

        if image.size == 0:
            raise ValueError("Empty image")

        # Auto rotation correction
        image = self.detector.Rotation(image)

        # Convert to PIL
        image = Image.fromarray(image)

        # Optional grayscale
        if self.grayscale:
            image = image.convert("L")

        return image

    # OCR PREDICTION
    def predict(self, image):
        processed_image = self.preprocess(image)

        _, words = self.detector.Get_Prediction(
            processed_image,
            psm=None,
            conf_score=self.confidence_threshold,
            custom_con=self.custom_config,
        )

        return " ".join(words)

    # CLEAN TEXT
    @staticmethod
    def clean_text(text, cleaning_regex):
        if not text:
            return ""

        cleaned = re.sub(cleaning_regex, " ", text)

        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned.strip()


# AMOUNT / DATE OCR
class SimpleOCR(BaseOCR):
    def __init__(self):

        custom_config = (
            r"--tessdata-dir ./Handwritten/DateAmount "
            r"-l Handwritten "
            r"--oem 1 "
            r"--psm 7"
        )

        super().__init__(
            custom_config=custom_config,
            confidence_threshold=35,
            grayscale=False,
        )

    def run_inference(
        self,
        image,
        text_cleaning_scheme=r"[^0-9a-zA-Z$.]+",
    ):
        # Run OCR for amount/date.
        try:

            text = self.predict(image)

            cleaned = self.clean_text(
                text,
                text_cleaning_scheme
            )

            return cleaned

        except Exception as e:

            logger.exception("SimpleOCR inference failed")

            return ""


# AUTHORIZER OCR
class C_Authorizer(BaseOCR):
    # OCR for cheque authorizer / maker name.
    def __init__(self):

        custom_config = (
            r'--oem 3 '
            r'--psm 6 '
            r'-l eng '
            r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata/"'
        )

        super().__init__(
            custom_config=custom_config,
            confidence_threshold=35,
            grayscale=True,
        )

    def run_inference(
        self,
        image,
        text_cleaning_scheme=r"[^0-9a-zA-Z$.,]+",
    ):
        # Run OCR for authorizer detection.
        try:

            text = self.predict(image)

            cleaned = self.clean_text(
                text,
                text_cleaning_scheme
            )
            return cleaned

        except Exception as e:

            logger.exception("C_Authorizer inference failed")

            return ""


# TESTING
if __name__ == "__main__":

    logger.info("Initializing OCR modules...")

    amount_ocr = SimpleOCR()

    authorizer_ocr = C_Authorizer()

    logger.info("OCR modules initialized successfully")