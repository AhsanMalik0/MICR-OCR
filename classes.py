"""
classes.py
Cleaned + Optimized MICR Detection Module
"""

import logging

import cv2
import numpy as np
from PIL import Image

from train.DLOCRMICRSelfTrain.src.configuration import Config
from train.DLOCRMICRSelfTrain.src.dataprocessing import DataProcessing
from train.DLOCRMICRSelfTrain.src.pipeline.micrpred import (
    predict_model,
    preprocess_array,
)

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# MICR DETECTION CLASS
class MICR_Detection:
    # Core MICR detection and preprocessing class.
    def __init__(self):

        self.image_processor = DataProcessing(
            config=Config()
        )

    # IMAGE VALIDATION
    @staticmethod
    def validate_image(image):
        """
        Validate image input.
        """

        if image is None:
            raise ValueError("Image is None")

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if image.size == 0:
            raise ValueError("Empty image")

        return image

    # ANGLE DETECTION
    @staticmethod
    def detect_angle(image):
        # Detect skew angle using Hough transform.
        image = MICR_Detection.validate_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            threshold1=50,
            threshold2=150,
            apertureSize=3,
        )
        lines = cv2.HoughLines(
            edges,
            1,
            np.pi / 180,
            100,
        )
        if lines is None:
            return 0.0
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        return float(np.median(angles))

    # ROTATION CORRECTION
    def Rotation(self, image):
        # Correct image rotation/skew.
        image = self.validate_image(image)

        angle = self.detect_angle(image)

        logger.info(f"Detected angle: {round(angle, 2)}")

        if abs(angle) < 1.0:
            return image

        height, width = image.shape[:2]

        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    # IMAGE PADDING
    @staticmethod
    def Padding(image, padding):
        # Add padding to image.
        image = MICR_Detection.validate_image(image)

        if image.ndim == 2:

            return np.pad(
                image,
                pad_width=padding,
                mode="reflect",
            )

        elif image.ndim == 3:

            return np.pad(
                image,
                pad_width=(
                    (padding, padding),
                    (padding, padding),
                    (0, 0),
                ),
                mode="edge",
            )

        else:
            raise ValueError(
                "Unsupported image dimensions"
            )

    # MODEL PREDICTION
    @staticmethod
    def Get_Prediction(
        image,
        psm=7,
        osm=3,
        conf_score=40,
        custom_con=None,
    ):
        # Run MICR OCR model prediction.
        try:
            image = np.array(image)
            processed = preprocess_array(image)
            prediction = predict_model(
                test_images=processed
            )
            return prediction
        except Exception as e:
            logger.exception(
                "Prediction failed"
            )
            raise RuntimeError(
                f"MICR prediction error: {str(e)}"
            )

    # CLEAN OCR TEXT
    @staticmethod
    def clean_text(text):
        # Clean MICR symbols.
        if not text:
            return ""

        text = "".join(text)

        cleaned = ''.join([
            '⑆' if char == 'T'
            else '⑈' if char == 'O'
            else char
            for char in text
        ])

        return cleaned

    # MAIN INFERENCE
    def run_inference(
        self,
        image,
        psm=[6, 11, 8],
        padding=[0, 2, 9, 3],
    ):
        # Main MICR inference pipeline.
        try:

            image = self.validate_image(image)

            padded_images = [
                self.Padding(image, p)
                for p in padding
            ]

            prediction = self.Get_Prediction(
                image=Image.fromarray(
                    padded_images[0]
                )
            )

            raw_text = " ".join(prediction)

            cleaned_text = self.clean_text(
                raw_text
            )

            return cleaned_text

        except Exception as e:

            logger.exception(
                "Inference pipeline failed"
            )

            return ""


# TESTING
if __name__ == "__main__":

    logger.info(
        "Initializing MICR Detection..."
    )
    detector = MICR_Detection()
    logger.info(
        "MICR Detection initialized successfully"
    )