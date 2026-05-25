import logging
import os
import platform
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from Authorizer import SimpleOCR, C_Authorizer
from Date import fix_date
from classes import MICR_Detection
from train.DLOCRMICRSelfTrain.src.pipeline.micrpred import (
    predict_model,
    preprocess_array,
)

# PLATFORM FIX
if platform.system() == "Windows":
    Path.PosixPath = Path.WindowsPath
else:
    Path.WindowsPath = Path.PosixPath

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = "./weights/best.pt"

CONFIDENCE_THRESHOLD = 0.60

PREDICTION_DIR = "./Prediction"
BOUNDING_BOX_DIR = f"{PREDICTION_DIR}/BBX_CHEQUE"
CHEQUE_DIR = f"{PREDICTION_DIR}/CHEQUE"

os.makedirs(BOUNDING_BOX_DIR, exist_ok=True)
os.makedirs(CHEQUE_DIR, exist_ok=True)

# Detection Classes
CLASS_MICR = 0
CLASS_DATE = 1
CLASS_AMOUNT = 2
CLASS_MAKER = 3

# =========================================================
# OCR INITIALIZATION
# =========================================================

ocr_amount = SimpleOCR()
authorizer_ocr = C_Authorizer()
micr_detector = MICR_Detection()

# =========================================================
# UTILITIES
# =========================================================


def extract_field(text: str) -> dict:
    # Extract MICR fields from OCR text.
    if not text:
        return {}

    text = text.replace("\n", " ").replace("D", "").strip()

    cleaned = re.sub(r"[^0-9⑈⑆ ]+", "", text)
    cleaned = cleaned.replace(" ", "")

    result = {
        "Cheque_Number": None,
        "Account_Number": None,
        "Routing_Number": None,
    }
    # ROUTING NUMBER
    routing_match = re.search(r"⑆(\d{9})⑆", cleaned)

    if routing_match:
        result["Routing_Number"] = routing_match.group(1)
    # ACCOUNT NUMBER
    account_match = re.search(r"((?=\d{8,})(?!0{4})\d{8,})⑈", cleaned)

    if account_match:
        result["Account_Number"] = account_match.group(1)

    # CHEQUE NUMBER
    cheque_match = re.search(r"(?:0{4,})?(\d{4,6})⑈?", cleaned)

    if cheque_match:
        result["Cheque_Number"] = cheque_match.group(1)

    return result


def save_failed_prediction(image, folder_name):
    # Save failed OCR crops for debugging.
    save_dir = f"{PREDICTION_DIR}/{folder_name}"

    os.makedirs(save_dir, exist_ok=True)

    image_count = len(os.listdir(save_dir)) + 1

    save_path = f"{save_dir}/failed_{image_count}.jpg"

    cv2.imwrite(save_path, image)

    logger.warning(f"Saved failed prediction: {save_path}")


def validate_image(image):
    # Validate input image.
    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be numpy array")

    if image.size == 0:
        raise ValueError("Empty image")

    if len(image.shape) != 3:
        raise ValueError("Invalid image dimensions")


# MODEL LOADING
def load_model():
    try:
        logger.info("Loading YOLO model...")

        model = YOLO(MODEL_PATH)

        logger.info("Model loaded successfully")

        return model

    except Exception as e:
        logger.exception("Model loading failed")
        raise RuntimeError(f"Unable to load model: {str(e)}")


# MICR OCR
def micr_ocr(gray_image):
    for psm in [6, 11]:
        for padding in [0, 1]:

            try:
                prediction_text = micr_detector.run_inference(
                    gray_image,
                    psm=[psm],
                    padding=[padding]
                )

                result = extract_field(prediction_text)

                if (
                    result.get("Cheque_Number")
                    and result.get("Account_Number")
                    and result.get("Routing_Number")
                ):
                    return result

            except Exception as e:
                logger.warning(f"MICR OCR failed: {str(e)}")

    return None


# CROP HELPERS
def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    return image[y1:y2, x1:x2]


def get_best_boxes(result):
    best_boxes = {}

    for idx, box in enumerate(result.boxes):

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        if (
            cls not in best_boxes
            or conf > float(result.boxes[best_boxes[cls]].conf[0])
        ):
            best_boxes[cls] = idx

    return best_boxes


# MAIN PREDICTION FUNCTION
def prediction(model, image, mode="OCR"):
    start_time = time.time()
    validate_image(image)
    result_output = {
        "Date": "",
        "Cheque_Number": "",
        "Account_Number": "",
        "Routing_Number": "",
        "Amount": "",
        "Maker": "",
    }

    try:
        detections = model(image, conf=0.50)
        for result in detections:
            best_boxes = get_best_boxes(result)
            for cls, idx in best_boxes.items():
                box = result.boxes[idx]
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                cropped = crop_image(image, box)
                # MICR
                if cls == CLASS_MICR and mode in ["MICR", "OCR"]:
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    processed = preprocess_array(gray)
                    processed = np.expand_dims(processed, axis=0)
                    micr_prediction = predict_model(
                        test_images=processed
                    )
                    micr_prediction = ''.join([
                        '⑆' if i == 'T'
                        else '⑈' if i == 'O'
                        else i
                        for i in str(micr_prediction)
                    ])
                    micr_result = extract_field(micr_prediction)
                    if micr_result:
                        result_output["Cheque_Number"] = (
                            micr_result.get("Cheque_Number", "")
                        )
                        result_output["Account_Number"] = (
                            micr_result.get("Account_Number", "")
                        )
                        result_output["Routing_Number"] = (
                            micr_result.get("Routing_Number", "")
                        )
                    else:
                        save_failed_prediction(gray, "MICR")
                # DATE
                elif cls == CLASS_DATE and mode in ["DM", "OCR"]:
                    date_result = fix_date(cropped)
                    result_output["Date"] = date_result or ""
                    if not date_result:
                        save_failed_prediction(cropped, "Date")
                # AMOUNT
                elif cls == CLASS_AMOUNT and mode in ["DM", "OCR"]:
                    amount = ocr_amount.run_inference(
                        cropped,
                        text_cleaning_scheme=r"[^0-9.]+"
                    )
                    amount = re.sub(r"[^0-9.]", "", amount)
                    result_output["Amount"] = amount
                    if not amount:
                        save_failed_prediction(cropped, "Amount")
                # MAKER
                elif cls == CLASS_MAKER and mode in ["DM", "OCR"]:
                    maker = authorizer_ocr.run_inference(
                        cropped,
                        text_cleaning_scheme=r"[^0-9^a-z^A-Z.,]+"
                    )
                    result_output["Maker"] = maker
                    if not maker:
                        save_failed_prediction(cropped, "Maker")

        logger.info(
            f"Prediction completed in "
            f"{round(time.time() - start_time, 2)} sec"
        )

        return result_output

    except Exception as e:
        logger.exception("Prediction pipeline failed")
        raise RuntimeError(f"Prediction failed: {str(e)}")