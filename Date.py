# IMPORTING LIBRARIES
import re
import easyocr
from datetime import datetime
import numpy as np
import cv2
from classes import MICR_Detection
from PIL import Image, ImageOps, ImageEnhance
from Authorizer import SimpleOCR
import numpy as np
import cv2
import re

# PREDICTION CLASS CALLING
ocr = SimpleOCR()

# FUNCTION FOR CORRECTING DATE
def date_prediction(parts):
    Date = ""
    month = day = year = " "

    # OCR MISREAD CORRECTION
    corrections = {
        'l': '1', 'I': '1', '|': '1',
        'o': '0', 'O': '0',
        'b': '6', 'B': '8',
        'Z': '2', 'S': '5',
        'g': '9'
    }

    # Month name mapping (case insensitive)
    month_names = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02', 'fe6': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07','ju1': '07',
        'aug': '08', 'august': '08', 'au9': '08',
        'sep': '09', 'sept': '09', 'september': '09',
        'oct': '10', 'october': '10', '0ct': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12'
    }

    # FUNCTION THAT CONVERT TEXT TO LOWCASE
    def correct(text):
        return ''.join(corrections.get(c.lower(), c.lower()) for c in text)

    # FUNCTION FOR DTECTION PREDICTION LEN IF THE LEN > 3 THAT RETURN INVALIDE DATE
    if len(parts) != 3:
        return  "-".join(part for parts in parts)

    # PROCESS FIXED ORDER ACC TO US FORMAT: [month, day, year]
    corrected_parts = [correct(p.strip()) for p in parts]
    raw_month, raw_day, raw_year = corrected_parts

    # --- MONTH PREDICTION ALGO ---
    if raw_month.isdigit():
        month = raw_month.zfill(2)
        
    else:
        lower_month = raw_month.lower()
        month = month_names.get(lower_month)
        
        # if not month:
        #     return ""

    # --- DAY PREDICTION ALGO ---
    if raw_day.isdigit() and 1 <= int(raw_day) <= 31:
        day = raw_day.zfill(2)
    # else:
    #     return ""

    # --- YEAR PREDICTION ALGO ---
    if raw_year.isdigit():
        if len(raw_year) == 4:
            year = raw_year
        elif len(raw_year) == 3:
            year = '20' + raw_year[-2:]
        elif len(raw_year) == 2:
            year = '20' + raw_year
        elif len(raw_year) == 1:
            year = '200' + raw_year
        # else:
        #     return None #f"Invalid year format: {raw_year}"
    else:
        return "-".join(str(i) for i in [month, day, year])

    if all([year, month, day]):
        print(f"{year}-{month}-{day}")
        return f'{month}-{day}-{year}'
        # return {"Year":year, "Month":month, "Day":day}
    else:
        return "-".join(str(i) for i in [month, day, year])


# MAIN DATE PREDICTION PIPELINE
def fix_date(image):
    # PREDICTION FUNCTION CALLING
    predicted = ocr.run_inference(image)
    # PREDICTION CLEANING ACC TO DESIRED OUTPUT
    text = re.sub(r'/{2,}', lambda m: m.group(0)[:-1] + '1', predicted)
    text = re.sub(r'[^0-9^a-z^A-Z.]', ' ', text)
    text = text.replace("/"," ")
    spliting = text.split(' ')
    spliting = [x for x in spliting if x]
    # normalized = predicted.strip()
    return date_prediction(spliting)