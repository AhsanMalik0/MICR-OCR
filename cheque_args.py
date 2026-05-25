import argparse
import sys
import os
class arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser("Cheque Detection OCR Argument")
        self.parser.add_argument('--pading', type=int, nargs='+', default=[0,2,9,3,15], help="Padding for Images")
        self.parser.add_argument('--PSM', type=int, nargs='+', default=[6,11,8], help="PSM for Tesseract Configuration")
        self.parser.add_argument('--OSM', type=int, default=3, help="OSM for Tesseract Configuration")
        self.parser.add_argument('--conf_score', type=int, default=40, help="Confidence Score Threshold")
        self.parser.add_argument('--Transit', type=str, default='⑆', help="Transit Symbol")
        self.parser.add_argument('--Amount', type=str, default="⑇" , help="Amount Symbol")
        self.parser.add_argument('--On_us', type=str, default="⑈", help="On-us Symbol")
        self.parser.add_argument('--Account_Pattern', type=str, default=r'((?=\d{8,})(?!0{4})\d{8,})⑈',help="Account Detecting Pattern")
        self.parser.add_argument('--Routing_Pattern', type=str, default=r'⑆(\d{9})⑆',help="Routing Detecting Pattern")
        self.parser.add_argument('--MICR_Char', type=str, default=r'[^0-9⑈⑆ ]+',help="MICR Description Charactor")
        self.parser.add_argument('--Clean_result', type=str, default=r'[^0-9 ]+',help="Clean Detection")
        
        self.parser.add_argument('--Cheque_Pattern_I', type=str, default=r'⑈(?:0{4,})?(\d{4,6})⑈',help="Cheque Detecting Pattern-I")
        self.parser.add_argument('--Cheque_Pattern_II', type=str, default=r'(?:0{4,})?(\d{4,6})⑈',help="Cheque Detecting Pattern-II")
        self.parser.add_argument('--Cheque_Pattern_III', type=str, default=r'(?:0{4,})?(\d{4,6})',help="Cheque Detecting Pattern-III")
        self.parser.add_argument('--Model_Path', type=str, default="./weights/best.pt", help= "Model Path")


        if "ipykernel" in sys.modules:
            self.args = self.parser.parse_args([])
        else:
            self.args = self.parser.parse_args()

    def get_args(self):
        return self.args