# ## This script takes all the outputs from a given folder and merges them into 1 final Excel file

import os
from pathlib import Path
from PyPDF2 import PdfReader
import pdfplumber
import pandas as pd
import camelot
import re
import sys
import warnings
from process_excel_export import read_from_excel, extract_ebom, prepare_output, add_child_type, get_all_child_descriptions, print_colors, sort_table_rows
from Resideo.HelperFiles.callable_GUI_chooseFile import chooseFile, chooseFile, chooseMoreFiles, chooseInputFile

folder = "C:\\Users\\simona.matouskova\\OneDrive - Akkodis\\Documents\\Resideo\\Resideo\\Data\\Real\\New_format_pdfs\\simkas_files\\few_processed"

content = {}
for ind, file in enumerate(os.listdir(folder)):
    file_path = Path(folder + '\\' + file)
    print(file_path)
    content[ind] = pd.read_excel(file_path, engine='openpyxl')

## Take header from first table
header = content[1].columns.tolist()







print("done")