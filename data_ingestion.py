# LIBRARIES 
# file manipulation 
import os
import glob
import shutil 

# text manipulation
import re

# libraries for vectorising 
# everything installed with pip to prevent environment issues 
from pypdf import PdfReader
############################################################################################################

# SET PATHS
PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
############################################################################################################

# GLOBAL VARS DECLARATION 
# to clean out unrelevant/noise data 
NOISE_PATTERNS = [
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*[APap][Mm].*",
    r"https?://\S+",
    r"Ask a question",
    r"Australian Government - Department of Home Affairs",
    r"Immigration and citizenship",
    r"\(#\w[\w-]*\)",
    r"\b\d+/\d+\b",
]

############################################################################################################
# FUNCTIONS DECLARATION 
############################################################################################################

############################################################################################################
# Text Manipulation
# sub any text that is in those noise pattern into empty strings 
def clean_text(text):
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    # multispace into a single space incase >=2 subsequent patterns matched 
    text = re.sub(r"\s+", " ", text).strip()
    return text

# extract text from pdf with the py_pdf 
# it returns pages 
def extract_text_with_pages(pdf_path):
    # read pfd in pages
    pdf_reader = PdfReader(pdf_path)
    pages_texts = []

    # for each page, extract text, clean, save 
    for i, page in enumerate(pdf_reader.pages):
        texts = page.extract_text() or ""
        texts = clean_text(texts)
        if texts:
            # save in tuple (index, texts) for each page
            pages_texts.append((i + 1, texts))

    return pages_texts

############################################################################################################
# main program 
def main():
    # get all pdf paths 
    pdf_files_paths = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    # check if pdfs exists, if no pdfs found, end ingest process
    if not pdf_files_paths:
        print(f"No PDFs found in {PDF_DIR}. Add visa PDFs there first.")
        return

    # get each pdf file, extract the text in pages, and chunk it 
    for pdf_file in pdf_files_paths:
        file_name = os.path.basename(pdf_file)
        print(f"Processing {file_name} ...")

        # extract texts from pdf file in pages :-> [(page_num,text), (page_num,text)]
        pages = extract_text_with_pages(pdf_file)

    print(f"\n{len(pdf_files_paths)} PDF files.")

if __name__ == "__main__":
    main()