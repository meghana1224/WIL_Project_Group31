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
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
############################################################################################################

# SET PATHS
PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
# ChromaDB collection name 
COLLECTION_NAME = "australian_visa_docs"
############################################################################################################

# GLOBAL VARS DECLARATION 
# how many chars each chunk should be 
CHUNK_SIZE = 600
# how many chars chunks can overlap each other 
CHUNK_OVERLAP = 150

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

# chunks texts with the recrusive text spitter 
def chunk_document(pages, file_name):
    # create splitter instance
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ",""], # possible delims
    )

    chunks = []
    metadatas = []
    # split each page in chunks 
    for page_num, text in pages:
        for chunk in splitter.split_text(text): 
            chunks.append(chunk)
            metadatas.append({"source": file_name, "page": page_num})

    return chunks, metadatas

############################################################################################################
# ChromaDB 
def create_db_collection():
    # clear old chorma db if exists to build with new pdfs (if there is any)
    if os.path.exists(CHROMA_DB_PATH): 
        shutil.rmtree(CHROMA_DB_PATH)

    # create path to store chromadb 
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    # create persistent client save chromadb locally 
    local_db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # embedder model 
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = "BAAI/bge-small-en-v1.5"
    )

    # delete collection if exists to "rebuild" db incase there are new pdfs 
    try:
        local_db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # create collection 
    new_collection = local_db.create_collection(
        name=COLLECTION_NAME, embedding_function = embedder
    )

    return new_collection

# add chunked text into chroma db 
def add_to_collection(chroma_db_collection, all_chunks, all_metadatas, all_ids):     
    batch_size = 100

    # add to collection in batches - recommended by chromaDB
    for i in range(0, len(all_chunks), batch_size): # add 100 chunks at a time
        chroma_db_collection.add(
            documents = all_chunks[i : i + batch_size],
            metadatas = all_metadatas[i : i + batch_size],
            ids = all_ids[i : i + batch_size],
        )

############################################################################################################
# main program 
def main():
    # get all pdf paths 
    pdf_files_paths = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    # check if pdfs exists, if no pdfs found, end ingest process
    if not pdf_files_paths:
        print(f"No PDFs found in {PDF_DIR}. Add visa PDFs there first.")
        return

    # create db and get new collection 
    chroma_db_collection = create_db_collection() 

    # MAIN CHUNKING PROCESS
    # to save chunks, metadatas, and id FOR ALL PDFS in PDF_DIR
    all_chunks = []
    all_metadatas = []
    all_ids = []
    doc_counter = 0

    # get each pdf file, extract the text in pages, and chunk it 
    for pdf_file in pdf_files_paths:
        file_name = os.path.basename(pdf_file)
        print(f"Processing {file_name} ...")

        # extract texts from pdf file in pages :-> [(page_num,text), (page_num,text)]
        pages = extract_text_with_pages(pdf_file)
        # chunk pages
        chunks, metadatas = chunk_document(pages, file_name)
        # add to "all info" lists 
        for chunk, meta in zip(chunks, metadatas):
            all_chunks.append(chunk)
            all_metadatas.append(meta)
            all_ids.append(f"doc_{doc_counter}")
            doc_counter += 1

        print(f"  -> {len(chunks)} chunks")

     # add chunked data into chromaDB
    add_to_collection(chroma_db_collection, all_chunks, all_metadatas, all_ids)

    print(f"\nCompleted data ingestion. Chunked {len(all_chunks)} chunks from {len(pdf_files_paths)} PDF files.")
    print(f"Vector store saved at: {CHROMA_DB_PATH}")
    
if __name__ == "__main__":
    main()