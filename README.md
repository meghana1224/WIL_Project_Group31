# WIL_Project_Group31

ImmiBrain: an Australian visa assistant built with Retrieval-Augmented Generation (RAG).

Created for WIL Project Group 31, Case Studies in Data Science, RMIT.

---

## About

ImmiBrain answers questions about Australian visas using official Department of Home Affairs web pages saved as PDFs. Instead of answering from the language model's own memory, the app first finds the most relevant text in the PDFs and then asks the model to write an answer using only that text. Every answer shows the source file and page, so users can check it themselves.

The dataset has 60 PDFs covering six visa subclasses: 186, 189, 190, 482, 485 and 500.

> This is a university prototype and not immigration advice. Always confirm with Home Affairs or a registered migration agent.

## How it works

**Preparing the data** (`data_ingestion.py`)

```
PDFs -> clean and chunk -> embed -> ChromaDB
```

**Answering a question** (`src/rag.py` and `app.py`)

```
Question -> retrieve chunks -> keyword re-ranking -> prompt 
         -> Llama 3.1 -> answer + sources
```

| Step | What is used |
|---|---|
| PDF text extraction | pypdf, with regex cleaning of timestamps, URLs and page footers |
| Chunking | LangChain RecursiveCharacterTextSplitter (600 characters, 150 overlap) |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Vector database | ChromaDB (local) |
| Retrieval | Closest chunks by meaning, then re-ranked by shared keywords; top k kept (default 6) |
| Language model | Llama 3.1 8B, run locally with Ollama |
| Interface | Streamlit chat app with a sources panel |

## Repository structure

```
WIL_Project_Group31/
├── app.py                 Streamlit chat app (ImmiBrain)
├── data_ingestion.py      Reads the PDFs, chunks them and builds the ChromaDB database
├── requirements.txt       Python packages
├── src/
│   └── rag.py             Retrieval, keyword re-ranking and answer generation
├── eval/
│   ├── test_questions.py  30 test questions (answerable, informal wording, should-refuse)
│   └── evaluate.py        Runs the test questions and saves results.csv
├── data/
│   └── pdfs/              The 60 visa PDFs
└── chroma_db/             Vector database (created by data_ingestion.py)
```

## Setup

These steps are for Windows (PowerShell). Python 3.12 is recommended.

1. **Install Ollama** from https://ollama.com and download the model:
```powershell
   ollama pull llama3.1:8b
```

2. **Create a virtual environment and install the packages:**
```powershell
   py -3.12 -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
```

3. **Build the database** (only needed on first run, or when the PDFs change):
```powershell
   python data_ingestion.py
```

4. **Run the app** (Ollama must be running):
```powershell
   streamlit run app.py
```

## Evaluation

The test set in `eval/test_questions.py` has 30 questions: 20 answerable, 4 informally worded, and 6 that the documents cannot answer (the model should refuse these). Run it with:

```powershell
python eval/evaluate.py
```

It takes around 15 to 25 minutes and saves `eval/results.csv`. The script reports:

| Measure | What it shows |
|---|---|
| Wrongly refused | Answerable questions the model refused to answer |
| Correctly refused | Unanswerable questions the model refused to answer |
| Source recall | Whether the correct PDF was among the retrieved chunks |
| Citation validity | Whether the file and page cited in the answer were among the retrieved chunks |
| Faithfulness | Whether the answer is supported by the retrieved text (judged by the same Llama model) |
| Latency | Seconds taken per answer |

## Limitations

- Answers can change with how a question is worded.
- The 8B model sometimes leaves out exceptions to a rule.
- Citations are written by the model. They are checked only against the retrieved chunks, not for truth.
- The faithfulness judge is the same model that wrote the answers, so it may be too lenient.
- The PDFs are a snapshot. Visa rules change, so the data must be refreshed to stay current.


Work is tracked on our Trello board, and changes are made on separate branches and merged into `main` with pull requests.
