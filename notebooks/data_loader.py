from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import json_loader


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain data structure 
    Supported: PDF, TXT, CSV, Excel, Word or JSON
    """

    # Use Project Rot Data Folder

    data_path = Path(data_dir).resolve()
    print(f"Data Path:{data_path}")
    documents = []

    #PDF Files

    pdf_files = List(data_path.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")
    for pdf_file in pdf_files:
        print(f"Loading PDF file: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            print(f"Loaded {len(loader.load())} documents from {pdf_file} to create total number of documents as {len(documents)}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

            



