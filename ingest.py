# ingest.py
import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import DATA_PATH, DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def load_document():
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    return loader.load()

def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)

def create_vector_database(chunks):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

def main():
    if not os.path.exists(DATA_PATH):
        print("document.txt ფაილი ვერ მოიძებნა data საქაღალდეში.")
        return
    docs = load_document()
    chunks = split_document(docs)
    create_vector_database(chunks)
    print("🔥 ვექტორული ბაზა წარმატებით შეიქმნა!")

if __name__ == "__main__":
    main()
