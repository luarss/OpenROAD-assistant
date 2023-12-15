from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import shutil
import os
from utils import load_docs_from_jsonl
from timer_utils import Timer

def load_embedding_model(model_name="all-MiniLM-L6-v2",
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": False}):
    # Supported: "all-MiniLM-L6-v2", "hkunlp/instructor-large"
    embedding_model = HuggingFaceEmbeddings(model_name= model_name,
                                            model_kwargs = model_kwargs,
                                            encode_kwargs = encode_kwargs)
    return embedding_model


def generate_new_embeddings(chunks, model_name="all-MiniLM-L6-v2",
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": False}):
    embedding_model = load_embedding_model(model_name, model_kwargs, encode_kwargs)    
    # Regenerate DB
    if os.path.isdir("./chroma_db"): shutil.rmtree("./chroma_db")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
    retriever = db.as_retriever()
    return db, retriever

def load_existing_embeddings(model_name="all-MiniLM-L6-v2",
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": False}):
    embedding_model = load_embedding_model(model_name, model_kwargs, encode_kwargs)    
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = db.as_retriever()
    return db, retriever

if __name__ == "__main__":
    t = Timer()

    t.start()
    chunks = load_docs_from_jsonl("./tempdata/data.jsonl")
    ret = generate_new_embeddings(chunks)
    t.stop()