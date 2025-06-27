from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import pandas as pd
from src.data.faiss_loader import load_faiss_index
from langchain_openai import OpenAIEmbeddings
from src.rag.retriever import CustomRetriever
from src.chain.qa_chain import build_qa_chain

app = FastAPI(
    title="LegalQA API",
    description="An API for answering legal questions.",
    version="1.0.0"
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PARQUET_PATH = os.getenv("PARQUET_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
ID_MAPPING_PATH = os.getenv("ID_MAPPING_PATH")

df = pd.read_parquet(PARQUET_PATH)
faiss_index, id_mapping = load_faiss_index(FAISS_INDEX_PATH, ID_MAPPING_PATH)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

retriever = CustomRetriever(
    embeddings=embeddings,
    faiss_index=faiss_index,
    id_mapping=id_mapping,
    documents_df=df,
)

qa_chain = build_qa_chain(retriever)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QuestionRequest):
    result = qa_chain.invoke({
        "question": req.question,
        "chat_history": []
    })
    return {
        "answer": result["answer"],
        "sources": [doc.metadata["doc_id"] for doc in result["source_documents"]]
    }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the LegalQA API!",
        "endpoints": {
            "/ask": "POST - Ask a question to the system"
        }
    }