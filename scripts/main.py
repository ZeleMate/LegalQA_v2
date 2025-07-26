import os

import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.chain.qa_chain import build_qa_chain
from src.data.faiss_loader import load_faiss_index
from src.rag.retriever import CustomRetriever

PARQUET_PATH = os.getenv("PARQUET_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
ID_MAPPING_PATH = os.getenv("ID_MAPPING_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

required_env_vars = [
    "PARQUET_PATH",
    "FAISS_INDEX_PATH",
    "ID_MAPPING_PATH",
    "GOOGLE_API_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(
        f"Missing environment variables: {', '.join(missing_vars)}. "
        f"Please check your .env file."
    )

df = pd.read_parquet(PARQUET_PATH)
faiss_index, id_mapping = load_faiss_index(FAISS_INDEX_PATH, ID_MAPPING_PATH)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)

retriever = CustomRetriever(
    embeddings=embeddings,
    faiss_index=faiss_index,
    id_mapping=id_mapping,
    documents_df=df,
)

qa_chain = build_qa_chain(retriever, GOOGLE_API_KEY)

print("\n📘 LegalQA Bot. Type your question or 'exit' to quit.")
chat_history = []

while True:
    question = input("\n❓ Question: ")
    if question.strip().lower() in ["exit", "quit"]:
        break

    result = qa_chain.ainvoke({"question": question, "chat_history": chat_history})

    print(f"\n🧾 Answer:\n{result['answer']}")
    chat_history.append((question, result["answer"]))
