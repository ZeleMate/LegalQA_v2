from dotenv import load_dotenv
import os
import pandas as pd
from src.data.faiss_loader import load_faiss_index
from src.rag.retriever import CustomRetriever
from src.chain.qa_chain import build_qa_chain
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

PARQUET_PATH = os.getenv("PARQUET_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
ID_MAPPING_PATH = os.getenv("ID_MAPPING_PATH")

df = pd.read_parquet(PARQUET_PATH)
faiss_index, id_mapping = load_faiss_index(FAISS_INDEX_PATH, ID_MAPPING_PATH)
embeddings = OpenAIEmbeddings()

retriever = CustomRetriever(
    embeddings=embeddings,
    faiss_index=faiss_index,
    id_mapping=id_mapping,
    documents_df=df,
)

qa_chain = build_qa_chain(retriever)

print("\n📘 LegalQA kérdezz-felelek. Írd be a kérdésed, vagy 'exit'-tel kiléphetsz.")
chat_history = []

while True:
    question = input("\n❓ Kérdés: ")
    if question.strip().lower() == ["exit", "quit"]:
        break
    
    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
        })
    
    print(f"\n🧾 Válasz:\n{result['answer']}")
    chat_history.append((question, result["answer"]))