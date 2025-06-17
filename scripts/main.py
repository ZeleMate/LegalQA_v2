from dotenv import load_dotenv
import os
import pandas as pd
from src.data.faiss_loader import load_faiss_index
from src.rag.retriever import CustomRetriever
from src.chain.qa_chain import build_qa_chain
from langchain_openai import OpenAIEmbeddings

PARQUET_PATH = os.getenv("PARQUET_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
ID_MAPPING_PATH = os.getenv("ID_MAPPING_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

required_env_vars = [
    "PARQUET_PATH",
    "FAISS_INDEX_PATH",
    "ID_MAPPING_PATH",
    "OPENAI_API_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Hiányzó környezeti változók: {', '.join(missing_vars)}. Kérlek, ellenőrizd a .env fájlt.")

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