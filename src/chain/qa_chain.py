from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.rag.retriever import CustomRetriever
from pathlib import Path
import os

legal_assistant_prompt_path = Path(__file__).parent.parent / "prompts" / "legal_assistant_prompt.txt"

def build_qa_chain(retriever: CustomRetriever) -> ConversationalRetrievalChain:
    """
    Builds a QA chain using LangChain.

    Args:
        retriever: A CustomRetriever object.

    Returns:
        ConversationalRetrievalChain: The configured QA chain.
    """
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-2025-04-14"
    )

    prompt = PromptTemplate.from_template(
        legal_assistant_prompt_path.read_text()
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=CustomRetriever(
            embeddings=retriever.embeddings,
            faiss_index=retriever.faiss_index,
            id_mapping=retriever.id_mapping,
            documents_df=retriever.documents_df
        ),
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        },
    )

    return chain