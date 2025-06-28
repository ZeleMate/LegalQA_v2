from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pathlib import Path
import os

from src.rag.retriever import RerankingRetriever

def format_docs(docs):
    """Helper function to format documents for the prompt."""
    return "\n\n".join(
        f"### Document ID: {doc.metadata.get('doc_id', 'N/A')}\\n"
        f"Content:\\n{doc.page_content}"
        for doc in docs
    )

def build_qa_chain(retriever: RerankingRetriever):
    """
    Builds a Question-Answering chain using LangChain Expression Language (LCEL).

    This chain takes a user's question, retrieves relevant documents using the
    provided RerankingRetriever, formats them into a context, and then uses an LLM
    to generate a final answer based on that context.

    Args:
        retriever: An instance of our custom RerankingRetriever.

    Returns:
        A Runnable object representing the QA chain.
    """
    # 1. Load the prompt template for the legal assistant
    prompt_path = Path(__file__).parent.parent / "prompts" / "legal_assistant_prompt.txt"
    try:
        template = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(f"Prompt file not found at: {prompt_path}")
        
    prompt = PromptTemplate.from_template(template)

    # 2. Define the LLM for generating the final answer
    llm = ChatOpenAI(
        model_name="gpt-4.1-2025-04-14",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 3. Build the LCEL chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain