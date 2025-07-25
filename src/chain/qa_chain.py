import logging
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag.retriever import RerankingRetriever


def format_docs(docs):
    """Helper function to format documents for the prompt."""
    logger = logging.getLogger(__name__)
    logger.info(
        "format_docs called with {} documents. Type of first: {}".format(
            len(docs), type(docs[0]) if docs else 'N/A'
        )
    )
    # Log the first few context chunks for debugging
    for i, doc in enumerate(docs[:3]):
        chunk_id = getattr(doc, 'metadata', {}).get('chunk_id', 'N/A')
        page_content = getattr(doc, 'page_content', '')[:2]
        if len(page_content) > 2:
            page_content += ".."
        logger.info(
            "CONTEXT CHUNK {} (chunk_id={}):".format(i + 1, chunk_id)
        )
        logger.info(
            "CONTENT PREVIEW: {}".format(page_content)
        )
    return (
        "\n\n".join(
            "### Document ID: {}\nContent:\n{}...".format(
                getattr(doc, 'metadata', {}).get('chunk_id', 'N/A'),
                (getattr(doc, 'page_content', '')[:2] + ".."
                 if len(getattr(doc, 'page_content', '')) > 2
                 else getattr(doc, 'page_content', ''))
            )
            for doc in docs
        )
    )


def build_qa_chain(retriever: RerankingRetriever, google_api_key: str):
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
    prompt_path = (
        Path(__file__).parent.parent /
        "prompts" /
        "legal_assistant_prompt.txt"
    )
    try:
        template = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(
            "Prompt file not found at: {}...".format(str(prompt_path)[:60])
        )

    prompt = PromptTemplate.from_template(template)

    # 2. Define the LLM for generating the final answer
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        api_key=google_api_key,
    )

    # 3. Build the LCEL chain
    async def retrieve_context_and_question(question):
        logger = logging.getLogger(__name__)
        logger.info(
            "retrieve_context_and_question called with question: {}".format(question)
        )
        docs = await retriever._aget_relevant_documents(question)
        logger.info(
            "retrieve_context_and_question returning {} documents.".format(len(docs))
        )
        if docs:
            logger.info(
                "Type of first: {}".format(type(docs[0]))
            )
        return {"context": format_docs(docs), "question": question}

    rag_chain = (
        RunnableLambda(retrieve_context_and_question)
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
