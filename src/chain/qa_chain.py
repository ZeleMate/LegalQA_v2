from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.rag.retriever import CustomRetriever
from src.prompts.legal_assistant_prompt import legal_assistant_prompt

def build_qa_chain(retriever: CustomRetriever, temperature: float = 0) -> ConversationalRetrievalChain:
    """
    Builds a QA chain using LangChain.

    Args:
        retriever: A CustomRetriever object.
        temperature (float): The temperature of the LLM.

    Returns:
        ConversationalRetrievalChain: The configured QA chain.
    """
    llm = ChatOpenAI(temperature=temperature)

    prompt = PromptTemplate.from_template(
        legal_assistant_prompt
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
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain