import os 

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

from langchain import hub

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__" : 
    print("Retrieving ...")
    
    embeddings = OllamaEmbeddings(model="llama3.2")
    llm = ChatOllama(model="llama3.2")

    query = "what is Pinecone?"
    # chain = PromptTemplate.from_template(template=query) | llm

    vectorstore = PineconeVectorStore(
        index_name=os.environ['INDEX_NAME'], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input":query})

    print(result)

    # another prompt template
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question:{question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question":RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)