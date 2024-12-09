import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader('D:/projects/langchain/test/bolg/medium.txt')
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="llama3.2")

    print("Ingesting ...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name="medium-blogs-embeddings-index")
    print("Finish")