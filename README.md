# RAG Medium Blogs

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for medium blogs using Llama 3.2 as the Large Language Model (LLM). The setup is designed to enable efficient retrieval and generation of information based on medium blog content.

## How It Works

1. Data Ingestion: The blogs are processed and converted into vector embeddings.
1. Indexing: The embeddings are stored in the Pinecone vector database.
1. Retrieval: Relevant embeddings are retrieved based on a user query.
1. Generation: Llama 3.2 generates detailed responses augmented by the retrieved information.

## Configure Environment Variables
Create a .env file in the root of the repository and add the following content:
``` 
INDEX_NAME=medium-blogs-embeddings-index
PINECONE_API_KEY=your-pinecone-api-key
LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="medium-blogs"
```
Replace your-pinecone-api-key and your-langchain-api-key with your actual API keys.
