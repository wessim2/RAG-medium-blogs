# RAG Medium Blogs

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for medium blogs using Llama 3.2 as the Large Language Model (LLM). The setup is designed to enable efficient retrieval and generation of information based on medium blog content.

## How It Works

1. Data Ingestion: The blogs are processed and converted into vector embeddings.
1. Indexing: The embeddings are stored in the Pinecone vector database.
1. Retrieval: Relevant embeddings are retrieved based on a user query.
1. Generation: Llama 3.2 generates detailed responses augmented by the retrieved information.
