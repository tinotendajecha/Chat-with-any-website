import os
import time
import asyncio
from langchain_openai import OpenAIEmbeddings

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

from langchain.chains import VectorDBQA
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Qdrant
from langchain.schema import retriever

import streamlit as st
from dotenv import load_dotenv

import time


def main():
    # Load the environment variables
    load_dotenv()

    # Configure the embedding model
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=768)
    st.set_page_config(page_title='Crypto compliance')

    st.header('Ask Crypto Regulatory Compliance Question!')
    st.write("Langchain pipeline")
    
    # Grab the user question
    user_question = st.text_input("Ask me anything crypto regulatory question!")

    qdrant_host = os.getenv('QDRANT_HOST')

    # Connect to the qdrant client
    client = QdrantClient(url=qdrant_host)

    # Initialize the document store
    doc_store = Qdrant(
        client=client,
        collection_name='test 768-dim', # Can change the collection here
        embeddings = embeddings_model
    )

    llm = OpenAI()

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever= doc_store.as_retriever(),
        return_source_documents=True
    )

    if user_question:
        st.write(f'Question: {user_question}')

        # Hook up the user question
        response = qa.invoke(user_question)

        st.write(response['result'])


        # print(f"Response: {response['result']}")

        


if __name__ == '__main__':
    main()