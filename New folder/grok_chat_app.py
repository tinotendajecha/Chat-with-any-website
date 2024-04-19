# Langchain pipeline - Streamlit app for querying crypto regulatory compliance questions
# Make sure to write the Qdrant Host [QDRANT_HOST] Url  and OpenAI API key [OPENAI_API_KEY] in the .env file before running this application

import os
import time
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Qdrant
import streamlit as st
from dotenv import load_dotenv
import time

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Read from .env file 
qdrant_host = os.getenv('QDRANT_HOST')

# Connect to the qdrant client
client = QdrantClient(url=qdrant_host)


def main():
    # Load the environment variables
    load_dotenv()
    
    # Give chat app page a name / title
    st.set_page_config(page_title='Empathy Stories')

    # Heading for the app
    st.header('Chat with me!')

    additional_ctx = st.text_input("Additional context", value='Tell me a story that helps me understand how to deal with')
    
    # Grab the user question
    user_question = st.text_input("Ask your question!",placeholder='Prompt')
    

    # Check if the user has asked a question
    if user_question:

        prompt = additional_ctx + ' ' + user_question

        print(prompt)
        time.sleep(20)

        # Configure the embedding model
        # embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=384)

        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        # Initialize the document store
        doc_store = Qdrant(
            client=client,
            collection_name= 'Empathy_Tales', # Can change the collection here 
            embeddings = embeddings_model
        )

        # Initialize the OpenAI model
        # llm = OpenAI()
        llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key='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')


        # Initialize the retrieval QA chain
        qa = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever= doc_store.as_retriever(),
            return_source_documents=True,
        )

        # Render the user question on the screen
        st.markdown(f':green[Question:] {user_question}')

        # Hook up the user question
        response = qa.invoke(prompt)

        # Print the response below
        st.markdown(':green[Response:]')
        st.write(response['result'])

        # Extract source docs meta data
        sources = response['source_documents']
        
        st.markdown(f':green[Sources used: ]')
        
        for source in sources:
            source = source.page_content[1:-1]

            st.write(f':green[{source}]')
            st.divider()


if __name__ == '__main__':
    main()