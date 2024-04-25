import streamlit as st

from qdrant_client import AsyncQdrantClient, models, QdrantClient
import os
from qdrant_client import AsyncQdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Qdrant

from langchain.chains import RetrievalQA

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

import uuid





# global variables
qdrant_client = QdrantClient(
    url="https://dd489e3d-8648-47fe-88cf-156f8e9e2c90.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="vCPG1V8yO1kaGzTMWA9y5ql8nn8u-HWiHiKlTNNdE3MKrEI14efpCg",
)

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

llm = ChatGroq(api_key=os.getenv['GROQ_API_KEY'])


# Collection name for the user

# create a uuid and concatenate to collection name

# unique_id = uuid.uuid4()
# unique_id = str(unique_id)
# collection_name = f'website_info_vector_store-{unique_id}'

collection_name = 'user_website'



def load_env_variables():
    os.environ["QDRANT_API_KEY"] = "vCPG1V8yO1kaGzTMWA9y5ql8nn8u-HWiHiKlTNNdE3MKrEI14efpCg"


def ingestion_into_vector_store(website_url):
    # create collection for the user in vector store
    
    collections_in_store =  qdrant_client.get_collections()
    existing_collections = [] 
    for each_collection in collections_in_store.collections:
        existing_collections.append(each_collection.name)

    collectionExists = True if collection_name in existing_collections else False

    # If collection does not exist create one using qdrant client and ingest data
    if collectionExists == False:
        # Load the webpage using its url
        loader = WebBaseLoader(website_url, verify_ssl=True)
        loader.requests_kwargs = {'verify': True}

        data_from_website = loader.load()

        # Chunk data from website

        # chunk the text from website
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200) 

        chunks_of_text = text_splitter.split_documents(data_from_website)

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
    
        vector_store = Qdrant.from_documents(
            chunks_of_text,
            embeddings_model,
            url="https://dd489e3d-8648-47fe-88cf-156f8e9e2c90.us-east4-0.gcp.cloud.qdrant.io:6333", 
            api_key="vCPG1V8yO1kaGzTMWA9y5ql8nn8u-HWiHiKlTNNdE3MKrEI14efpCg",
            collection_name=collection_name,
            force_recreate=True
        )

def reset_user_session():
    # Reset user session in browser

    # Delete the user collection from the database
    qdrant_client.delete_collection(collection_name=collection_name)

    # # reload the page
    # st.experimental_rerun()




def streamlit_app():
    load_env_variables()
    st.set_page_config(page_title='Chat with any website!')


    # Have a sidebar prompting user to enter a website url
    st.sidebar.title('Website URL')
    website_url = st.sidebar.text_input('Paste your website url', label_visibility='collapsed',placeholder='https://www.react-redux.com/')

    # Hook up a delete session button and function
    delete_session = st.sidebar.button('Reset Session')
    if delete_session:
        reset_user_session()
    

    # Ingest data from website into the qdrant vector store
    if website_url:
        # Setup the chat application UI
        st.title('Chat with your website!')
        st.write('Chat with any website, crypto whitepaper, coding documentation etc')

        # Ingest data from website url
        ingestion_into_vector_store(website_url)

        # Setup conversation memory
        conversational_memory_length = 5
        memory = ConversationBufferWindowMemory(k=conversational_memory_length) # Set the memory length    

        # Setup up chat history object
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history=[]
        else:
            for message in st.session_state.chat_history:
                memory.save_context({'input':message['human']},{'output':message['AI']})


        user_question = st.chat_input("Ask your question!")

        if user_question:
            # Connect to a document store
            doc_store = Qdrant(
                client=qdrant_client,
                collection_name=collection_name, # Can change the collection here
                embeddings = embeddings_model
            )

            # Create the retrieval chain
            qa = RetrievalQA.from_chain_type(
                llm = llm,
                chain_type="stuff",
                retriever= doc_store.as_retriever(),
                return_source_documents=True
            )

            # Get the response
            response = qa.invoke(user_question)

            answer = response['result']

            # Append question and answer to chat history session state
            message = {'human': user_question, 'AI' : answer}

            st.session_state.chat_history.append(message)

            # Render the conversation on chat screen
            for message in st.session_state['chat_history']:
                with st.chat_message("user"):
                    st.markdown(message['human'])
                
                with st.chat_message('assistant'):
                    st.markdown(message['AI'])
    
    else:
        st.title(':green[Please provide a website url to get started! 🙂]')
    

        




if __name__ == '__main__':
    streamlit_app()


# Notes for developer

# 1)later will implement functionality for deleting a user session by deleting their vector collection, i will have to assign a unique name for each collection for every diffrent user and persist their session in the browser