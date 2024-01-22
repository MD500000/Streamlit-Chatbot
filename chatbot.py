import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, Document
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore
import openai
import itertools
from llama_index import SimpleDirectoryReader
from llama_index.embeddings import OptimumEmbedding
from datetime import datetime
from chromadb.utils import embedding_functions
from uuid import uuid4
import chromadb


openai.api_key = st.secrets.openai_key
st.header("Chat with MD's test project.")


if 'user_messages' not in st.session_state:
    st.session_state.user_messages = []

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Let's chat!"}
    ]


chroma_client = chromadb.PersistentClient(path="./db")

openai_embeddingFunction = embedding_functions.OpenAIEmbeddingFunction(
     api_key= st.secrets.openai_key,
     model_name="text-embedding-ada-002"
)

chroma_collection = chroma_client.get_or_create_collection("chat_history", embedding_function=openai_embeddingFunction)
embed_model = "default"

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0, system_prompt= "You are a chatbot who's purpose is to be friendly and talk with the user casually."))

vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
storage_context = StorageContext.from_defaults(vector_store = vector_store)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context = storage_context, service_context= service_context)

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)



if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_messages.append(prompt)


for message in st.session_state.user_messages:
    chroma_collection.add(
        documents=[message],
        metadatas=[{"time": str(datetime.now())}],
        ids=[str(uuid4())]
    )

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])



SystemPromptTest = """
You are having a conversation with a user.
{context}
"""

results = chroma_collection.query(
        query_texts=["What is relevant to : " +str(prompt)],
        n_results=3,
    )

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #systemPrompt = "Have a normal conversation with the user but check the query to understand the history of the user's previous messages as context"
            combined_prompt = SystemPromptTest.format(context = results)
            response = st.session_state.chat_engine.chat(combined_prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history       