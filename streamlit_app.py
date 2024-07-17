import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

def load_document(file):
    name, extension = os.path.splitext(file.name)

    if extension == '.pdf':
        loader = PyPDFLoader(file.name)
    elif extension == '.docx':
        loader = Docx2txtLoader(file.name)
    elif extension == '.txt':
        loader = TextLoader(file.name)
    else:
        raise ValueError('Unsupported file format')
    
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings(api_key):
    return OpenAIEmbeddings(openai_api_key=api_key)

def create_vectorstore(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)

def create_conversation_chain(vectorstore, api_key):
    llm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4-0613', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_user_input(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"Human: {message.content}")
        else:
            st.write(f"AI: {message.content}")

def main():
    st.set_page_config(page_title="GPT-4 Chatbot", page_icon=":robot:")
    st.header("GPT-4 Chatbot with Document Q&A")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])

    if uploaded_file and api_key:
        documents = load_document(uploaded_file)
        text_chunks = split_documents(documents)
        embeddings = create_embeddings(api_key)
        vectorstore = create_vectorstore(text_chunks, embeddings)
        st.session_state.conversation = create_conversation_chain(vectorstore, api_key)

        st.write("Document uploaded and processed. You can now ask questions about it.")

    user_input = st.text_input("Ask a question about the uploaded document:")

    if user_input and api_key:
        if st.session_state.conversation:
            handle_user_input(user_input)
        else:
            st.write("Please upload a document first.")

if __name__ == "__main__":
    main()