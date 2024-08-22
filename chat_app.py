import os
import time
from io import BytesIO
import streamlit as st
from config.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from src.rephrase import Rephrase
from src.rag_pipeline import RAG, TextProcessor

# Initialize global variables and classes
def initialize_global_state():
    """
    Initializes global state variables and necessary objects.
    """
    # Initialize TextProcessor with predefined chunk size and overlap
    st.session_state.text_processor = TextProcessor(CHUNK_SIZE, CHUNK_OVERLAP)

    # Set Pinecone API Key environment variable
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

    # Initialize session state for necessary variables
    session_variables = [
        'api_key', 'endpoint', 'version', 
        'model_name', 'embedding_model_name', 
        'openai_type', 'uploaded_files', 'chat_history', 
        'input_key', 'messages'
    ]

    for var in session_variables:
        if var not in st.session_state:
            st.session_state[var] = '' if var not in ['uploaded_files', 'messages', 'chat_history'] else []

# Function to display the model details form
def display_model_details_modal():
    """
    Displays a modal for the user to input OpenAI or Azure OpenAI credentials.
    """
    with st.expander("Please enter your Model Details:", expanded=False):
        service = st.selectbox("Select Service", ["OpenAI", "Azure OpenAI"])

        with st.form(key='model_details_form'):
            if service == "OpenAI":
                configure_openai()
            elif service == "Azure OpenAI":
                configure_azure_openai()

            st.warning("Use only text-embedding-ada-002 model since the index has been created according to its dimension")

            if st.form_submit_button("Submit"):
                st.success("Details submitted successfully!")
                initialize_rephrase_and_rag_objects()

# Function to configure OpenAI
def configure_openai():
    """
    Configures the session state for OpenAI usage.
    """
    st.session_state.openai_type = "openai"
    st.session_state.api_key = st.text_input("API Key", type="password")
    st.session_state.model_name = st.text_input("Model Name")
    st.session_state.embedding_model_name = st.text_input("Embedding Model Name")

# Function to configure Azure OpenAI
def configure_azure_openai():
    """
    Configures the session state for Azure OpenAI usage.
    """
    st.session_state.openai_type = "azure_openai"
    st.session_state.api_key = st.text_input("API Key", type="password")
    st.session_state.endpoint = st.text_input("Endpoint")
    st.session_state.version = st.text_input("Version")
    st.session_state.model_name = st.text_input("Model Name")
    st.session_state.embedding_model_name = st.text_input("Embedding Model Name")

# Function to initialize Rephrase and RAG objects
def initialize_rephrase_and_rag_objects():
    """
    Initializes the Rephrase and RAG objects based on session state.
    """
    if st.session_state.api_key:
        st.session_state.rephrase_obj = Rephrase(
            gpt_engine_name=st.session_state.model_name, 
            api_key=st.session_state.api_key, 
            azure_endpoint=st.session_state.endpoint, 
            api_version=st.session_state.version, 
            openai_type=st.session_state.openai_type
        )

        st.session_state.rag_obj = RAG(
            PINECONE_INDEX_NAME, 
            st.session_state.text_processor, 
            PINECONE_API_KEY, 
            st.session_state.model_name, 
            st.session_state.embedding_model_name, 
            st.session_state.api_key, 
            st.session_state.endpoint, 
            st.session_state.version, 
            st.session_state.openai_type
        )

# Function to display notification
def show_notification(message, message_type="success"):
    """
    Displays a notification on the screen.

    Args:
        message (str): The message to display.
        message_type (str): Type of the message ('success' or 'error').
    """
    css = f"""
    <style>
    .notification {{
        position: fixed;
        top: 0;
        right: 0;
        margin: 20px;
        padding: 10px 20px;
        color: white;
        border-radius: 5px;
        z-index: 1000;
        background-color: {"#4CAF50" if message_type == "success" else "#f44336"};
    }}
    </style>
    """
    notification_placeholder = st.empty()
    notification_placeholder.markdown(f'{css}<div class="notification">{message}</div>', unsafe_allow_html=True)
    time.sleep(5)
    notification_placeholder.empty()

# Function to handle file upload and insertion into the RAG index
def handle_file_upload(uploaded_files):
    """
    Handles the file upload process and inserts the files into the RAG index.

    Args:
        uploaded_files (list): List of uploaded file objects.
    """
    for uploaded_file in uploaded_files:
        if uploaded_file not in st.session_state.uploaded_files:
            bytes_data = uploaded_file.read()
            file_content = BytesIO(bytes_data)
            file_name = uploaded_file.name
            st.session_state.uploaded_files.append(uploaded_file)

            try:
                if st.session_state.api_key:
                    show_notification(f"{uploaded_file.name} preprocessing started, Please wait for a while!")
                    st.session_state.rag_obj.insert_doc(file_content, file_name)
                    show_notification(f"{uploaded_file.name} inserted successfully!")
                else:
                    show_notification("Please enter your OpenAI credentials before uploading documents!", message_type='error')
            except Exception as e:
                show_notification("Something went wrong while preprocessing, please try again!", message_type='error')

# Function to build the chat history from session state
def build_chat_history():
    """
    Constructs the chat history from the last few exchanges.

    Returns:
        str: The chat history string.
    """
    history = ""
    for message in st.session_state.messages[-2:]:
        if message['role'] == 'user':
            history += f"User: {message['content']}\n"
        elif message['role'] == 'Bot':
            history += f"Bot: {message['content']}\n"
    return history

# Function to process user input and generate responses
def process_input_and_generate_response(query):
    """
    Processes the user query, rephrases it, and generates a response using RAG.

    Args:
        query (str): The user query.
    """
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    try:
        if st.session_state.api_key:
            history = build_chat_history()
            rephrased_query = st.session_state.rephrase_obj.followup_query(query, history)
            answer = st.session_state.rag_obj.qna(query=rephrased_query)

            with st.chat_message("BOT"):
                st.write_stream(answer)

            st.session_state.messages.append({"role": "Bot", "content": st.session_state.rag_obj.answer})
        else:
            st.error("Please enter your API Key and other details at the top.")
    except Exception as e:
        st.error("Something went wrong, please try again!")

# Main function to run the chatbot interface
def main():
    # Initialize session state and objects
    initialize_global_state()

    # Set page title and layout
    st.set_page_config(page_title="Pawan Kalyan Chatbot Interface", layout="wide")

    # Display the modal to input model details
    display_model_details_modal()

    # Main content area
    st.markdown("Pawan's Data has already been ingested! Please query regarding pawan or you can upload a new document to query upon.")
    st.write("You can upload documents in the sidebar.")

    # Sidebar for file upload
    with st.sidebar:
        uploaded_files = st.file_uploader('Upload files', type=['pdf'], accept_multiple_files=True, label_visibility="hidden")
        if uploaded_files:
            handle_file_upload(uploaded_files)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input processing
    if prompt := st.chat_input("Type your query..."):
        process_input_and_generate_response(prompt)

    # Auto scroll to keep the chat view updated
    st.empty()

if __name__ == "__main__":
    main()