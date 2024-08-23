# **Chatbot Interface with Rephrasing and RAG Integration**

## **Overview**

This repository hosts a chatbot interface that leverages OpenAI/Azure OpenAI for rephrasing user queries and integrates a Retrieval-Augmented Generation (RAG) pipeline with Pinecone to deliver accurate and contextually relevant responses. The application allows users to upload documents, query them, and receive intelligent answers based on the uploaded content.

## **Features**
- **Rephrasing Module:** Enhances query understanding by rephrasing user inputs.
- **RAG Pipeline:** Retrieves relevant document chunks and generates contextually accurate responses.
- **Document Ingestion:** Supports PDF uploads, processes them into chunks, and stores them in a Pinecone vector index for efficient querying.
- **User Interface:** A clean and user-friendly interface built with Streamlit.

## **Live Demo**
Access the live application here: [Streamlit App](https://pawan-chatbot.streamlit.app/)

## **Setup and Installation**

### **Prerequisites**
Before setting up the project, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- A Pinecone API Key
- OpenAI/Azure OpenAI API Key and Endpoint (if using Azure)

### **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/PawanKalyanJada/ragbot
   cd your-repo
   ```

2. **Install Dependencies**
   Create a virtual environment and activate it:
   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**
   Replace your creds in`config.py` file in the config directory of the project and add the following:
   ```bash
   PINECONE_API_KEY=your-pinecone-api-key
   ```

5. **Run the Application**
   Launch the Streamlit app using the following command:
   ```bash
   streamlit run app.py
   ```

   This will open the application in your default web browser.

## **Usage**

1. **Model Setup:**
   - Select between OpenAI or Azure OpenAI in the app interface.
   - Enter the required API credentials and model details.
   - Submit to initialize the model.

2. **Document Upload:**
   - Upload PDF documents via the sidebar.
   - The documents are processed and indexed in Pinecone for querying.

3. **Querying:**
   - Input your query in the chatbox.
   - The system rephrases the query (if necessary), retrieves relevant chunks, and generates a response.

## **Project Structure**

```bash
├── chat_app.py                   # Main application file
├── requirements.txt         # Python dependencies
├── src/
│   ├── rephrase.py          # Rephrase module
│   ├── rag_pipeline.py      # RAG pipeline and TextProcessor
├── config/
│   ├── config.py            # Configuration constants (API keys, etc.)
│   └── prompts.py           # Rephrase prompt template
└── README.md                # Project documentation
```
