import os
import time
import PyPDF2
from config.prompts import QNA_PROMPT
from config.config import TOP_K, TEMPERATURE
from typing import List, Optional, Generator, Any
from openai import OpenAI, AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """
    A class responsible for processing text documents by extracting and splitting text into chunks.
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Initialize the TextProcessor with chunk size and overlap.
        
        Args:
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name='gpt-3.5-turbo',
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def extract_text(self, file_content: str, file_name: str) -> List[Document]:
        """
        Extracts text from a PDF file and returns a list of Document objects.
        
        Args:
            file_path (str): The path to the PDF file.
            file_name (str): The name of the PDF file.
            
        Returns:
            List[Document]: A list of Document objects containing the extracted text and metadata.
        """
        try:
            reader = PyPDF2.PdfReader(file_content)
            texts = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if(text):
                    texts.append(text)

            pdf_text = " ".join(texts)
            docs = [Document(page_content=doc, metadata={"filename": file_name}) 
                for doc in self.text_splitter.split_text(pdf_text)]
            return docs
        except Exception as e:
            raise Exception(f"Error in extracting text from PDF file: {e}")

class RAG:
    """
    A class that handles the Retrieval-Augmented Generation (RAG) process.
    This includes embedding, indexing, and querying documents.
    """
    
    def __init__(self, 
                 index_name: str, 
                 text_processor: TextProcessor, 
                 PINECONE_API_KEY: str, 
                 gpt_engine_name: str, 
                 embedding_model_name: str, 
                 api_key: str, 
                 azure_endpoint: Optional[str] = None, 
                 api_version: Optional[str] = None, 
                 openai_type: str = 'openai') -> None:
        """
        Initialize the RAG system with Pinecone and OpenAI configurations.
        
        Args:
            index_name (str): The name of the Pinecone index.
            text_processor (TextProcessor): An instance of the TextProcessor class.
            PINECONE_API_KEY (str): The API key for Pinecone.
            gpt_engine_name (str): The name of the GPT model to be used.
            embedding_model_name (str): The name of the embedding model to be used.
            api_key (str): The API key for OpenAI or Azure OpenAI.
            azure_endpoint (str, optional): The Azure endpoint, if using Azure OpenAI.
            api_version (str, optional): The API version for Azure OpenAI.
            openai_type (str): Specifies whether to use 'openai' or 'azure_openai'.
        """
        self.index_name = index_name
        self.docsearch = None
        self.answer = ''
        self.gpt_engine_name = gpt_engine_name
        self.doc_processing = text_processor
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        if openai_type == 'azure_openai':
            self.openai_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
            self.embeddings = AzureOpenAIEmbeddings(
                model=embedding_model_name, 
                api_key=api_key, 
                azure_endpoint=azure_endpoint, 
                openai_api_version=api_version
            )
        else:
            self.openai_client = OpenAI(api_key=api_key)
            self.embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=api_key)

        self.__initialize_index()

    def __initialize_index(self) -> None:
        """
        Initialize the Pinecone index if it does not already exist.
        """
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name = self.index_name,
                dimension = 1536,
                metric = "cosine",
                spec = ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        self.docsearch = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

    def insert_doc(self, file_content: str, file_name: str) -> str:
        """
        Process and insert a document into the Pinecone index.
        
        Args:
            file_path (str): The path to the PDF file.
            file_name (str): The name of the PDF file.
            
        Returns:
            str: A success message upon successful indexing.
        """
        try:
            docs = self.doc_processing.extract_text(file_content, file_name)
            self.docsearch.add_documents(docs)
            return 'Successfully indexed the document'
        except Exception as e:
            raise Exception(f"Error in processing and indexing the document: {e}")

    def _qna_helper(self, query: str, context: str) -> Any:
        """
        Helper function to generate a structured answer using the OpenAI GPT model.
        
        Args:
            query (str): The user's query.
            context (str): The context from which to generate the answer.
            
        Returns:
            Any: The generated response from the GPT model.
        """
        return self.openai_client.chat.completions.create(
            model = self.gpt_engine_name,
            messages = [
                {'role': 'system', 'content': QNA_PROMPT.format(context=context, date = time.strftime("%d/%m/%Y"))},  
                {'role': 'user', 'content': f"User Query: ```{query}```"}
            ],
            temperature = TEMPERATURE,
            stream=True
        )

    def qna(self, query: str) -> Generator[str, None, None]:
        """
        Perform a question-answer operation based on the provided query and filters.
        
        Args:
            query (str): The user's query.
            filters (Dict[str, Any]): A dictionary of filters to apply when searching the documents.
            
        Yields:
            Generator[str, None, None]: The generated answer, streamed line by line.
        """
        docs = self.docsearch.similarity_search(query, k=TOP_K)

        # Build context from the retrieved documents
        context = ''
        for doc in docs:
            file_name = doc.metadata['filename']
            text = doc.page_content
            context += f'Section Text({file_name}): {text}\n-----------------------\n\n'

        # Generate the answer using the context
        answer = self._qna_helper(query, context)
        final_answer = ''
        for chunk in answer:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                final_answer += text
                yield text
                time.sleep(0.02)

        self.answer = final_answer