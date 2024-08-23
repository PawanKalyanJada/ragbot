# Configuration constants for the application

# Text chunking settings
CHUNK_SIZE = 500  # The size of each text chunk
CHUNK_OVERLAP = 50  # The overlap between consecutive text chunks

# Query settings
TOP_K = 3  # Number of top documents to retrieve for a query
TEMPERATURE = 0.1  # Temperature setting for the language model to control randomness

# Pinecone API settings
PINECONE_API_KEY = "acfbcc99-b8e3-41f3-b7ca-7e153d9bd6e3"  # API key for accessing Pinecone
PINECONE_INDEX_NAME = "rag-index"  # Name of the Pinecone index to use