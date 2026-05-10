from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import qdrant_client

# Load documents
reader = SimpleDirectoryReader(
    input_dir="C:\\Users\\charl\\Code\\Python\\vLLMLearning\\data", # Path to the directory, where the contents within will be read
    recursive=True
)
documents = reader.load_data(show_progress=True)

# Chunk documents
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64
)
chunks = splitter.get_nodes_from_documents(documents=documents)  # All PDF file content's chunks

# Embed chunks
embedding_model = HuggingFaceEmbedding(model_name= "nvidia/llama-embed-nemotron-8b")  # From nvidia, 8b embedding model
# Setup Qdrant
db_client = qdrant_client.QdrantClient(path="./qdrant_db")  # DB client

vector_store = QdrantVectorStore(
    client= db_client,
    collection_name= "my_docs"
)

storage_context = StorageContext.from_defaults(
    vector_store= vector_store
)

# Build persisting indexes (will re-embed if ran again, unless we load it)
index = VectorStoreIndex(
    chunks,
    storage_context= storage_context,
    embed_model= embedding_model
)

# Test
print(db_client.get_collection(collection_name= "my_docs"))



