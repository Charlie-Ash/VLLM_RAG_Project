# Llama Index for chunking and linking up with other parts
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
# Qdrant DB client
import qdrant_client
# PDF parser
import pymupdf
# Path resolving
from pathlib import Path
# Pytorch to empty CUDA cache (after embedding)
import torch
# Garbage collector
import gc

# Collection name
COLLECTION_NAME = "my_docs"


def load_documents() -> list[Document]:  # Probably add a path argument here in the future for better modification

    documents = []

    # Current file directory (src/)
    CURRENT_DIR = Path(__file__).resolve().parent

    # Move up one level, then enter data/
    DATA_DIR = CURRENT_DIR.parent / "data"

    # Find all PDF files recursively
    pdf_files = DATA_DIR.rglob("*.pdf")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")
        pdf = pymupdf.open(pdf_file)
        full_text = ""

        for page in pdf:
            full_text += page.get_text()

        pdf.close()

        # Create LlamaIndex Document object
        documents.append(

            Document(text= full_text, metadata= {"file_name": pdf_file.name})

        )

    return documents


def build_vector_index() -> tuple[VectorStoreIndex, qdrant_client.QdrantClient]:

    # Document loading
    documents = load_documents()

    # Chunk documents
    splitter = SentenceSplitter(  # 和潤切片策略元件
        chunk_size=128,
        chunk_overlap=16
    )
    chunks = splitter.get_nodes_from_documents(documents=documents)  # All PDF file content's chunks

    # Embed chunks
    embedding_model = HuggingFaceEmbedding(
        model_name= "nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True,
        device="cpu"  # Embedding to run on CPU since it interferes with LLM in the GPU (resource-wise)
    )  # From nvidia, 8b embedding model
    
    # Setup Qdrant
    db_client = qdrant_client.QdrantClient(path="./qdrant_db")  # DB client

    if db_client.collection_exists(COLLECTION_NAME):  # Delete duplications of vector stores
        db_client.delete_collection(COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client= db_client,
        collection_name= COLLECTION_NAME
    )

    storage_context = StorageContext.from_defaults(
        vector_store= vector_store
    )

    # Build persisting indexes (will re-embed if ran again)
    index = VectorStoreIndex(  # 和潤向量索引元件
        chunks,
        storage_context= storage_context,
        embed_model= embedding_model
    )

    # Clean-up (Delete embedding model & empty CUDA cache)
    del embedding_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return Index    
    return index, db_client  # REMEMBER TO DELETE: db_client's returning was only for dataIngestion.py's testing


def load_vector_index() -> tuple[VectorStoreIndex, qdrant_client.QdrantClient]:  # To load the index if the DB is already created 

    db_client = qdrant_client.QdrantClient(path="./qdrant_db")  # DB client

    vector_store = QdrantVectorStore(
        client= db_client,
        collection_name= COLLECTION_NAME
    )

    storage_context = StorageContext.from_defaults(
        vector_store= vector_store
    )

    # Embedding model is stioll needed when loading existing DB. Still used in user query.
    embedding_model = HuggingFaceEmbedding(
        model_name= "nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True
    )  # From nvidia, 8b embedding model

    # Persisting indexes
    index = VectorStoreIndex.from_vector_store(  # 和潤向量索引元件
        vector_store= vector_store,
        storage_context= storage_context,
        embed_model= embedding_model,
        device="cpu"  # Embedding to run on CPU since it interferes with LLM in the GPU (resource-wise)
    )

    # Clean-up (Delete embedding model & empty CUDA cache)
    del embedding_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return Index    
    return index, db_client



# Test portal 
if __name__ == "__main__":

    index, db_client = build_vector_index()
    print(db_client.get_collection(collection_name= COLLECTION_NAME))
    db_client.close()  # To close the DB, fixing import error


