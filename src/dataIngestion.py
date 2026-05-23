from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import qdrant_client
import pymupdf

from pathlib import Path

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
        chunk_size=256,
        chunk_overlap=32
    )
    chunks = splitter.get_nodes_from_documents(documents=documents)  # All PDF file content's chunks

    # Embed chunks
    embedding_model = HuggingFaceEmbedding(
        model_name= "nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True
    )  # From nvidia, 8b embedding model
    
    # Setup Qdrant
    db_client = qdrant_client.QdrantClient(path="./qdrant_db")  # DB client

    if db_client.collection_exists("my_docs"):  # Delete duplications of vector stores
        db_client.delete_collection("my_docs")

    vector_store = QdrantVectorStore(
        client= db_client,
        collection_name= "my_docs"
    )

    storage_context = StorageContext.from_defaults(
        vector_store= vector_store
    )

    # Build persisting indexes (will re-embed if ran again, unless we load it)
    index = VectorStoreIndex(  # 和潤向量索引元件
        chunks,
        storage_context= storage_context,
        embed_model= embedding_model
    )

    # Return Index    
    return index, db_client  # REMEMBER TO DELETE: db_client's returning was only for dataIngestion.py's testing


# Test portal 
if __name__ == "__main__":

    index, db_client = build_vector_index()
    print(db_client.get_collection(collection_name= "my_docs"))
    db_client.close()  # To close the DB, fixing import error


