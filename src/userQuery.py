from dataIngestion import build_PDF_vector_index # del
from llama_index.core import VectorStoreIndex
import typing


def user_query(index: VectorStoreIndex, input_string: str) -> list[str]:

    # Retriever
    retriever = index.as_retriever(similarity_top_k= 3)  # Top 3 similar chunks to return

    # Results
    # "retriever" is already connected to "QdrantVectorStore". This'll automatically embed the input, go to the DB and find relevant chunks
    results_node = retriever.retrieve(input_string)  

    # Tranfer relevant text into a list
    # Note: returning chunk size is already determined as 512 during ingestion
    results_text_list = []
    i = 1
    for node in results_node:

        # Test print
        print(f"node {i} score: {node.score}")
        print(node.text)
        i = i + 1

        # Add text into "results_text_list"
        results_text_list.append(node.text)

    return results_text_list


# Test portal 
if __name__ == "__main__":

    index, db_client = build_PDF_vector_index()
    results_text_list = user_query(index, "What was Pete's favorite subject when he was young?")
    print(f"RESULTS: {results_text_list}")

