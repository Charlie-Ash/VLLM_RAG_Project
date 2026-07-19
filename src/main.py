import torch
from vllm import LLM, SamplingParams
from userQuery import user_query
from dataIngestion import build_vector_index, load_vector_index
import qdrant_client
from config import COLLECTION_NAME, QDRANT_DB_PATH, LLM_MODEL, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN

# Reporting function if GPU runs out of memory during RAG
def check_gpu_memory(gpu_memory_utilization: float) -> None:
    """Fail fast with an actionable message instead of vLLM's bare ValueError.

    vLLM's own pre-flight check just reports free vs. required VRAM with no
    guidance on why memory might already be missing.
    """

    if not torch.cuda.is_available():
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024 ** 3)
    total_gib = total_bytes / (1024 ** 3)
    required_gib = total_gib * gpu_memory_utilization

    print(f"GPU memory: {free_gib:.1f} GiB free / {total_gib:.1f} GiB total "
          f"(vLLM will request ~{required_gib:.1f} GiB)")

    if free_gib < required_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free, but gpu_memory_utilization="
            f"{gpu_memory_utilization} needs ~{required_gib:.1f} GiB on a "
            f"{total_gib:.1f} GiB device. Likely causes: a stale vLLM worker "
            f"process from a previous crashed run still holding the CUDA "
            f"context (check `nvidia-smi` and kill leftover python "
            f"processes), another application using the GPU, or -- once "
            f"running alongside the A2A orchestrator -- its model already "
            f"being loaded. Lower RAG_GPU_MEMORY_UTILIZATION or free up VRAM "
            f"before retrying."
        )


def main():

    # Check if vector DB already exists
    try:
        db_client = qdrant_client.QdrantClient(
            path=QDRANT_DB_PATH
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open Qdrant DB at {QDRANT_DB_PATH}: {e}") from e

    if not db_client.collection_exists(COLLECTION_NAME):  # Build another DB

        print("No vector database found.")
        print("Building vector database...")

        # Build vector DB
        db_client.close()
        index, db_client = build_vector_index()

        print("Embedding complete.")

    else:

        print("Vector database already exist.")
        print("Loading vector database...")

        # Load vector DB
        db_client.close()
        index, db_client = load_vector_index()

        print("Loading completed.")

    # LLM
    print("Loading LLM...")
    check_gpu_memory(GPU_MEMORY_UTILIZATION)

    try:
        llm = LLM(
            model=LLM_MODEL,
            trust_remote_code=True,  # required for Gemma 4's custom architecture code
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load LLM '{LLM_MODEL}': {e}") from e

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.65,  # temperture: randomness
        top_p=0.95,  # top_p; nucleus sampling
        max_tokens=512,  # Max tokens outputted
        repetition_penalty = 1.1  # Penalty to apply if tokens continue repeating.
    )   # top_p; nucleus sampling,

    # Main logic loop
    while True:

        try:
            input_text = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if input_text.lower() == "bye":
            break

        if not input_text.strip():
            continue

        # Structured prompt, fed to llm.chat() so the model's own chat
        # template (turn-formatting special tokens) gets applied -- Gemma's
        # instruction-tuned checkpoints expect these; a flattened raw string
        # (as generate() would receive) degrades answer quality.
        sys_prompt = "You are a research assitant. Please use the relevant context below to answer the user's question. If the answer is unclear, reply the fact that you don't know."

        # Retrieval relevant chunks
        relevant_info_list = user_query(index, input_text)
        context_block = "\n\n".join(
            f"Relevant context {i}:\n{chunk}" for i, chunk in enumerate(relevant_info_list)
        )

        user_message = f"{sys_prompt}\n\n{context_block}\n\nUser Query:\n{input_text}"
        messages = [{"role": "user", "content": user_message}]

        # Feed structured prompt to LLM
        try:
            outputs = llm.chat(messages, sampling_params)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue

        print("\n\nANSWER: \n\n")
        print(outputs[0].outputs[0].text)


# This part acts as a safeguard to block the vLLM worker process from spawning
# With the worker process, Python re-imports the entire script inside a new child processes
if __name__ == "__main__":
    main()
