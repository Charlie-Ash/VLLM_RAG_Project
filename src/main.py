from vllm import LLM, SamplingParams
from userQuery import user_query
from dataIngestion import build_vector_index, load_vector_index
import qdrant_client

COLLECTION_NAME = "my_docs"


def main():

    # Check if vector DB already exists
    db_client = qdrant_client.QdrantClient(
        path="./qdrant_db"
    )

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
    llm = LLM(
        model="RedHatAI/gemma-4-26B-A4B-it-NVFP4",
        gpu_memory_utilization=0.9,  # reserve up to 90% of available VRAM for the KV cache and runtime buffers
        max_model_len=32768  # sets the maximum context window that vLLM will allocate KV cache for
    )  # Use this to install gemma4:26B quantized via Huggingface

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.65,  # temperture: randomness
        top_p=0.95,  # top_p; nucleus sampling
        max_tokens=512,  # Max tokens outputted
        repetition_penalty = 1.1  # Penalty to apply if tokens continue repeating.
    )   # top_p; nucleus sampling,

    # Main logic loop
    while True:

        input_text = input(">>> ")
        if input_text.lower() == "bye":
            break

        # Structured prompt (fed to vllm.chat()), OpenAI format
        sys_prompt = "You are a research assitant. Please use the relevant context passed in from the role 'system' to answer the users question. If the answer is unclear, reply the fact that you don't know."
        prompts = [{

            "role": "system",
            "content": sys_prompt

        }]

        # User's question prompt
        user_prompt = {

            "role": "user",
            "content": input_text

        }
        prompts.append(user_prompt)

        # Retrieval relevant chunks
        relevant_info_list = user_query(index, input_text)
        for i in range(len(relevant_info_list)):

            info_prompt = {

                "role": "system",
                "content": f"Relevant context {i}:\n{relevant_info_list[i]}"

            }
            prompts.append(info_prompt)


        # Feed structured prompt to LLM
        outputs = llm.chat(prompts, sampling_params)

        # Print out outputs
        for output in outputs:
            print(output.outputs[0].text)


# This part acts as a safeguard to block the vLLM worker process from spawning
# With the worker process, Python re-imports the entire script inside a new child processes
if __name__ == "__main__":
    main()
