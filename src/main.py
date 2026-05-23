from vllm import LLM, SamplingParams
from userQuery import user_query
from dataIngestion import build_vector_index

def main():

    # Vector index built here
    index, db_client = build_vector_index()

    # LLM
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

    # Structured prompt (fed to vllm.chat()), OpenAI format
    sys_prompt = "You are a research assitant. Please use the relavent context passed in from the role 'system' to answer the users question. If the answer is unclear, reply the fact that you don't know."
    prompts = [{

        "role": "system",
        "content": sys_prompt

    }, {"placeholder": "placeholder"}, {"placeholder": "placeholder"}, {"placeholder": "placeholder"}, {"placeholder": "placeholder"}]

    # Main logic loop
    input_text = input(">>> ")
    while input_text.lower() != "bye":

        # User's question prompt
        user_prompt = {

            "role": "user",
            "content": input_text

        }
        prompts[1] = user_prompt

        # Retrieval relavent chunks
        relavent_info_list = user_query(index, input_text)
        for i in range(3):

            info_prompt = {

                "role": "system",
                "content": f"Relevant context {i}:\n{relavent_info_list[i]}"

            }
            prompts[i + 2] = info_prompt


        # Feed structured prompt to LLM
        outputs = llm.chat(prompts, sampling_params)

        # Print out outputs
        for output in outputs:
            print(output.outputs[0].text)

        input_text = input(">>> ")


# This part acts as a safeguard to block the vLLM worker process from spawning
# With the worker process, Python re-imports the entire script inside a new child processes
if __name__ == "__main__":
    main()
