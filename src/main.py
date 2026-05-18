from vllm import LLM, SamplingParams

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
prompts = [{

    "role": "system",
    "content": "Hello, my name is Charlie. Can you introduce youself to me?"

}, {"placeholder": "placeholder"}]

# Main logic loop
input_text = input(">>> ")
while input_text.lower() != "bye":

    user_prompt = {

        "role": "user",
        "content": input_text

    }
    prompts[1] = user_prompt

    # Feed structured prompt to LLM
    outputs = llm.chat(prompts, sampling_params)

    # Print out outputs
    for output in outputs:
        print(output.outputs[0].text)

    input_text = input(">>> ")
