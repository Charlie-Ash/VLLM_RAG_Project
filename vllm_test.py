'''
RUN THIS ON LINUX SYSTEM
'''

from vllm import LLM, SamplingParams

# Initialize the engine
llm = LLM(model="RedHatAI/gemma-4-26B-A4B-it-NVFP4",
          gpu_memory_utilization=0.9,  # reserve up to 90% of available VRAM for the KV cache and runtime buffers
          max_model_len=32768  # sets the maximum context window that vLLM will allocate KV cache for
          )  # Use this to install gemma4:26B quantized via Huggingface

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.65,  # temperture: randomness
                                 top_p=0.95,  # top_p; nucleus sampling
                                 max_tokens=512,  # Max tokens outputted
                                 repetition_penalty = 1.1  # Penalty to apply if tokens continue repeating.
                                 )   # top_p; nucleus sampling,

# Run inference
prompts = [{

    "role": "user",
    "content": "Hello, my name is Charlie. Can you introduce youself to me?"

}]
outputs = llm.chat(prompts, sampling_params)

# Print the response (there's only going to be 1 "output" here to loop through, since there's only 1 qustion request)
for output in outputs:
    print(output.outputs[0].text)

