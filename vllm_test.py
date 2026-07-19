'''
RUN THIS ON LINUX SYSTEM
'''

from vllm import LLM, SamplingParams

# Initialize the engine
# Same model as src/main.py (google/gemma-4-E2B-it-qat-w4a16-ct): official Google
# QAT-quantized checkpoint, ~7.3 GiB VRAM, small enough to coexist with the A2A
# orchestrator's E4B-it model on one GPU.
llm = LLM(model="google/gemma-4-E2B-it-qat-w4a16-ct",
          trust_remote_code=True,  # required for Gemma 4's custom architecture code
          gpu_memory_utilization=0.5,
          max_model_len=4096
          )

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

