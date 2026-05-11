from vllm import LLM, SamplingParams

# Initialize the engine
llm = LLM(model="RedHatAI/gemma-4-26B-A4B-it-NVFP4",
          gpu_memory_utilization=0.9,  # reserve up to 90% of available VRAM for the KV cache and runtime buffers
          max_model_len=32768  # sets the maximum context window that vLLM will allocate KV cache for
          )  # Use this to install gemma4:26B quantized via Huggingface

