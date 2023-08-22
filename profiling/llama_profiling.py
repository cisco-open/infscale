import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from profile_in_json import get_model_inference_profile

with get_accelerator().device(0):
    tokenizer = LlamaTokenizer.from_pretrained('openlm-research/open_llama_13b')
    model = LlamaForCausalLM.from_pretrained('openlm-research/open_llama_13b', torch_dtype=torch.float16, device_map='auto')
    
    prompt = 'Q: What is the largest animal?\nA:'
    input = dict(tokenizer(prompt, return_tensors="pt"))
    print(input)
    flops, macs, params = get_model_inference_profile(
        model,
        kwargs=input,
        print_profile=True,
        detailed=True,
        mode="generate",
        output_file="./llama_profile.json"
    )