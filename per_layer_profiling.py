import os
import json
import time
import numpy as np
import copy
from PIL import Image

import torch
from torch import nn

from transformers import (
    AutoModelForImageClassification,
    PretrainedConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
    T5ForConditionalGeneration,
)
from accelerate.utils.modeling import set_module_tensor_to_device

from infscale.module.model_metadata import ResnetModelMetaData, Llama3ModelMetaData, BertModelMetaData, T5ModelMetaData
from infscale.module.modelir import ModelIR

import argparse

from typing import Dict, List, Optional, Set, Tuple, Union

import gc

# ======== Color Printing ========

def print_color(text, color='yellow', style='normal'):
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
    }
    styles = {
        'normal': '\033[0m',
        'bold': '\033[1m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blink': '\033[5m',
        'reverse': '\033[7m',
    }
    reset = '\033[0m'
    print(f"{styles.get(style, '')}{colors.get(color, colors['yellow'])}{text}{reset}")

# ======== Utility Functions ========

def clear_device_cache(garbage_collection=False):
    """
    Clears the device cache by calling `torch.{backend}.empty_cache`. Can also run `gc.collect()`, but do note that
    this is a *considerable* slowdown and should be used sparingly.
    """
    if garbage_collection:
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_device_same(first_device, second_device):
    """
    Utility method to check if two `torch` devices are similar. When dealing with CUDA devices, torch throws `False`
    for `torch.device("cuda") == torch.device("cuda:0")` whereas they should be the same
    """
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        # In case the first_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        # In case the second_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        second_device = torch.device("cuda", index=0)

    return first_device == second_device

# ======== Memory Profiling Utilities ========

def calculate_tensor_size(tensor):
    """Calculate memory used by a tensor in bytes"""
    return tensor.element_size() * tensor.nelement()

def get_model_static_memory(model):
    """Calculate static memory used by model parameters and buffers in MB"""
    param_size = sum(calculate_tensor_size(p) for p in model.parameters())
    buffer_size = sum(calculate_tensor_size(b) for b in model.buffers())
    return (param_size + buffer_size)

def measure_actual_gpu_memory(models):
    """Measure actual GPU memory consumed when loading models to GPU"""
    # Make sure nothing is on GPU initially
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    # Move all models to GPU and measure
    for model in models:
        model.to("cuda")
    
    # Measure after loading
    end_mem = torch.cuda.memory_allocated()
    
    # Return actual memory usage in MB
    return (end_mem - start_mem)

# ======== Create inputs for profiling ========

def create_inputs(model_type, model, device, batch_size=1, seq_length=None):
    if model_type.startswith("llama"):
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        assert seq_length is not None
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length)).to(device)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)
        past_key_values = DynamicCache()
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values, 'use_cache': True}
    elif model_type == "resnet":
        input_shape = (batch_size, 3, 224, 224)
        # input_shape = (batch_size, 3, 32, 32)   # for CIFAR-10
        inputs = {"pixel_values": torch.randn(input_shape).to(device)}
    elif model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        assert seq_length is not None
        assert seq_length <= 512, "BERT only supports sequence lengths up to 512"
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length)).to(device)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)
        token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    elif model_type == "t5":
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        assert seq_length is not None
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length)).to(device)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)
        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]] * batch_size).to(device)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids}
    else:
        print(f"Warning: Unknown model type '{model_type}', defaulting to ResNet-style inputs")
        # Default to ResNet-style inputs if model_type is unknown
        input_shape = (batch_size, 3, 224, 224)
        inputs = {"pixel_values": torch.randn(input_shape).to(device)}
    
    return inputs

# ======== Command Line Arguments ========

def parse_args():
    parser = argparse.ArgumentParser(description="Profile a PyTorch model with memory and time tracking.")
    parser.add_argument("--profile_reptitions", type=int, default=20, help="Number of repetitions to profile each module.")
    parser.add_argument("--model_type", type=str, help="Model type to profile.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", help="List of batch sizes to profile")
    parser.add_argument("--seq_lengths", type=int, nargs="+", help="List of sequence lengths to profile (only applicable for LLMs)")
    parser.add_argument("--profile_decode", action="store_true", help="Profile the decoding phase instead of prefilling")
    parser.add_argument("--output_suffix", type=str, help="Optional text to append to the output folder name")
    parser.add_argument("--trimmed_model_dir", type=str, default=None, help="Path to trimmed LLaMA model directory; if set, load model from here")
    
    args = parser.parse_args()
    return args

# ======== Main Function ========

LM_MODELS = ["llama", "llama_70b", "bert", "t5"]
MODEL_WITH_KV_CACHE = ["llama", "llama_70b"]

if __name__ == '__main__':
    # ======== Initialization ========
    args = parse_args()
    print_color(args)

    profile_reptitions = args.profile_reptitions
    model_type = args.model_type
    batch_sizes = args.batch_sizes
    seq_lengths = args.seq_lengths if args.seq_lengths is not None else [512]  # Default to 1k if not specified
    profile_decode = args.profile_decode
    output_suffix = args.output_suffix
    trimmed_original_layers = None

    # Use the GPU for inference
    gpu_device = "cuda"
    cpu_device = "cpu"

    # Load the pre-trained model
    if model_type == "resnet":
        model_name = "microsoft/resnet-152"

        config = AutoModelForImageClassification.from_pretrained(model_name).config
        full_model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Create model metadata and IR
        model_metadata = ResnetModelMetaData(model_name, config)
        model_metadata.trace_inputs = ["pixel_values"]
        model_ir = ModelIR(model_metadata)
        
        # Image models don't use sequence length
        seq_lengths = [None]
    elif model_type.startswith("llama"):
        if args.trimmed_model_dir is not None:
            model_name = args.trimmed_model_dir
        else:
            if model_type == "llama":
                model_name = "meta-llama/Meta-Llama-3.1-8B"
            elif model_type == "llama_70b":
                model_name = "meta-llama/Meta-Llama-3.1-70B"
            else:
                raise ValueError(f"Model type {model_type} not supported")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        full_model = AutoModelForCausalLM.from_pretrained(model_name)
        config = full_model.config

        # If using a trimmed model directory, try reading metadata to restore original layer count
        if args.trimmed_model_dir is not None:
            meta_path = os.path.join(model_name, "trim_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if "original_num_hidden_layers" in meta:
                        trimmed_original_layers = int(meta["original_num_hidden_layers"])
                        print_color(f"Detected trimmed model. original_num_hidden_layers={trimmed_original_layers}", color='green')
                except Exception as e:
                    print_color(f"Failed to read trim metadata: {e}", color='red')

        # Create model metadata and IR
        model_metadata = Llama3ModelMetaData(model_name, config)
        model_metadata.trace_inputs = ["input_ids", "attention_mask"]
        model_ir = ModelIR(model_metadata)
    elif model_type == "bert":
        model_name = "bert-large-uncased"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoModelForCausalLM.from_pretrained(model_name).config
        config.use_cache = False
        full_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Create model metadata and IR
        model_metadata = BertModelMetaData(model_name, config)
        model_metadata.trace_inputs = ["input_ids", "token_type_ids", "attention_mask"]
        model_ir = ModelIR(model_metadata)
    elif model_type == "t5":
        model_name = "t5-large"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = T5ForConditionalGeneration.from_pretrained(model_name).config
        config.use_cache = False
        full_model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Create model metadata and IR
        model_metadata = T5ModelMetaData(model_name, config)
        model_metadata.trace_inputs = ["input_ids", "attention_mask", "decoder_input_ids"]
        model_ir = ModelIR(model_metadata)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    # Get sharded layers
    layers = model_ir.layers
    print(f"Number of sharded layers: {len(layers)}")

    # Initialize all layers on the device
    print("Initializing layers on device...")
    layer_modules = []
    for i, layer in enumerate(layers):
        print(f"Initializing layer {i}")
        # Create empty layer on cpu
        layer = layer.to_empty(device=cpu_device)
            
        # Copy parameters from full model using device transfer helper
        for name, param in layer.named_parameters():
            full_param = full_model.get_parameter(name)
            set_module_tensor_to_device(
                layer,
                name,
                cpu_device,
                value=full_param.data,
                dtype=full_param.dtype
            )
        
        # Copy buffers similarly
        for name, buf in layer.named_buffers():
            full_buf = full_model.get_buffer(name)
            set_module_tensor_to_device(
                layer,
                name,
                cpu_device,
                value=full_buf.data,
                dtype=full_buf.dtype
            )
        
        layer_modules.append(layer)
    
    # Process each batch size and sequence length combination
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            if model_type not in LM_MODELS and seq_length is not None:
                raise ValueError("Sequence length is not supported for non-LLM models")
                
            print(f"\n{'='*50}")
            if seq_length is not None:
                print(f"PROFILING BATCH SIZE: {batch_size}, SEQUENCE LENGTH: {seq_length}")
            else:
                print(f"PROFILING BATCH SIZE: {batch_size}")
            print(f"{'='*50}\n")
            
            # Create inputs for this batch size and sequence length
            inputs = create_inputs(model_type, full_model, gpu_device, batch_size=batch_size, seq_length=seq_length)
            
            # ======== Static Memory Profiling ========
            static_memory_usage = []
            theoretical_static_memory = []
            print("\nCalculating static memory usage per layer...")
            for i, layer in enumerate(layer_modules):
                # Measure actual static memory when moving to GPU
                print(f"  Measuring actual GPU memory for layer {i}...")
                clear_device_cache(garbage_collection=True)
                actual_static_memory = measure_actual_gpu_memory([layer])
                static_memory_usage.append(actual_static_memory)
                theoretical_static_memory.append(get_model_static_memory(layer))
                print(f"Layer {i} static memory usage: {actual_static_memory:.2f} bytes (theoretical: {theoretical_static_memory[i]:.2f} bytes)")
                
                # Move layer back to CPU to free GPU memory
                layer.to(cpu_device)
                clear_device_cache()

            if 'past_key_values' in inputs:
                print("Past key values found in inputs. Adding kv cache to profiling...")

                kv_cache_usage = []

                with torch.inference_mode():
                    curr_input = inputs.copy()

                    for i, layer in enumerate(layer_modules):
                        assert "cpu" in str(next(layer.parameters()).device)
                        layer.to(gpu_device)

                        dynamic_cache = curr_input['past_key_values']
                        key_cache_size = sum(x.nelement() * x.element_size() for x in dynamic_cache.key_cache)
                        value_cache_size = sum(x.nelement() * x.element_size() for x in dynamic_cache.value_cache)

                        before_kv_cache_size = key_cache_size + value_cache_size
                        
                        output = layer(**curr_input)
                        
                        key_cache_size = sum(x.nelement() * x.element_size() for x in dynamic_cache.key_cache)
                        value_cache_size = sum(x.nelement() * x.element_size() for x in dynamic_cache.value_cache)

                        after_kv_cache_size = key_cache_size + value_cache_size
                        print(f"KV cache size difference for layer {i}: {(after_kv_cache_size - before_kv_cache_size)/1e9} GB")

                        kv_cache_usage.append(after_kv_cache_size - before_kv_cache_size)    # We treat the kv cache as static memory

                        curr_input = output

                        layer.to(cpu_device)
            
            # ======== Dynamic Memory Profiling ========
            dynamic_memory_usage = [[] for _ in range(len(layers))]
            input_sizes = [0] * len(layers)
            output_sizes = [0] * len(layers)
            print(f"\nRunning {profile_reptitions} repetitions for memory profiling...")

            if profile_decode and model_type in MODEL_WITH_KV_CACHE:
                # ======== Generate first token to get KV cache for decode profiling ========
                print("\nGenerating first token to create KV cache for decode profiling...")

                inputs = create_inputs(model_type, full_model, cpu_device, batch_size=batch_size, seq_length=seq_length)

                assert 'cpu' in str(next(full_model.parameters()).device)
                
                # Run all layers to generate the first token and create KV cache
                with torch.inference_mode():
                    outputs = full_model(**inputs)
                
                # Extract next token using greedy decoding
                if isinstance(output, dict) and 'logits' in output:
                    next_token_logits = output['logits'][:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    raise ValueError("Could not extract logits from model output")
                
                # Create decode inputs - only the new token as input_ids
                decode_inputs = {
                    'input_ids': next_token.to(gpu_device),
                    'attention_mask': torch.cat([inputs['attention_mask'].to(gpu_device), torch.ones_like(next_token).to(gpu_device)], dim=-1),
                    'use_cache': True,
                    'past_key_values': outputs['past_key_values']
                }
                
                dynamic_cache = decode_inputs['past_key_values']

                for i in range(len(dynamic_cache.key_cache)):
                    dynamic_cache.key_cache[i] = dynamic_cache.key_cache[i].to(gpu_device)
                for i in range(len(dynamic_cache.value_cache)):
                    dynamic_cache.value_cache[i] = dynamic_cache.value_cache[i].to(gpu_device)

                # Use these prepared inputs for decoding profiling
                inputs = decode_inputs
            else:
                inputs = create_inputs(model_type, full_model, gpu_device, batch_size=batch_size, seq_length=seq_length)

            with torch.inference_mode():
                curr_input = inputs.copy()
                for i, layer in enumerate(layer_modules):
                    assert "cpu" in str(next(layer.parameters()).device)
                    layer.to(gpu_device)

                    # Clear cache before profiling
                    clear_device_cache(garbage_collection=True)
                    
                    # Capture input size for this layer
                    input_sizes[i] = sum(x.nelement() * x.element_size() for x in curr_input.values() if isinstance(x, torch.Tensor))

                    print(f"Input size for layer {i}: {input_sizes[i]}")

                    for key, value in curr_input.items():
                        if not isinstance(value, torch.Tensor):
                            print_color(f"Non-tensor input: {key}: {value}")

                    print(f"  Running {1} repetitions for layer {i}...")
                    for rep in range(1):
                        # Reset peak memory stats
                        torch.cuda.reset_peak_memory_stats()
                        mem_before = torch.cuda.memory_allocated()
                        
                        # Run layer
                        output = layer(**curr_input)
                        
                        # Calculate dynamic memory
                        mem_after_peak = torch.cuda.max_memory_allocated()
                        dynamic_mem = (mem_after_peak - mem_before)
                        dynamic_memory_usage[i].append(dynamic_mem)
                        
                        # Clear cache between repetitions
                        clear_device_cache()

                    # Capture output size for this layer
                    if isinstance(output, dict):
                        output_sizes[i] = sum(x.nelement() * x.element_size() for x in output.values() if isinstance(x, torch.Tensor))
                    else:
                        output_sizes[i] = output.nelement() * output.element_size()

                    print(f"Output size for layer {i}: {output_sizes[i]}")
                    
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if not isinstance(value, torch.Tensor):
                                print_color(f"Non-tensor output: {key}: {value}")
                    else:
                        if not isinstance(output, torch.Tensor):
                            print_color(f"Non-tensor output: {output}")
                    
                    curr_input = output

                    layer.to(cpu_device)
            
            # ======== Time Profiling ========
            print(f"\nRunning {profile_reptitions} repetitions for time profiling...")
            
            # Pre-allocate CUDA events for timing
            print("Pre-allocating CUDA events...")
            start_events = [[torch.cuda.Event(enable_timing=True) for _ in range(profile_reptitions)] for _ in range(len(layers))]
            end_events = [[torch.cuda.Event(enable_timing=True) for _ in range(profile_reptitions)] for _ in range(len(layers))]
            
            # Create inputs again
            clear_device_cache(garbage_collection=True)

            if profile_decode and model_type in MODEL_WITH_KV_CACHE:
                # ======== Generate first token to get KV cache for decode profiling ========
                print("\nGenerating first token to create KV cache for decode profiling...")

                inputs = create_inputs(model_type, full_model, cpu_device, batch_size=batch_size, seq_length=seq_length)
                
                # Run all layers to generate the first token and create KV cache
                with torch.inference_mode():
                    outputs = full_model(**inputs)
                
                # Extract next token using greedy decoding
                if isinstance(output, dict) and 'logits' in output:
                    next_token_logits = output['logits'][:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    raise ValueError("Could not extract logits from model output")
                
                # Create decode inputs - only the new token as input_ids
                decode_inputs = {
                    'input_ids': next_token.to(gpu_device),
                    'attention_mask': torch.cat([inputs['attention_mask'].to(gpu_device), torch.ones_like(next_token).to(gpu_device)], dim=-1),
                    'use_cache': True,
                    'past_key_values': outputs['past_key_values']
                }
                
                dynamic_cache = decode_inputs['past_key_values']

                for i in range(len(dynamic_cache.key_cache)):
                    dynamic_cache.key_cache[i] = dynamic_cache.key_cache[i].to(gpu_device)
                for i in range(len(dynamic_cache.value_cache)):
                    dynamic_cache.value_cache[i] = dynamic_cache.value_cache[i].to(gpu_device)

                # Use these prepared inputs for decoding profiling
                inputs_for_timing = decode_inputs
            else:
                inputs_for_timing = create_inputs(model_type, full_model, gpu_device, batch_size=batch_size, seq_length=seq_length)

            with torch.inference_mode():
                curr_input = inputs_for_timing
                for i, layer in enumerate(layer_modules):
                    print(f"Profiling layer {i}...")
                    assert "cpu" in str(next(layer.parameters()).device)
                    layer.to(gpu_device)

                    if 'past_key_values' in curr_input:
                        dynamic_cache = copy.deepcopy(curr_input['past_key_values'])

                    for rep in range(profile_reptitions):
                        if 'past_key_values' in curr_input:
                            curr_input['past_key_values'] = copy.deepcopy(dynamic_cache)

                        # Record start event
                        start_events[i][rep].record()
                        
                        # Run layer normally
                        output = layer(**curr_input)
                        
                        # Record end event
                        end_events[i][rep].record()

                    curr_input = output

                    layer.to(cpu_device)
            
            # Ensure all operations are complete before calculating statistics
            torch.cuda.synchronize()
            
            # ======== Results Processing ========
            print("\nProcessing profiling results...")
            
            # Storage for processed timing results
            timing_results = []
            
            for i in range(len(layers)):
                # Calculate latencies for this layer
                latencies = []
                for rep in range(profile_reptitions):
                    latency = start_events[i][rep].elapsed_time(end_events[i][rep])
                    latencies.append(latency)
                
                # Remove outliers (2 highest and 2 lowest)
                latencies = np.sort(latencies)[2:-2]
                timing_results.append(latencies)
                
                # Calculate latency statistics
                print(f"\nLayer {i} stats:")
                print(f"  Latency: avg = {np.mean(latencies):.2f} ms, min = {np.min(latencies):.2f} ms, max = {np.max(latencies):.2f} ms, std dev = {np.std(latencies):.2f} ms, [{latencies}]")
                
                # Calculate parameter count
                num_params = sum(p.numel() for p in layer_modules[i].parameters())
                print(f"  Parameters: {num_params:,}")
                
                # Print memory statistics if enabled
                print(f"  Static memory (weights): {static_memory_usage[i]:.2f} bytes")
                if dynamic_memory_usage[i]:
                    # Remove outliers (2 highest and 2 lowest if we have enough measurements)
                    if len(dynamic_memory_usage[i]) > 4:
                        dynamic_mems = np.sort(dynamic_memory_usage[i])[2:-2]
                    else:
                        dynamic_mems = dynamic_memory_usage[i]
                    print(f"  Dynamic memory (activations): avg = {np.mean(dynamic_mems):.2f} bytes, max = {np.max(dynamic_mems):.2f} bytes")
            
            # ======== Data Persistence ========
            print("\nSaving profiling data to file...")
            if profile_decode and model_type in MODEL_WITH_KV_CACHE:
                profile_type = "decode"
            elif model_type in MODEL_WITH_KV_CACHE:
                profile_type = "prefill"
            else:
                profile_type = "n/a"

            profiling_data = {
                "model_name": model_name,
                "batch_size": batch_size,
                "profile_type": profile_type,
                "layers": []
            }
            
            computed_layers = []
            for i in range(len(layers)):
                # Calculate dynamic memory statistics
                if len(dynamic_memory_usage[i]) > 4:
                    dynamic_mems = np.sort(dynamic_memory_usage[i])[2:-2]
                else:
                    dynamic_mems = dynamic_memory_usage[i]
                # Use processed timing results
                latencies = timing_results[i]
                layer_data = {
                    "layer_num": i,
                    "layer_name": layer_modules[i].__class__.__name__,
                    "forward_latency_ms": float(np.mean(latencies)),
                    "static_memory_bytes": int(static_memory_usage[i]),
                    "theoretical_static_memory_bytes": int(theoretical_static_memory[i]),
                    "dynamic_memory_bytes": int(np.mean(dynamic_mems)),
                    "input_size_bytes": int(input_sizes[i]),
                    "output_size_bytes": int(output_sizes[i])
                }
                if model_type in MODEL_WITH_KV_CACHE:
                    layer_data["kv_cache_usage_bytes"] = int(kv_cache_usage[i])
                computed_layers.append(layer_data)

            # If profiling decode on a trimmed LLaMA model, duplicate decoder block and keep aux layers (norm, lm_head)
            if (
                model_type.startswith("llama")
                and trimmed_original_layers is not None
            ):
                # In a LLaMA architecture, decoder blocks are first in order, followed by model.norm and lm_head.
                num_trimmed_blocks = 1

                # Base per-layer stats come from the first decoder block of the trimmed model.
                # Keep layer 0 as-is (embedding), duplicate decoder block for layers 1..trimmed_original_layers
                # computed_layers[0] may correspond to embedding stage; decoder block starts at index 1.
                base_idx = 1
                base_block = computed_layers[base_idx]

                # Preserve layer 0
                layer0 = computed_layers[0].copy()
                layer0["layer_num"] = 0

                # Duplicate decoder blocks to restore original count, numbered 1..trimmed_original_layers
                duplicated_blocks = []
                for j in range(1, trimmed_original_layers + 1):
                    dup = base_block.copy()
                    dup["layer_num"] = j
                    duplicated_blocks.append(dup)

                # Append auxiliary layers (e.g., norm, lm_head) with re-indexed layer_num after decoder blocks
                aux_start = base_idx + num_trimmed_blocks
                aux_layers = computed_layers[aux_start:]
                reindexed_aux = []
                for idx, aux in enumerate(aux_layers):
                    aux_copy = aux.copy()
                    aux_copy["layer_num"] = trimmed_original_layers + 1 + idx
                    reindexed_aux.append(aux_copy)

                profiling_data["layers"] = [layer0] + duplicated_blocks + reindexed_aux
            else:
                profiling_data["layers"] = computed_layers
                
            # Add sequence length to profiling data for LLMs
            if model_type in LM_MODELS and seq_length is not None:
                profiling_data["sequence_length"] = seq_length
            
            # Save to file
            if profile_decode and model_type.startswith("llama"):
                if model_type == "llama_70b":
                    output_dir = f"profile_data/{full_model.__class__.__name__}_70b_decode"
                else:
                    output_dir = f"profile_data/{full_model.__class__.__name__}_decode"
            elif model_type.startswith("llama"):
                if model_type == "llama_70b":
                    output_dir = f"profile_data/{full_model.__class__.__name__}_70b_prefill"
                else:
                    output_dir = f"profile_data/{full_model.__class__.__name__}_prefill"
            else:
                output_dir = f"profile_data/{full_model.__class__.__name__}"
            
            # Append the output suffix if provided
            if output_suffix:
                output_dir = f"{output_dir}_{output_suffix}"
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Include sequence length in the filename for LLMs
            if model_type in LM_MODELS and seq_length is not None:
                output_file = os.path.join(output_dir, f"batch_size_{batch_size}_seq_length_{seq_length}.json")
            else:
                output_file = os.path.join(output_dir, f"batch_size_{batch_size}.json")
            
            with open(output_file, 'w') as f:
                json.dump(profiling_data, f, indent=2)
            print(f"Profiling data saved to {output_file}")