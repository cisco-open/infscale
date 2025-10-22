import argparse
import json
import os
import copy

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def build_trimmed_state_dict(full_sd: dict, kept_layer_index: int) -> dict:
    """
    Construct a trimmed state dict containing:
    - model.embed_tokens.*
    - model.layers.0.* (copied from model.layers.{kept_layer_index}.*)
    - model.norm.*
    - lm_head.*

    Any other keys are dropped.
    """
    trimmed_sd = {}

    # Copy embedding, final norm and lm_head as-is
    for key in list(full_sd.keys()):
        if key.startswith("model.embed_tokens."):
            trimmed_sd[key] = full_sd[key]
        elif key.startswith("model.norm."):
            trimmed_sd[key] = full_sd[key]
        elif key.startswith("lm_head."):
            trimmed_sd[key] = full_sd[key]

    # Map the selected uniform layer to index 0
    src_prefix = f"model.layers.{kept_layer_index}."
    dst_prefix = "model.layers.0."
    for key, tensor in full_sd.items():
        if key.startswith(src_prefix):
            new_key = dst_prefix + key[len(src_prefix):]
            trimmed_sd[new_key] = tensor

    return trimmed_sd


def _get_decoder_layers(model):
    """Return the decoder layers ModuleList for LLaMA-like models."""
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise ValueError("Given model does not expose model.layers; expected LLaMA-like architecture")
    return backbone.layers


def print_layers_overview(model, kept_layer_index: int) -> None:
    """Print all decoder layers and a detailed view of the selected layer."""
    layers = _get_decoder_layers(model)
    print("==== Decoder layer overview ====")
    print(f"Total decoder layers: {len(layers)}")
    for idx, layer in enumerate(layers):
        num_params = sum(p.numel() for p in layer.parameters())
        child_names = [name for name, _ in layer.named_children()]
        mark = "<-- kept" if idx == kept_layer_index else ""
        print(f"[{idx}] {layer.__class__.__name__} | params={num_params:,} | submodules={child_names} {mark}")

    # Detailed view for the kept layer
    print(f"\n==== Detailed parameters for kept layer {kept_layer_index} ====")
    kept_layer = layers[kept_layer_index]
    for name, param in kept_layer.named_parameters():
        try:
            shape = tuple(param.shape)
        except Exception:
            shape = "?"
        print(f"  - {name}: {shape} | dtype={param.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Trim a LLaMA model to a single uniform decoder layer.")
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "llama_70b"], help="Model family to trim (default: llama)")
    parser.add_argument("--output_dir", type=str, default="trimmed_model", help="Directory to save the trimmed model")
    parser.add_argument("--kept_layer_index", type=int, default=0, help="Which decoder block index to keep (default: 0)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve base model from model_type if not provided
    if args.model_type == "llama":
        base_model = "meta-llama/Meta-Llama-3.1-8B"
    elif args.model_type == "llama_70b":
        base_model = "meta-llama/Meta-Llama-3.1-70B"
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    # Load base config and model on CPU
    base_config = AutoConfig.from_pretrained(base_model)
    if not hasattr(base_config, "num_hidden_layers"):
        raise ValueError(f"Model {base_model} does not appear to be a LLaMA-like decoder with num_hidden_layers")

    full_model = AutoModelForCausalLM.from_pretrained(base_model)
    full_model.to("cpu")
    full_model.eval()

    orig_layers = base_config.num_hidden_layers
    print(f"Original number of layers: {orig_layers}")
    if args.kept_layer_index < 0 or args.kept_layer_index >= orig_layers:
        raise ValueError(f"kept_layer_index {args.kept_layer_index} out of range [0, {orig_layers-1}]")

    # Print all layers and highlight the selected one
    print_layers_overview(full_model, kept_layer_index=args.kept_layer_index)

    print("==== Starting to build trimmed model ====")

    # Build a new config with a single layer
    new_config = copy.deepcopy(full_model.config)
    new_config.num_hidden_layers = 1
    # Preserve name_or_path for downstream tokenization
    new_config.name_or_path = base_model

    # Instantiate a new tiny model and load trimmed weights
    tiny_model = AutoModelForCausalLM.from_config(new_config)
    tiny_model.to("cpu")
    tiny_model.eval()

    full_sd = full_model.state_dict()
    trimmed_sd = build_trimmed_state_dict(full_sd, kept_layer_index=args.kept_layer_index)

    # Load with non-strict to ignore generated buffers (like rotary emb) not present in trimmed_sd
    incompatible = tiny_model.load_state_dict(trimmed_sd, strict=False)
    unexpected = getattr(incompatible, "unexpected_keys", [])
    missing = getattr(incompatible, "missing_keys", [])
    if unexpected:
        # Unexpected keys in trimmed_sd should be rare; warn but continue
        raise ValueError(f"[warn] Unexpected keys while loading trimmed state dict: {unexpected}")
    if missing:
        # Some buffers (e.g., rotary caches) are expected to be missing
        raise ValueError(f"[info] Missing keys initialized from config: {missing}")

    # Save trimmed model and tokenizer
    tiny_model.save_pretrained(args.output_dir, safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"[warn] Failed to save tokenizer: {e}")

    # Write metadata to help reconstruct per-layer profiles
    meta = {
        "base_model": base_model,
        "original_num_hidden_layers": int(orig_layers),
        "kept_layer_index": int(args.kept_layer_index),
        "uniform_layers_path": "model.layers",
        "note": "This trimmed model contains a single decoder block. Multiply per-layer metrics by original_num_hidden_layers to approximate full-model totals.",
    }
    with open(os.path.join(args.output_dir, "trim_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved trimmed model to {args.output_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


