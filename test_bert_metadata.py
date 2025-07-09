import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, PretrainedConfig
from infscale.module.modelir import ModelIR
from infscale.module.model_metadata import BertModelMetaData

from accelerate.utils.modeling import set_module_tensor_to_device
import numpy as np

def test_bert_metadata():
    """Test the BertModelMetaData get_output_parser and get_predict_fn functions"""
    
    print("="*60)
    print("Testing BertModelMetaData with Masked Language Modeling")
    print("="*60)
    
    # Initialize model metadata
    model_name = "bert-base-uncased"
    print(f"Loading model: {model_name}")
    
    # Load the actual model for testing
    full_model = AutoModelForMaskedLM.from_pretrained(model_name)
    config = full_model.config
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create model metadata and IR
    model_metadata = BertModelMetaData(model_name, config)
    model_metadata.trace_inputs = ["input_ids", "token_type_ids", "attention_mask"]
    model_ir = ModelIR(model_metadata)

    layers = model_ir.layers

    layer_modules = []
    for i, layer in enumerate(layers):
        print(f"Initializing layer {i}")
        # Create empty layer on cpu
        layer = layer.to_empty(device="cpu")
            
        # Copy parameters from full model using device transfer helper
        for name, param in layer.named_parameters():
            full_param = full_model.get_parameter(name)
            set_module_tensor_to_device(
                layer,
                name,
                "cpu",
                value=full_param.data,
                dtype=full_param.dtype
            )
        
        # Copy buffers similarly
        for name, buf in layer.named_buffers():
            full_buf = full_model.get_buffer(name)
            set_module_tensor_to_device(
                layer,
                name,
                "cpu",
                value=full_buf.data,
                dtype=full_buf.dtype
            )
        
        layer_modules.append(layer.to("cuda"))

    print(f"We have {len(layer_modules)} layers")
    
    # Test sentences with [MASK] tokens
    test_sentences = [
        "The capital of France is [MASK].",
        "Berlin is the capital of [MASK].",
        "The [MASK] is a large mammal that lives in the ocean.",
        "Python is a popular [MASK] language."
    ]
    
    print(f"\nTesting with {len(test_sentences)} sentences:")
    for i, sentence in enumerate(test_sentences):
        print(f"{i+1}. {sentence}")
    
    # Tokenize all sentences (batch processing)
    print(f"\nTokenizing sentences...")
    inputs = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Batch size: {inputs['input_ids'].shape[0]}")
    
    # Run inference through the model
    print(f"\nRunning inference...")
    with torch.inference_mode():
        outputs = inputs
        for i, layer in enumerate(layer_modules):
            outputs = layer(**outputs)

    outputs["input_ids"] = inputs["input_ids"]

    bert_predictions = model_metadata.get_predict_fn()(outputs)
    print(bert_predictions)

if __name__ == "__main__":
    test_bert_metadata() 