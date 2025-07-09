from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
model.eval()

# Text with [MASK] token for prediction
# text = "Columbus is the state capital of Ohio, known for its [MASK] university and tech scene."
texts = ["The capital of France is [MASK].",
         "The capital of Germany is [MASK].",
         "The capital of France is [MASK] and the capital of Germany is [MASK]."]

inputs = tokenizer(texts, return_tensors="pt", padding=True)
mask_token_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

with torch.no_grad():
    outputs = model(**inputs)
    
# Get predictions for all masked tokens
token_logits = outputs.logits
print(f"Original text: {texts}")
print(f"Found {len(mask_token_indices)} mask token(s)")

# Process each sequence in the batch
results = []
for i, (sequence_logits, sequence_input_ids) in enumerate(zip(token_logits, inputs["input_ids"])):
    # Find mask token positions
    mask_token_indices = torch.where(sequence_input_ids == tokenizer.mask_token_id)[0]
    print(f"Mask token indices: {mask_token_indices}")
    
    if len(mask_token_indices) == 0:
        # No mask tokens, just decode the original text
        decoded_text = tokenizer.decode(sequence_input_ids, skip_special_tokens=True)
        results.append(decoded_text)
    else:
        # Create a copy of the input_ids to modify
        modified_input_ids = sequence_input_ids.clone()
        
        # Replace mask token IDs with predicted token IDs
        for mask_pos in mask_token_indices:
            mask_logits = sequence_logits[mask_pos]
            top_token_id = torch.argmax(mask_logits).item()
            modified_input_ids[mask_pos] = top_token_id
        
        # Decode the modified sequence
        complete_text = tokenizer.decode(modified_input_ids, skip_special_tokens=True)
        results.append(complete_text)

# Print results
for i, result in enumerate(results):
    print(f"Result {i}: {result}")