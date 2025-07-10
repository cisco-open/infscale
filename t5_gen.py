import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def t5_autoregressive_generate(model, tokenizer, input_text, 
                                max_new_tokens=50, temperature=1.0, top_p=1.0):
    device = model.device
    model.eval()

    # Prepare the input (encoder side)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    encoder_outputs = model.encoder(input_ids=input_ids)
    
    # Start decoding with <pad> (bos token in T5)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)

    generated_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )
            lm_logits = model.lm_head(decoder_outputs.last_hidden_state)  # [1, t, vocab_size]
            next_token_logits = lm_logits[:, -1, :] / temperature

            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[0, indices_to_remove] = -float("inf")

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Break on eos
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token.item())

        # Update decoder_input_ids
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

if __name__ == "__main__":
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")

    input_prompt = "summarize: Hugging Face Transformers provides many models and tools for NLP tasks. The library is designed to be flexible and easy to use, with a focus on providing a high-level API for building and training models."
    output_text = t5_autoregressive_generate(model, tokenizer, input_prompt)

    print("Input:", input_prompt)
    print("Output:", output_text)