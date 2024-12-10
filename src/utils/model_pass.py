import torch


def get_lm_response(prompts, model, tokenizer, model_name, max_new_tokens=32):
    """
    Generalized forward pass function to generate predictions from a batch of prompts.
    
    Args:
        prompts (list): A list of input prompts for the model.
        model: The loaded model object.
        tokenizer: The tokenizer associated with the model.
        model_name (str): The name of the model to handle specific configurations.
        max_new_tokens (int): The maximum number of tokens to generate.
    
    Returns:
        list: The candidate batch of generated responses.
    """
    if 'gpt' not in model_name and 'gemini' not in model_name:
        # Non-GPT/Gemini models
        device = next(model.parameters()).device
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device=device)
        inputs = {key: value.to(dtype=torch.int32) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        candidate_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        candidate_batch = [(candidate[len(prompt):]) for candidate, prompt
                           in zip(candidate_batch, prompts)]
    else:
        # GPT/Gemini models
        candidate_batch = [model(prompt) for prompt in prompts]

    return candidate_batch
