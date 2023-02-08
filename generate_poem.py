import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# Load the fine-tuned model checkpoint
model = torch.load('gpt2_poem_model.pth')
model.eval()

# Define a function to generate text
def generate_text(model, prompt, temperature=0.8):
    model.to('cpu')
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    logits = logits[0, -1, :] / temperature
    logits = F.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(logits, num_samples=1).item()
    generated_text = tokenizer.decode(input_ids[0].tolist() + [next_token_id])

    return generated_text

# Generate text using the fine-tuned model
prompt = "كلمة بداية للشعر"
generated_text = generate_text(model, prompt)
print(generated_text)