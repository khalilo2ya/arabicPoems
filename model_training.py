import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import re

# Load the preprocessed poems data
with open("preprocessed_poems.txt", "r") as f:
    poems = f.readlines()

# Load the pre-trained GPT-2 model
model = torch.hub.load("huggingface/transformers", "modelWithLMHead", "gpt2")
model.eval()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Fine-tune the model for a specified number of epochs
num_epochs = 10
for epoch in range(num_epochs):
    random.shuffle(poems)
    for poem in poems:
        optimizer.zero_grad()
        input_ids = torch.tensor(tokenizer.encode(poem)).unsqueeze(0)
        loss = criterion(model(input_ids)[0], input_ids[0, 1:])
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
torch.save(model.state_dict(), "gpt2_poem_model.pth")