import re

def preprocess_poem(poem):
    # Lowercase all words
    poem = poem.lower()
    # Remove any punctuation or symbols
    poem = re.sub(r'[^\w\s]', '', poem)
    # Add the special tokens
    poem = "<SOS> " + poem + " <EOS>"
    return poem

with open("poems.txt", "r") as f_in, open("preprocessed_poems.txt", "w") as f_out:
    poems = f_in.readlines()
    preprocessed_poems = [preprocess_poem(poem) for poem in poems]
    f_out.write("\n".join(preprocessed_poems))