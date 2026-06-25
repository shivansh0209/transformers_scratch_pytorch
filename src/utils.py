import torch
import os
import urllib.request
import numpy as np
import zipfile
import tensorflow as tf
from keras.layers import TextVectorization

# def generate_positional_encoding(input_tensor):
#     batch_size, seq_len, d_model = input_tensor.size()
#     positional_encoding = torch.zeros_like(input_tensor, dtype=torch.float32)

#     for batch_idx in range(batch_size):
#         for pos in range(seq_len):
#             for i in range(int(d_model / 2)):
#                 denominator = 10000 ** ((2 * i) / d_model)
                
#                 # Wrap the scalar math inside torch.tensor()
#                 val = torch.tensor(pos / denominator, dtype=torch.float32)
                
#                 positional_encoding[batch_idx, pos, 2 * i] = torch.sin(val)
#                 positional_encoding[batch_idx, pos, 2 * i + 1] = torch.cos(val)

#     return positional_encoding


# RETROSPECTIVE: Optimization point 1
def generate_positional_encoding(input_tensor):
    batch_size, seq_len, d_model = input_tensor.size()
    
    # 1. Create a 1D column vector of positions: [0, 1, 2, ..., seq_len-1]
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    
    # 2. Compute the denominators for the even indices (2i) using steps of 2
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    denominator = 10000 ** (i / d_model)
    
    # 3. Create a blank 2D matrix for a single sequence
    pe = torch.zeros(seq_len, d_model)
    
    # 4. Use slice step notation [start:end:step] to fill even and odd columns
    # pos / denominator automatically uses broadcasting to create a (seq_len, d_model/2) grid
    pe[:, 0::2] = torch.sin(pos / denominator) # Fills indices 0, 2, 4, ...
    pe[:, 1::2] = torch.cos(pos / denominator) # Fills indices 1, 3, 5, ...
    
    # 5. Add a batch dimension at index 0 -> shape: (1, seq_len, d_model)
    # PyTorch will automatically stretch this 1 to match your batch_size during addition
    return pe.unsqueeze(0)



def load_and_vectorize_data():
    url = "https://www.manythings.org/anki/fra-eng.zip"
    # Create the directory safely
    os.makedirs("../data", exist_ok=True)
    zip_path = "../data/fra-eng.zip"
    txt_path = "../data/fra.txt"

    if not os.path.exists(txt_path):
        print("Downloading English-French dataset with browser headers...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("../data/")
        print("Download and extraction complete.")

    # --- 2. LOAD RAW TEXT ---
    raw_eng, raw_fra_in, raw_fra_out = [], [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts) >= 2:
                eng = parts[0].lower().strip()
                fra = parts[1].lower().strip()
                
                raw_eng.append(eng)
                raw_fra_in.append(f"startseq {fra}")
                raw_fra_out.append(f"{fra} endseq")

    # Slice a generous subset (e.g., 20,000) so your RAM doesn't crash, 
    # while keeping sentences of any arbitrary length!
    num_samples = 20000 
    eng_sentences = raw_eng[:num_samples]
    fra_in_sentences = raw_fra_in[:num_samples]
    fra_out_sentences = raw_fra_out[:num_samples]

    # --- 3. DYNAMICALLY FIND MAX LENGTHS ---
    # Instead of guessing 10 or 12, let's find the true max of this subset
    max_src_len = max(len(s.split()) for s in eng_sentences)
    max_trg_len = max(len(s.split()) for s in fra_in_sentences)

    # Source (English) Vectorizer
    src_vectorizer = TextVectorization(
        max_tokens=None,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=max_src_len
    )
    src_vectorizer.adapt(eng_sentences)

    # Target (French) Vectorizer
    def custom_standardize(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, r"[.?!,¿]", "")

    trg_vectorizer = TextVectorization(
        max_tokens=None,
        standardize=custom_standardize,
        output_mode="int",
        output_sequence_length=max_trg_len
    )
    trg_vectorizer.adapt(fra_in_sentences + fra_out_sentences)

    # Convert strings directly to integer tensors
    encoder_input = src_vectorizer(np.array(eng_sentences))
    decoder_input = trg_vectorizer(np.array(fra_in_sentences))
    decoder_target = trg_vectorizer(np.array(fra_out_sentences))

    src_vocab_size = len(src_vectorizer.get_vocabulary())
    trg_vocab_size = len(trg_vectorizer.get_vocabulary())

    print(f"Vectorization Ready. English Vocab: {src_vocab_size}, French Vocab: {trg_vocab_size}")
    print(f"Max Source Length: {max_src_len}, Max Target Length: {max_trg_len}")

    src_vectorizer.vocab_size = len(src_vectorizer.get_vocabulary())
    trg_vectorizer.vocab_size = len(trg_vectorizer.get_vocabulary())
    trg_vectorizer.word_to_index = {word: i for i, word in enumerate(trg_vectorizer.get_vocabulary())}
    
    # Convert TensorFlow tensors -> NumPy arrays -> PyTorch tensors
    enc_in_pt = torch.tensor(encoder_input.numpy(), dtype=torch.long)
    dec_in_pt = torch.tensor(decoder_input.numpy(), dtype=torch.long)
    dec_tar_pt = torch.tensor(decoder_target.numpy(), dtype=torch.long)

    return enc_in_pt, dec_in_pt, dec_tar_pt, src_vectorizer, trg_vectorizer, max_src_len, max_trg_len


def translate_sentence(sentence, model, src_vectorizer, trg_vectorizer, max_trg_len):
    model.eval() # Set model to evaluation mode
    
    # 1. PREPROCESS SOURCE (ENGLISH) TEXT
    # Vectorize the raw string input using your trained TextVectorization layer
    # Add a batch dimension using unsqueeze(0) -> shape: (1, max_src_len)
    src_tokens = src_vectorizer([sentence])
    src_tensor = torch.tensor(src_tokens.numpy(), dtype=torch.long)
    
    # 2. INITIALIZE DECODER WITH START TOKEN
    # Look up the index for "startseq" dynamically from your vocabulary mapping
    start_idx = trg_vectorizer.word_to_index["startseq"]
    end_idx = trg_vectorizer.word_to_index["endseq"]
    
    # Initialize our running tracker of predicted target tokens
    # Start shape: (1, 1) -> contains just [[start_idx]]
    trg_indices = [[start_idx]]
    
    # 3. AUTOREGRESSIVE GENERATION LOOP
    with torch.no_grad(): # Disable gradient calculations to save memory
        for _ in range(max_trg_len):
            # Convert current list of target tokens into a PyTorch tensor
            trg_tensor = torch.tensor(trg_indices, dtype=torch.long)
            
            # Forward pass: pass the entire source sequence and the targets generated SO FAR
            # Output shape: (1, current_seq_len, trg_vocab_size)
            outputs = model(src_tensor, trg_tensor)
            
            # Grab the prediction for the VERY LAST token position only
            # shape: (trg_vocab_size,)
            next_token_logits = outputs[0, -1, :]
            
            # Pick the word index with the highest probability
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Append the predicted word ID to our decoder sequence tracker
            trg_indices[0].append(next_token_id)
            
            # Stop generating immediately if the model predicts the end-of-sentence token
            if next_token_id == end_idx:
                break
                
    # 4. CONVERT TOKEN IDS BACK TO STRINGS
    vocab = trg_vectorizer.get_vocabulary()
    translated_words = [vocab[idx] for idx in trg_indices[0] if vocab[idx] not in ["startseq", "endseq", ""]]
    
    return " ".join(translated_words)

__all__ = ['generate_positional_encoding', 'load_and_vectorize_data', 'translate_sentence']