import torch
import os
import urllib.request
import numpy as np
import zipfile
import tensorflow as tf
from keras.layers import TextVectorization

def generate_positional_encoding(input_tensor):
    batch_size, seq_len, d_model = input_tensor.size()
    positional_encoding = torch.zeros_like(input_tensor, dtype=torch.float32)

    for batch_idx in range(batch_size):
        for pos in range(seq_len):
            for i in range(int(d_model / 2)):
                denominator = 10000 ** ((2 * i) / d_model)
                
                # Wrap the scalar math inside torch.tensor()
                val = torch.tensor(pos / denominator, dtype=torch.float32)
                
                positional_encoding[batch_idx, pos, 2 * i] = torch.sin(val)
                positional_encoding[batch_idx, pos, 2 * i + 1] = torch.cos(val)

    return positional_encoding



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


__all__ = ['generate_positional_encoding']