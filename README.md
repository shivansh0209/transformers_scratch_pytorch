# Transformer From Scratch: English-to-French Translation

A production-grade, fully vectorized Transformer architecture built entirely from scratch using PyTorch. This project avoids high-level abstractions to map exactly how multi-dimensional tensors interact, perform matrix operations, and flow through a sequence-to-sequence neural network.

---

## 📂 Project Structure

The repository follows a clean, professional, and decoupled layout separating data assets, scratchpad exploration, and modular source configurations:

```text
transformer_from_scratch/
├── data/
│   ├── _about.txt
│   ├── fra-eng.zip
│   └── fra.txt
├── notebooks/
│   └── model_training_and_inference.ipynb
├── plots/
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── encoder_decoder.py
│   ├── modules.py
│   ├── transformer_model.py
│   └── utils.py
├── .gitignore
├── README.md
├── reqs.txt
└── RETROSPECTIVE.md
```

### File Manifest & Responsibilities
* src/modules.py: The computational core. Houses heavily optimized implementations of MultiHeadAttention, CrossAttention, and LayerNorm.

* src/encoder_decoder.py: Defines the sequential block stacks (EncoderLayer/Encoder and DecoderLayer/Decoder) that aggregate the attention mechanisms and feed-forward networks.

* src/transformer_model.py: The top-level wrapper class that glues the Encoder and Decoder together with source/target token embeddings and the final Language Model output linear layer.

* src/utils.py: Contains structural mathematical utilities including fully vectorized positional encoding generators.

* notebooks/model_training_and_inference.ipynb: End-to-end driver notebook used to load text datasets, vectorize sequences, manage the batched training loop, and execute generation testing.

### 🛠️ The Vectorization Breakthrough (Engineering Retrospective)
Building the math from scratch is one hurdle; making it run efficiently is another. The initial baseline implementation leaned heavily on naive Python tracking loops—specifically inside the multi-head attention slice logic, layer normalization loops, and row-by-row positional encoding loops.

* The Problem: Running a single epoch on a subset of 20,000 sentences took over 20 minutes on a standard CPU due to Python interpreter overhead and sequential instruction bottlenecks.

* The Optimization: Every component was systematically refactored using pure PyTorch Vectorization techniques:

* Replaced nn.ModuleList looping heads with single, wide projection linear layers (nn.Linear(d_model, d_model)).

* Leveraged Tensor Broadcasting inside masking logic and LayerNorm calculations (dim=-1, keepdim=True) to avoid memory-heavy .repeat() calls.

* Pre-calculated full dynamic positional coordinate matrices out-of-loop using torch.arange.

The Result: Training time collapsed from 20 minutes down to a blazing 1:30 minutes per epoch on CPU, validating the massive performance gains of utilizing raw C++ hardware acceleration via PyTorch tensors.

### Implemented modern adaptation - Rope
In rope instead of "producing absolute vectors which doesnt contribute in any extrapolation and doesnt ingest the positions directly into the attention matrix but they distort the input as are additive", we use the concept of rotation of token embeddings in the visualization space before calculation of the attention matrix.

This leads to better idea of relative positions as it preserved it like if two tokens are three tokens apart then whatever there position is they will portray the same relative distance in the Attention matrix unlike the sinusoidal which creates non linear aboslute position vectors


# 📈 Model Performance & Current Results

The network was evaluated on an **English-to-French translation task** using a **20,000-sample subset** of the **Anki sentence pair dataset**.

---

## Short Sentences (High Accuracy)

For target lengths of **3 to 4 words** such as:

- `"I am a student"`
- `"Go away"`

the model achieves:

- High translation precision
- Strong target structure matching
- Effective sequence alignment

Because the sliced training subset is dominated by shorter phrase structures, the **cross-attention layers quickly converged** on localized sequence alignment dependencies.

---

## Long Sentences (Structural Degradation)

Performance noticeably decreases as sequence lengths increase.

This degradation is primarily caused by:

### 1. Positional Horizon

The sorted nature of the raw dataset caused:

- Longer sentence variations to be underrepresented during training
- Positional encoding patterns to have limited exposure
- Attention layers to develop blind spots for extended contexts

---

### 2. Exposure Bias

The current inference pipeline uses a **Greedy Search strategy**:

```python
torch.argmax()
```

This creates a cascading error problem:

- Model predicts an incorrect early token
- The incorrect output is fed back into the decoder
- Future predictions become increasingly biased
- Translation quality decreases for longer sequences


# 🚀 Next Steps & Planned Improvements

To overcome current limitations and move toward **production-level translation quality**, the following improvements are planned:
---

## 1. Implementing Beam Search

Replace the current **greedy decoder** with a **Beam Search generation algorithm**.

### Benefits:

- Tracks top $K$ translation hypotheses simultaneously
- Reduces dependency on a single prediction path
- Mitigates cascading inference errors

---

## 2. Dataset Shuffling Pipelines

Introduce randomized dataset preprocessing before:

- Tokenization
- Sequence slicing

### Benefits:

- Prevents ordering bias
- Exposes attention layers to diverse sequence lengths
- Improves long-context learning from the beginning of training

---

## 3. Learning Rate Scheduling

Introduce:

- Warm-up phase
- Cosine decay scheduling

### Benefits:

- More stable early optimization
- Better attention weight coordination
- Improved convergence of transformer attention matrices