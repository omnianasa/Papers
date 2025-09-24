# CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation

Today we explain the CANINE model paper. CANINE is a Transformer encoder that does not depend on tokenizers or fixed vocabularies. Instead, it processes text directly at the character level using Unicode codepoints.

---

## Why CANINE Matters

Most NLP models (like BERT, GPT, etc.) depend on tokenization (splitting text into subword units). While useful, tokenizers have big downsides:

* They struggle with languages that have complex word forms (Arabic, Turkish) or languages that lack spaces (Chinese, Thai).
* They fail with noisy text (typos, emojis, unusual symbols).
* Tokenizers require heavy engineering and lock the model into one vocabulary.

CANINE avoids tokenization entirely. It works directly with characters, so it can handle any script, any symbol, and even text it never saw before.

---

## Core Ideas

1. **Unicode-based preprocessing**
   Instead of token IDs, CANINE uses raw Unicode codepoints (e.g., `'a' → 97`). This step is trivial and stable across all languages.

2. **Hash-based embeddings**
   Full Unicode is huge (>140k characters). CANINE uses multiple hash functions and small embedding tables to represent any character efficiently. Concatenating these small embeddings creates the character vector.

3. **Downsampling for efficiency**
   Character sequences are long. CANINE first uses a *local transformer* to capture short-range patterns, then a strided convolution to shrink the sequence length. This keeps compute similar to BERT.

4. **Deep transformer encoder**
   The compressed sequence is fed into a standard deep Transformer stack (12 layers in the main setup) to capture context across the sentence.

5. **Upsampling when needed**
   For tasks that need outputs at the character level (like tagging), CANINE reconstructs character-level representations by combining local and global features.

6. **Pretraining objectives**

   * CANINE-C: predicts masked characters directly (autoregressive over masked spans).
   * CANINE-S: uses subword spans *only during pretraining* to define the loss, then discards the tokenizer. This makes tokenization just a soft bias, not a requirement.

---

## Step-by-Step Pipeline

1. **Text → Codepoints**
   Each character becomes its Unicode codepoint integer.

2. **Hash Embeddings**
   Each codepoint is hashed multiple times, looked up in small tables, then concatenated to form the character embedding.

3. **Local Character Encoder**
   A lightweight transformer attends within small blocks (e.g., 128 chars) to build word-like features.

4. **Downsampling**
   A strided convolution reduces sequence length (e.g., 4× shorter).

5. **Deep Transformer Stack**
   The compressed sequence goes through a 12-layer transformer.

   * For classification: take the first output vector as the summary (`y_cls`).
   * For sequence tasks: continue to upsampling.

6. **Upsampling (for sequence tasks)**
   Expand the compressed sequence back to character length, merge with local features, and run a final transformer for character-level outputs.

---

## Example Walk-through

**Input Sentence:**

```
The cat sat.
```

**1. Codepoints**

* 'T' → 84, 'h' → 104, 'e' → 101, ' ' → 32, 'c' → 99, 'a' → 97, 't' → 116, '.' → 46.
* Sequence = `[84, 104, 101, 32, 99, 97, 116, 32, 115, 97, 116, 46]`

**2. Hash Embeddings**
Each codepoint goes through 8 hash functions → 8 small vectors → concatenated into a 768-d vector per character.

**3. Local Transformer (h\_init)**
Captures very short patterns, like `"Th"` or `"cat"`, building word-like chunks.

**4. Downsampling**
Stride=4 convolution groups characters:

* \[84, 104, 101, 32] → position 0
* \[99, 97, 116, 32] → position 1
* \[115, 97, 116, 46] → position 2

Now we have 3 compact vectors instead of 12 characters.

**5. Deep Transformer Stack (ENCODE)**
These 3 vectors go through 12 layers of transformers. Context is shared, e.g., it learns `"cat"` is subject and `"sat"` is verb.

**6. For Classification Task (y\_cls)**
Take the first vector (position 0). This becomes the summary of the sentence. Feed it into a classifier → label = "neutral sentiment".

**7. For Sequence Task (y\_seq)**
Each compressed vector is replicated 4 times to cover the original characters, combined with local features, and processed again. Final outputs = one contextual vector per character, usable for tagging or QA.

---

## Key Hyperparameters

* Embedding dimension: 768
* Hash functions: 8
* Hash buckets: 16k
* Downsampling stride: 4
* Sequence length before downsampling: 2048 chars → after = 512 positions
* Transformer depth: 12 layers

---

## Strengths & Trade-offs

Strengths:

* No tokenizer needed — works for any language/script.
* Robust to typos, emojis, and unseen characters.
* Efficient training and inference (thanks to downsampling).
* Strong performance on morphologically rich languages.

Trade-offs

* Memorization of specific tokens is weaker than vocab-based models.
* Design choices (hash size, stride) affect results.
* Requires careful pretraining setup.

---

## Paper

check the paper [here](https://arxiv.org/pdf/2103.06874)
