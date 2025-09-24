# SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing

## 0. What SentencePiece Is

- SentencePiece is a small, language independent tool that breaks text into subword pieces.  
- Unlike older tools, it can learn directly from raw sentences (no pre-splitting of words), which makes it easy to use for any language — English, Japanese, Chinese, etc.  
- It gives reproducible, lossless tokenization and can be used in training neural text models or to tokenize text at runtime.  

### Why this matters  

When you train language models or machine translation, you need to split text into units the model understands.  
Whole words cause problems: very large vocabularies, rare words, and languages without spaces.  
SentencePiece fixes this by learning small building blocks (subwords) that cover words and parts of words.  
It works the same for any language and keeps enough information to reconstruct the original text exactly.  

---

## 1. Problem Setup

We want to map a raw sentence \( S \) into a sequence of subword tokens  

SentencePiece builds a vocabulary \( V \) of subwords and learns how to segment efficiently.  

We also want to be able to decode back to the exact normalized \( S \).  

---

## 2. Byte Pair Encoding (BPE)

BPE builds the vocabulary by merging the most frequent pairs of symbols.  

- Start with a character-level vocabulary.  
- Count pair frequencies in the training corpus.  
- Merge the most frequent pair into a new symbol.  
- Repeat until the target vocabulary size is reached.  

### Example (toy data)

Corpus:
```
low, lowest, new, newer
```

Initial symbols:
```
l o w
l o w e s t
n e w
n e w e r
```

Step 1: Count frequent pairs  
- "l o" → 2 times  
- "o w" → 3 times  
- "n e" → 2 times  

Most frequent = "o w". Merge → "ow".  

New symbols:
```
l ow
l ow e s t
n e w
n e w e r
```

Step 2: Next most frequent = "l ow". Merge → "low".  
Continue until vocabulary size target is reached.  
---

## 3. Unigram Language Model (ULM)

ULM treats segmentation as a probabilistic model.  
We start with a large vocabulary and prune unlikely tokens.  

Training is done with the EM algorithm:  
- E-step: compute expected counts of each token using forward-backward.  
- M-step: re-estimate token probabilities.  
- Iteratively remove tokens with low probability.  

### Example

Sentence: `"low"`  
Possible segmentations:  
- ["low"]  
- ["l", "ow"]  
- ["lo", "w"]  
- ["l", "o", "w"]  

The model assigns higher probability to the most natural segmentation.  

---

## 4. Subword Regularization

Instead of always taking the most likely segmentation, SentencePiece can sample from the distribution of possible tokenizations. This acts as data augmentation: the model sees slightly different tokenizations during training.  

---

## 5. Summary

- BPE: iterative merging of frequent pairs.  
- ULM: probabilistic segmentation using EM.  
- Both methods build a subword vocabulary that can represent any text, even unseen words.  
- SentencePiece = practical implementation with normalization, training, encoding, decoding, and subword regularization.  
