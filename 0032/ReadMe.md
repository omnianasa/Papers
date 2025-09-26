# Neural Machine Translation of Rare Words with Subword Units

## Overview
NMT models usually work with a fixed size vocabulary (30k–50k words).  
However, real-world translation is an open vocabulary problem: new, rare, and morphologically complex words constantly appear.  
Traditional solutions rely on back-off dictionaries (substituting rare words with dictionary lookups).  

This paper introduces a simpler and more effective approach: Represent rare and unseen words as sequences of subword units  

The authors propose using a variant of Byte Pair Encoding (BPE) for word segmentation, enabling open-vocabulary NMT.

---

## Key Contributions
1. Demonstrated that NMT can handle open vocabulary translation by splitting words into subwords.  
2. Adapted Byte Pair Encoding (BPE) as a segmentation algorithm for translation tasks.  
3. Showed that subword based NMT outperforms baselines with large vocabularies and back-off dictionaries.  
4. Provided detailed evaluation on WMT’15 English -> German and English -> Russian translation tasks.  

---

## Core Ideas

### Encoder-Decoder with Attention
NMT follows the encoder-decoder framework with attention.  
- The encoder reads input tokens and produces hidden states:  

```math
h_j = [ \overrightarrow{h_j} ; \overleftarrow{h_j} ]
```  

- The decoder predicts the next target word based on hidden states, previous outputs, and a context vector:  

```math
c_i = \sum_j lpha_{ij} h_j
```  

where \( lpha_{ij} \) is the attention weight aligning source position *j* with target word *i*.

---

### Subword Translation Categories
Subword modeling is especially powerful for:  
- Named Entities -> copy or transliterate (e.g., *Obama* → *Обама*).  
- Loanwords & Cognates -> apply regular transformations (*claustrophobia* → *Klaustrophobie*).  
- Compounds / Morphology -> translate parts independently (*solar system* → *Sonnensystem*).  

---

### Byte Pair Encoding (BPE)
BPE is adapted for segmentation:  
- Start with character vocabulary.  
- Iteratively merge most frequent symbol pairs.  
- Build a vocabulary of variable-length subword units.  

Example:  
Dictionary = { "low", "lower", "newest", "widest" }  
After BPE merges:  
- `l o → lo`  
- `lo w → low`  
- `e r → er`  

So, unseen word **"lower"** becomes segmented as:  
```
low | er
```  

---

## Results

### Corpus Statistics (German)
| Method              | #Tokens | #Types | #UNK in test |
|----------------------|---------|--------|--------------|
| None (word-level)   | 100M    | 1.75M  | 1079         |
| Char bigrams        | 306M    | 20k    | 34           |
| Morfessor           | 109M    | 544k   | 237          |
| **BPE (60k merges)** | 112M    | 63k    | 0            |

---

### Translation Quality (EN→DE, newstest2015)
| System   | BLEU | CHRF3 | Rare Word F1 |
|----------|------|-------|---------------|
| WDict    | 24.2 | 52.4  | 36.8%         |
| C2-50k   | 25.3 | 53.5  | 40.5%         |
| BPE-60k  | 24.5 | 53.9  | 40.9%         |
| **BPE-J90k** | 24.7 | **54.1** | **41.8%** |

---


## Resources
- Paper: [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)  
- Code: [subword-nmt](https://github.com/rsennrich/subword-nmt)  

