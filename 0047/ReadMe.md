# DeepSeek-OCR: Contexts Optical Compression

This paper explains the core concepts behind **DeepSeek-OCR**, a specialized Vision-Language Model (VLM) designed to compress massive amounts of text into tiny visual "tokens."

---

## The Big Idea
Current AI models (LLMs) struggle with very long documents. The longer the text, the slower and more expensive the AI becomes. 

**DeepSeek's solution:** Instead of feeding the AI thousands of words as text, why not take a "picture" of the page? 
A single image can hold 1,000 words but can be represented by just a few "vision tokens." DeepSeek-OCR acts like a **high-powered digital compressor** that reads images and turns them into text with extreme efficiency.

---

## 🛠 How It Works (Architecture)
The model is split into two main parts:

### 1. The DeepEncoder (The "Eyes")
This is a 380M parameter encoder that "sees" the image. It has a unique **Serial Design**:
*   **Detailed Vision (SAM):** First, it looks at the fine details (like small letters or lines).
*   **The Compressor:** This is the magic part. It shrinks the visual data by **16x**. 
*   **General Knowledge (CLIP):** Then, it looks at the overall layout of the page.
*   **Result:** It turns a high-resolution image into a very small number of tokens (as few as 64 to 100).

### 2. The Decoder (The "Brain")
It uses **DeepSeek-3B-MoE**. 
*   Even though it has 3 Billion parameters, it only "wakes up" about **570 Million** parameters at a time to save power and speed.
*   It takes the tiny compressed tokens from the encoder and "decompresses" them back into perfect text, tables, or formulas.

---

## Key Features & Performance

### 1. Massive Compression
The paper proves a "10x Rule":
*   **Up to 10x Compression:** 97% accuracy. (e.g., representing 1000 words with only 100 vision tokens).
*   **20x Compression:** 60% accuracy. (Useful for summarizing or "forgetting" unimportant details).

### 2. Deep Parsing (More than just text)
DeepSeek-OCR doesn't just read words; it understands:
*   **Charts:** Converts graphs into clean HTML tables.
*    **Chemical Formulas:** Reads complex molecular structures.
*    **Geometry:** Understands shapes, lines, and angles.
*    **Natural Images:** Can describe photos found inside documents.

### 3. Production Speed
This model is built for "The Real World":
*   A single A100 GPU can process **200,000+ pages per day**.
*   A full cluster can process **33 Million pages per day**.

---

##  Comparison with Others

| Model | Efficiency | Performance |
| :--- | :--- | :--- |
| **GPT-4o** | Expensive/High Token Usage | Very Smart, but slow for bulk OCR |
| **MinerU 2.0** | Uses ~7,000 tokens per page | Good, but very "heavy" |
| **DeepSeek-OCR** | Uses **< 800** tokens (Gundam Mode) | **State-of-the-Art** accuracy & ultra-fast |

---

##  Conclusion
DeepSeek-OCR proves that **"A picture is worth a thousand words"** is literally true for AI. By using vision to compress text, we can build AIs that read faster, cost less, and handle much longer contexts than ever before.

---
*Based on the paper: "DeepSeek-OCR: An Investigation into the Feasibility of Compressing Long Contexts via Optical 2D Mapping"*