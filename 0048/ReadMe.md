# DeepSeek-OCR 2: Visual Causal Flow

DeepSeek-OCR 2 is a next-generation Vision-Language Model (VLM) that moves away from traditional "raster-scan" (left-to-right) image processing. Instead, it mimics human vision by reordering visual information based on **logic and semantics** before the main brain (LLM) ever reads it.

---

## The Core Innovation: "Causal Flow"
Most AI models read images like a scanner: row by row, top to bottom. This works for simple text but fails on complex layouts like newspapers, multi-column research papers, or messy tables.

**DeepSeek-OCR 2** introduces **DeepEncoder V2**, which treats an image as a sequence of logically connected events. It uses a "Causal Flow" mechanism to decide which part of the image should be processed first, second, and so on, based on the actual content.

---

## Architectural Breakdown

### 1. Vision Tokenizer (The Filter)
*   **Components:** SAM-base (80M) + Convolutional Layers.
*   **Role:** It cuts the image into patches and compresses them by **16x**.
*   **Efficiency:** It reduces a high-resolution image into a manageable number of tokens, saving massive amounts of RAM.

### 2. DeepEncoder V2 (The "Thinking" Eye)
This is the biggest upgrade. DeepSeek replaced the standard CLIP encoder with a **Compact LLM (Qwen2-500M)**.
*   **Hybrid Attention Mask:** 
    *   **Part 1 (Global):** The model looks at the whole image at once to understand the layout.
    *   **Part 2 (Causal):** Using a "triangular mask" (like GPT), it forces the visual tokens to flow in a logical sequence.
*   **Result:** The encoder doesn't just see pixels; it understands the *reading order* of the document.



### 3. DeepSeek-MoE Decoder (The Brain)
*   **Structure:** A 3B-parameter Mixture-of-Experts (MoE) model.
*   **Active Parameters:** Only ~500M parameters are active during inference.
*   **Role:** It takes the logically ordered tokens from the encoder and generates the final output (Text, Markdown, HTML, or LaTeX).

---

## Performance & Capabilities

### Smart Scaling (Dynamic Resolution)
The model dynamically crops images into $k$ sub-images (up to 6) depending on the size. 
*   **Token Budget:** Between **256 and 1120 tokens**. This matches the efficiency of top-tier models like Gemini-3 Pro while maintaining higher accuracy for dense documents.

### Deep Parsing
DeepSeek-OCR 2 excels at:
*   **Complex Tables:** Handles merged cells and nested structures by understanding the causal relationship between rows and columns.
*   **Formula Recognition:** Flawlessly converts complex math and chemistry into LaTeX/SMILES.
*   **Visual Logic:** Can follow arrows in diagrams or the flow of a multi-page document.

---

## Comparison: OCR 1 vs. OCR 2

| Feature | DeepSeek-OCR (v1) | DeepSeek-OCR 2 |
| :--- | :--- | :--- |
| **Encoder Base** | CLIP-large | **Qwen2-500M (LLM-based)** |
| **Scanning Order** | Rigid (Left-to-Right) | **Dynamic (Causal/Logical)** |
| **Accuracy** | High | **Superior (+3.73% on OmniDocBench)** |
| **Tokens** | ~1156 (Gundam Mode) | **~1120 (More Efficient)** |

---

## Conclusion
DeepSeek-OCR 2 bridges the gap between "seeing" and "reading." By using an LLM-based encoder, it treats visual data with the same logical depth as text, making it one of the most powerful and efficient tools for document digitization and multimodal research available today.

---
*Based on the paper: "DeepSeek-OCR 2: Moving Toward Human-Like Causal Visual Encoding"*