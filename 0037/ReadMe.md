# Video-to-Video Synthesis

**[Paper](https://arxiv.org/pdf/1808.06601):** *Video-to-Video Synthesis* by Ting-Chun Wang et al., arXiv:1808.06601 (2018).

## Summary
This paper shows a way to turn one video into another video. For example, it can take a video of segmentation maps (labels for each pixel) and make a realistic street scene video. The method makes sure each frame looks real and that the video is smooth over time (frames do not flicker). The model is trained with a conditional GAN (generator and discriminator) and special losses and modules to keep temporal consistency.

## Problem they solve
Given a source video `s` (like segmentation maps or poses), the goal is to create a new video `x˜` that matches how real videos look for the same source. The generated video should follow the input content and be temporally smooth.

They write this goal as wanting the conditional distribution of generated videos to equal the conditional distribution of real videos:
```math
p(\tilde{x}_{1:T}\mid s_{1:T}) = p(x_{1:T}\mid s_{1:T}).
```

## Main idea 
1. Use a *sequential generator* that creates frames one by one. Each new frame depends on a few past frames and the current input.
2. Use *optical flow* to warp pixels from previous frames to the next frame. This keeps things consistent and saves effort.
3. Also synthesize new pixels for parts that were not visible before (occluded areas).
4. Use two discriminators:
   - an **image discriminator** to make single frames look real.
   - a **video discriminator** to make a sequence of frames look temporally real.
5. Train with a mix of adversarial losses, a flow loss (to make predicted flow correct), and feature matching / perceptual losses for stability.

## Key formulas 
**Minimax GAN objective** (train generator `G` and discriminator `D`):
```math
\max_D \min_G \; \mathbb{E}_{x_{1:T},s_{1:T}}[\log D(x_{1:T}, s_{1:T})]
+ \mathbb{E}_{s_{1:T}}[\log(1 - D(G(s_{1:T}), s_{1:T}))]
```

**Markov factorization (sequential generation)**  
They assume each generated frame only depends on the recent L frames:
```math
p(\tilde{x}_{1:T}\mid s_{1:T}) = \prod_{t=1}^T p(\tilde{x}_t \mid \tilde{x}_{t-L}^{t-1}, s_{t-L}^t).
```

**Generator frame model using flow + mask + hallucination**  
Let `F` be the frame-by-frame synthesis function. It blends warped pixels from previous frame(s) with newly synthesized pixels using an occlusion mask `\tilde{m}_t`:
```math
F(\tilde{x}_{t-L}^{t-1}, s_{t-L}^t) = (1 - \tilde{m}_t) \odot \tilde{w}_{t-1}(\tilde{x}_{t-1}) + \tilde{m}_t \odot \tilde{h}_t,
```
where:
- `\odot` is element-wise product,
- `\tilde{w}_{t-1}` is the predicted warping (optical flow) that warps the previous frame,
- `\tilde{h}_t` is the “hallucinated” image for new pixels,
- `\tilde{m}_t` is a soft occlusion mask (values between 0 and 1).

**Flow loss** (make predicted flow and warped pixels correct):
```math
\mathcal{L}_W = \frac{1}{T-1}\sum_{t=1}^{T-1} ( \| \tilde{w}_t - w_t \|_1 + \| \tilde{v}_t(x_t) - x_{t+1} \|_1 ).
```
Here `w_t` is the ground-truth flow and `\tilde{v}_t(x_t)` denotes warping `x_t` by the predicted flow.

**Full learning objective (simple form)**  
They combine image GAN loss, video GAN loss, and flow loss:
```math
\min_F \; ( \max_{D_I} \mathcal{L}_I(F,D_I) + \max_{D_V} \mathcal{L}_V(F,D_V) ) + \lambda_W \mathcal{L}_W(F).
```

## Important modules
- **Optical flow predictor `W`**: predicts how pixels move between frames. It helps warp previous pixels to the new frame.
- **Mask predictor `M`**: predicts a soft mask that says which pixels should be copied (from the warped frame) and which should be newly generated.
- **Hallucination network `H`**: generates new pixel values where warping cannot fill (e.g., newly revealed areas).
- **Generator `F`**: uses `W`, `M`, and `H` to produce the next frame from previous frames and current source input.
- **Image discriminator `D_I`**: judges single frames to be real or fake.
- **Video discriminator `D_V`**: judges short stacks of frames (and their flow) to enforce realistic temporal changes.

## Tricks and design choices 
- They use a **soft occlusion mask** (not binary). This helps to smoothly blend warped pixels and synthesized pixels and reduces flickering.
- They split hallucination into **foreground and background** models when segmentation masks are available. Background usually moves slowly and can be well-handled by warping; foreground often needs new texture generation.
- They use **multiscale, multi-patch discriminators** to avoid mode collapse and to handle different spatial details.
- They train progressively: start with low resolution and short sequences, then increase to higher resolution and longer sequences. This stabilizes training.

## Results 
- The method can generate realistic-looking videos from segmentation maps, poses, sketches, and other input types.
- They show much better temporal consistency and visual quality than strong baselines.
- The model can generate videos at high resolution (e.g., 2K) and for tens of seconds.

## Training tips from the paper
- Use an optimizer like ADAM with a small learning rate and train for many epochs.
- Use feature matching and perceptual (VGG) losses to improve stability.
- Use many GPUs for large resolution training (the authors used many GPUs for 2K outputs).
