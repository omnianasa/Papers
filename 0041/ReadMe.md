
# Self-Attention Generative Adversarial Networks (SAGAN)

**Paper:** *Self-Attention Generative Adversarial Networks*  
**Authors:** Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena  
**arXiv:** arXiv:1805.08318 (PDF) 

---

## Overview

SAGAN augments convolutional GANs with self-attention modules so that both the generator and discriminator can model long-range, non-local dependencies across spatial positions in feature maps. It also applies spectral normalization to the generator (in addition to the discriminator) and uses an imbalanced learning-rate scheme (TTUR) to stabilize training and improve sample quality. 

---

## Why this matters 

Convolutions are local: a pixel's generation depends mainly on nearby receptive fields. This makes modeling long-range geometric relationships (e.g., consistent limbs of an animal, spatially separated object parts) hard without stacking many layers. Self-attention directly allows every position to attend to every other position, enabling the model to coordinate distant regions when synthesizing details. The authors show this leads to qualitatively and quantitatively better images on ImageNet. 

---

## Key contributions

- Introduces a self-attention (non-local) module adapted for GANs so the model can reason about global context when generating images. 
- Applies spectral normalization to both the generator and discriminator to stabilize training and improve conditioning.
- Uses TTUR (separate learning rates for G and D) to compensate for slowed discriminator learning caused by regularization; this enables stable 1:1 update schedules.
- Shows large empirical gains on ImageNet (detailed below). 

---

## Mathematical core — the self-attention block

The module is an adaptation of the non-local / self-attention operation. Given input features at one layer:
- Let `x ∈ R^{C×N}` be the feature map reshaped so N is the number of spatial positions (e.g., H×W) and C is the channel dimension.
- The module computes query, key and value embeddings via learned 1×1 convolutions (linear maps): `f(x) = W_f x`, `g(x) = W_g x`, `h(x) = W_h x`.
- Attention weights and output are computed as:

```math
s_{ij} = f(x_i)^\top g(x_j)
\beta_{j,i} = \frac{\exp(s_{ij})}{\sum_{i=1}^N \exp(s_{ij})}
o_j = \sum_{i=1}^N \beta_{j,i} \, v(x_i)
```

Where `v(x_i) = W_v x_i` (a value projection). Finally, a learnable scalar `\gamma` scales the attention output and we add a residual connection back to the input:

```math
y_j = \gamma \, o_j + x_j
```

Important implementation choices from the paper:
- The intermediate channel size C̄ for `f`, `g`, `h` is set to `C/8` (k=8) for memory efficiency (authors report little performance drop when reducing channels). 
- Initialize `\gamma = 0` so the network initially relies on local cues and gradually learns to use non-local attention. 

(Equations and notation follow the paper's Section 3.)

---

## Discriminator & Generator — architecture notes

- SAGAN inserts self-attention **blocks at middle-to-high-level feature maps** (not at the earliest low-level conv layers). This placement helps the model attend over semantic features rather than raw pixels.
- Uses **conditional batch normalization** (cBN) in the generator for class-conditional generation and the **projection discriminator** technique for conditioning in D (as in Miyato & Koyama). 
- Spectral normalization is applied to layers in **both** the generator and discriminator by default. This constrains layer spectral norms and stabilizes training. 

---

## Loss & training recipes (practical)

- **Adversarial loss:** SAGAN uses the **hinge loss** variant for both G and D. In the paper they write:

```math
L_D = - \mathbb{E}_{x,y \sim p_{\text{data}}}[\min(0, -1 + D(x, y))]
      - \mathbb{E}_{z \sim p_z, y \sim p_{\text{data}}}[\min(0, -1 - D(G(z), y))]

L_G = - \mathbb{E}_{z \sim p_z, y \sim p_{\text{data}}}[ D(G(z), y) ]
```

(See Section 4 and Eq. (4) in the paper.) 

- **Optimizer & hyperparameters (paper defaults):** Adam with `β1 = 0`, `β2 = 0.9`. By default they set the discriminator learning rate to `lr_D = 4e-4` and the generator learning rate to `lr_G = 1e-4`. Models generate `128×128` images in the experiments. 

- **Two-timescale update rule (TTUR):** Use different learning rates for G and D so that with spectral normalization you can train with a balanced 1:1 step ratio while avoiding instability. Using TTUR in combination with SN lets the model avoid needing extra D updates per G step. 

---

## Empirical results (high level)

- The paper reports sizable improvements on ImageNet (128×128): Inception Score and FID are significantly improved compared to earlier spectral-normalized baselines. Their best reported improvement increases the Inception score and lowers FID (see paper for exact numbers and tables). 

- They also provide visualizations of attention maps demonstrating that the network attends to semantically meaningful regions (object parts) rather than arbitrary local neighborhoods.

---

## Implementation tips & common gotchas

- **Where to insert attention:** add self-attention blocks at mid-to-high feature map resolutions (e.g., after several upsampling steps in G and after several downsampling steps in D). The authors found this placement most beneficial. 
- **Memory:** attention scales with N^2 (N=H×W). To reduce memory, reduce the channel dimension for f/g/h (paper uses C/8) or insert attention only at a few layers.
- **Initialization:** initialize γ = 0 so attention is introduced smoothly. 
- **Stability:** pair spectral normalization on both G and D with TTUR (separate learning rates) to stabilize training and allow 1:1 updates. 

---

## Pseudocode — self-attention block (PyTorch-like pseudocode)

```python
# x: (B, C, H, W)
B, C, H, W = x.shape
N = H * W
x_flat = x.view(B, C, N)               # reshape to (B, C, N)

f = conv1x1_f(x)                       # (B, C//8, H, W) -> (B, C//8, N)
g = conv1x1_g(x)                       # (B, C//8, N)
h = conv1x1_h(x)                       # (B, C//8, N)  or (B, C//8, N) depending on implementation

# compute attention
s = torch.bmm(f.transpose(1,2), g)    # (B, N, N) similarity
beta = softmax(s, dim=-1)             # attention weights per row
o = torch.bmm(h, beta.transpose(1,2)) # (B, C', N) then project to (B, C, N) via W_v

o = conv1x1_v(o)                       # map back to C channels
y = gamma * o + x_flat                 # residual connection (gamma scalar per channel or global)
y = y.view(B, C, H, W)
```

---

## Where to find code & checkpoints

The authors linked to an official code repository in the paper (see references). Many community implementations exist (PyTorch/TensorFlow). See paper references for the official repo.

---

## Quick summary 

SAGAN augments convolutional GANs with a lightweight self-attention module that enables long-range dependency modeling, applies spectral normalization to both generator and discriminator, and uses TTUR to stabilize training—together these changes lead to improved image quality and diversity on ImageNet-scale experiments. See the paper for full results, ablations, and attention visualizations.

---
