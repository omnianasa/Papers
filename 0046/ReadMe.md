# Improved Denoising Diffusion Probabilistic Models 

This repository implements the improvements suggested in "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).

## Key Technical Concepts

### 1. Learned Variances
We parameterize the reverse process variance $\Sigma_\theta(x_t, t)$ as an interpolation:
$$\Sigma_\theta(x_t, t) = \exp(v \log \beta_t + (1 - v) \log \tilde{\beta}_t)$$
This allows for competitive log-likelihoods and much faster sampling (up to 10x speedup).

### 2. Cosine Noise Schedule
Unlike the linear schedule, the cosine schedule prevents the sudden loss of information:
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \text{ where } f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

### 3. Hybrid Objective
The model is trained using a weighted loss to balance visual fidelity and distribution coverage:
$$L_{hybrid} = L_{simple} + 0.001 \cdot L_{vlb}$$

## Performance
- **Faster Sampling:** High-quality results in 50-100 steps.
- **Better Precision:** Achieves better log-likelihoods on ImageNet and CIFAR-10.
- **Improved Recall:** Better coverage of the data distribution compared to GANs.

## Reference
- **Paper:** [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- **Official Code:** [openai/improved-diffusion](https://github.com/openai/improved-diffusion)