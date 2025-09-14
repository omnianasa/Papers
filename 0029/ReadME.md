# DDIM

## Intro

Imagine you slowly mess up a photo by adding little bits of blur/noise until it looks like static. Many modern diffusion models learn to reverse that process — turning noise back into a clean photo — but they do it in lots and lots of tiny steps, which is slow. DDIMs let you use the same trained network but take *bigger, smarter steps* (even deterministic ones) so you can get great images much faster and treat the initial noise as a real latent you can edit or interpolate.

---

## Plain‑English intuition (no math first)
- Start with a clean image: call it x0.
- Forward process: we add small random noise many times until the image is garbage (pure noise). Think of shaking a box gradually — a little shake each step.
- Original DDPM approach: learn how to undo each tiny shake. The trained model can reverse the process, but must follow all tiny undo steps — slow.
- DDIM idea: the training *only* depends on what the noisy images look like at each step, not on the exact way we shook the box. That means we can choose a different reverse procedure — one that jumps more directly back to the clean image. Same trained model, different (faster) sampling.

Because the DDIM reverse can be deterministic, the initial noise becomes a true latent vector you can play with (blend two noises to morph between images, or reconstruct an image back and forth).

---

## The core stuff
### What you train (same as DDPM)
You train a neural network to **predict the noise** that was added to an image after t steps. Practically, during training you take a clean image, add a known amount of noise, and ask the network: "what noise did I add?" The loss is just mean squared error between the true noise and the predicted noise.

(That’s it for training — no adversarial tricks, no complicated losses.)

### What you change at sampling time (DDIM)
Instead of following the slow, noisy reverse chain, DDIM defines a different step formula — you can pick it so there is **no random noise added** during sampling. That makes the mapping from initial noise to image deterministic and much faster when you skip steps.

---

## Small math 
I keep symbols simple: x0 is the image, xt is the noisy image after t steps, alpha_bar_t is "how much signal remains" after t steps.

Predict x0 from xt using the model's predicted noise:

```math
x0_hat(xt) = ( xt - sqrt(1 - alpha_bar_t) * eps_theta(xt, t) ) / sqrt(alpha_bar_t)
```

Deterministic DDIM step: compute xt-1 from xt and x0_hat:

```math
xt_minus_1 = sqrt(alpha_bar_{t-1}) * x0_hat(xt) + sqrt(1 - alpha_bar_{t-1}) * eps_theta(xt, t)
```

If you prefer some randomness, add a noise term scaled by sigma_t (controlled by a parameter eta). With eta=0 you are deterministic; eta>0 mixes in randomness.

---

## How to actually use it
1. Train as usual (DDPM training): teach eps_theta to predict noise. Save your model and the arrays alpha_bar_t used during training.
2. At sampling time, pick how many steps you want to run (S). S can be much smaller than the training T (e.g., T=1000, S=50).
3. Start from Gaussian noise x_T. For each chosen timestep (descending), apply the DDIM deterministic step above to jump to the next timestep. If you want diversity, set eta>0 and add noise.
4. After the final step, optionally compute x0_hat and use that as your output image.

This is usually 10–50× faster in wall‑clock time and often retains high visual quality.

---

## Practical code snippet
This is a tiny pseudocode sketch you can paste into your project (PyTorch style). It assumes you already have a trained eps_theta model and alpha_bar array.

```python
# xt shape: (batch, channels, height, width)
# sample_ts: list of timesteps descending, e.g. [T, T-k, ..., 1]
def predict_x0(xt, t, eps_pred, alpha_bar):
    sqrt_ab = alpha_bar[t] ** 0.5
    sqrt_1_ab = (1 - alpha_bar[t]) ** 0.5
    return (xt - sqrt_1_ab * eps_pred) / sqrt_ab

def ddim_step(xt, t, t_prev, eps_theta, alpha_bar, eta=0.0):
    eps_pred = eps_theta(xt, t)          # network predicts noise
    x0_hat = predict_x0(xt, t, eps_pred, alpha_bar)

    sqrt_ab_prev = alpha_bar[t_prev] ** 0.5
    sqrt_1_ab_prev = (1 - alpha_bar[t_prev]) ** 0.5

    x_prev = sqrt_ab_prev * x0_hat + sqrt_1_ab_prev * eps_pred

    if eta == 0.0:
        return x_prev
    else:
        # sigma formula is one valid option
        sigma = eta * ((1 - alpha_bar[t_prev]) / (1 - alpha_bar[t])) ** 0.5 * ((1 - alpha_bar[t] / alpha_bar[t_prev]) ** 0.5)
        return x_prev + sigma * torch.randn_like(xt)

# Sampling loop
x = torch.randn(batch_shape)  # x_T
for i in range(len(sample_ts) - 1):
    t = sample_ts[i]
    t_prev = sample_ts[i+1]
    x = ddim_step(x, t, t_prev, eps_theta, alpha_bar, eta=0.0)
# final x0_hat if you want cleaned output
x0 = predict_x0(x, sample_ts[-1], eps_theta(x, sample_ts[-1]), alpha_bar)
```

---

## Practical tips & little secrets
- **Don't retrain**: you get DDIM benefits without changing training. Just swap the sampler.
- **Alpha arrays in FP32**: compute alpha_bar in float32 even if you run the model in fp16 — avoids tiny numerical errors.
- **Timestep choice**: start with uniform spacing when choosing S timesteps. If you get artifacts, use more steps early in the chain.
- **Interpolation trick**: to morph between two images, compute their x_T (forward noise), blend those latents, then decode with deterministic steps — smooth morphs.
- **Tradeoff**: deterministic (eta=0) → reproducible and great for interpolation; stochastic (eta>0) → more variety.

---
