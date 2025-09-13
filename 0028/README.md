# Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel, 2020)

## Intro

Diffusion models learn to generate images by **teaching a neural network how to remove noise** that was gradually added to real images. Training is easy (predict the added noise), sampling is iterative (start from pure noise and denoise step-by-step), and the approach produces very high-quality images and interesting compression properties.

---

## High level intuitive story

1. **Forward (noise) process**: start with a real image and gradually add small amounts of Gaussian noise for $T$ steps until the image becomes effectively pure Gaussian noise. This process is *fixed* and known.
2. **Reverse (denoising) process**: learn a neural network that, given a noisy image at timestep $t$, predicts how to step back to $t-1$ — i.e., how to *remove* noise. Repeating this learned step $T$ times (starting from pure noise) yields a generated image.
3. **Training trick**: instead of asking the network to output a denoised image directly, train it to **predict the noise** that was added. That MSE objective is simple and works extremely well.

This simple idea connects to older methods (score matching, Langevin dynamics) and yields excellent sample quality.

---

## Notation quick-reference

* $x_0$: real data (an image).
* $x_t$: noisy version of $x_0$ at timestep $t$ (0 ≤ t ≤ T).
* $q(\cdot)$: the **forward** (fixed) noising process.
* $p_\theta(\cdot)$: the **reverse** (learned) denoising process parameterized by neural net weights $\theta$.
* $\beta_t$: noise variance at forward step $t$.
* $\alpha_t := 1-\beta_t$, $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$.
* $\epsilon$: standard Gaussian noise.

---

## The math (step-by-step, explained)

### 1) Forward (diffusion) process — closed form

The forward process adds Gaussian noise in small increments:

$$
q(x_t\mid x_{t-1}) = \mathcal{N}\big(x_t;\, \sqrt{\alpha_t} x_{t-1},\, \beta_t I\big)
$$

Because this is linear Gaussian, we can directly sample $x_t$ from $x_0$:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \qquad \epsilon\sim\mathcal{N}(0,I)
$$

This formula is useful: it lets us produce a noisy example at any timestep $t$ in one shot.

### 2) Reverse process — learned Gaussian

The reverse (sampling) model is also Gaussian:

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t,t),\, \sigma_t^2 I)
$$

We must design $\mu_\theta$ (and optionally $\sigma_t$). The paper shows a powerful parameterization: make the network predict the noise $\epsilon$ used in the forward process.

### 3) Predicting noise: reparameterize and train

From the forward closed form, we know the relationship between $x_t$ and $x_0$:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon.
$$

If a network $\epsilon_\theta(x_t,t)$ can predict $\epsilon$, we can reconstruct a denoised mean estimate and then sample $x_{t-1}$ from a Gaussian with that mean. Practically, the training objective used by the paper is the **simple MSE between true and predicted noise**:

$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t\sim\text{Unif}[1..T],\, x_0,\,\epsilon} \big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2.
$$

Implementation-wise: sample a random $t$, construct $x_t$ via the closed form, and update $\theta$ by minimizing squared error. This is Algorithm 1 in the paper.

### 4) Sampling (Algorithm 2, step-by-step)

Start from $x_T \sim \mathcal{N}(0,I)$. For t = T down to 1:

1. Compute $\epsilon_\theta(x_t,t)$.
2. Form the predicted mean for $x_{t-1}$:

$$
\tilde{\mu}_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\, \epsilon_\theta(x_t,t)\Big).
$$

3. Sample $x_{t-1}$ = $\tilde{\mu}_\theta + \sigma_t z$ where $z\sim \mathcal{N}(0,I)$ (and $\sigma_t$ is a small known constant; often chosen related to $\beta_t$).

After repeating, $x_0$ is the generated image.

### 5) Relationship to score matching / Langevin dynamics

The parameterization that predicts $\epsilon$ makes the objective look like denoising score matching across multiple noise levels. Sampling (repeated denoising steps) resembles an annealed Langevin dynamics with a learned score (gradient) field. This connection explains why the MSE noise-prediction objective works well.

---

## Practical implementation notes (what the paper used)

* **T** (number of timesteps): 1000.
* **Beta schedule**: linear from $\beta_1=1e-4$ to $\beta_T=0.02$ for images scaled to $[-1,1]$.
* **Network**: U-Net backbone with time conditioning (sinusoidal embeddings) and GroupNorm; self-attention at intermediate resolution (paper used at 16×16).
* **Parameterization**: model predicts $\epsilon$. Fix $\sigma_t$ to simple constants (paper discusses a couple options).
* **Training objective**: simplified MSE noise-prediction loss (works best for sample quality).
* **Data scaling**: images \[0..255] → linearly scaled to \[-1,1].

These settings reproduced the results in the paper and are a good starting point for new datasets.

---

## Results summary (what the paper reported)

* On unconditional CIFAR-10, the model achieved state-of-the-art sample quality for likelihood-based models at the time: Inception Score ≈ 9.46, FID ≈ 3.17 (paper reports both train-set and test-set calculations and shows competitive results on 256×256 LSUN).
* The paper also shows that diffusion models are good **lossy compressors**: most of their lossless code length is spent on imperceptible image details; progressive decoding reveals coarse structure first then details.

(See the original paper for full tables and images.)

---

## Reproducible checklist (quick)

* [ ] Use T = 1000 steps.
* [ ] Use linear beta schedule from 1e-4 to 0.02 (scale images to \[-1,1]).
* [ ] Implement closed-form sampling of x\_t from x\_0 for training (no need to simulate entire forward chain).
* [ ] Train a U-Net that consumes x\_t and timestep t (sinusoidal embedding).
* [ ] Optimize MSE between true $\epsilon$ and predicted $\epsilon_\theta(x_t,t)$.
* [ ] Use mixed precision + Adam optimizers and moderate batch sizes for stable training.

---

## Pseudocode: training and sampling

**Training (core loop)**

```
for minibatch x0 in dataset:
    t = UniformSample(1..T)
    eps = Normal(0,I)
    xt = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps
    pred = eps_theta(xt, t)
    loss = MSE(pred, eps)
    backprop and update theta
```

**Sampling**

```
x_T ~ Normal(0,I)
for t = T..1:
    pred_eps = eps_theta(x_t, t)
    mean = (1/sqrt(alpha_t)) * (x_t - ((1-alpha_t)/sqrt(1-alpha_bar_t)) * pred_eps)
    if t > 1:
        x_{t-1} = mean + sigma_t * Normal(0,I)
    else:
        x_{t-1} = mean
return x_0
```

---

## Tips, common gotchas, and ablations (practical experience)

* Predicting noise ($\epsilon$) is simpler and often improves sample quality over directly predicting x0 or the mean.
* Learning reverse variances (instead of fixing them) can be unstable; the paper fixes them to small time-dependent constants.
* The simplified MSE objective usually yields **better samples** even if it is not the tightest variational bound.
* If training is unstable, reduce learning rate, use gradient clipping, or simplify the network.


