# Auto-Encoding Variational Bayes 

**Paper:** *Auto-Encoding Variational Bayes*  
**Authors:** Diederik P. Kingma & Max Welling  
**ArXiv:** 1312.6114

---

## summary

This paper introduces a practical way to do approximate Bayesian inference and learning for models with continuous latent variables — especially when the true posterior is intractable. The key ideas are:

- Use a learnable *recognition model* (also called encoder) \(q_\phi(z \mid x)\) to approximate the true posterior \(p_\theta(z \mid x)\).
- Reparameterize the stochastic sampling from \(q_\phi\) to push randomness into an auxiliary variable. This makes the objective differentiable w.r.t. the encoder parameters \(\phi\). This is the famous reparameterization trick.
- Optimize a stochastic estimate of the variational lower bound (ELBO) using standard stochastic gradient methods (mini-batches, Adam, etc.). When the encoder/decoder are neural networks, this model is often called a Variational Autoencoder (VAE)

---

## Notation

- \(x\) — observed data (one datapoint).
- \(z\) — continuous latent variable (per datapoint).
- \(\theta\) — parameters of the generative model \(p_\theta(x \mid z)\) and the prior \(p_\theta(z)\).
- \(\phi\) — parameters of the recognition/encoder model \(q_\phi(z \mid x)\).
- \(\mathcal{L}(\theta, \phi; x)\) — the variational lower bound (ELBO) for datapoint \(x\).

---

## Key equations 

### ELBO (variational lower bound)
``` math
\log p_\theta(x) \ge \mathcal{L}(\theta, \phi; x)
= \mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x, z) - \log q_\phi(z\mid x)\big]
```

A common alternative form:
``` math
\mathcal{L}(\theta, \phi; x) = -\mathrm{KL}\big(q_\phi(z\mid x) \,||\, p_\theta(z)\big)
+ \mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]
```

### Reparameterization trick (for Gaussian \(q_\phi\))
If \(q_\phi(z\mid x) = \mathcal{N}(z; \mu_\phi(x), \operatorname{diag}(\sigma^2_\phi(x)))\), then sample:
``` math
\epsilon \sim \mathcal{N}(0, I), \qquad
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon.
```
With this, expectations over \(q_\phi(z\mid x)\) can be written as expectations over \(\epsilon\sim p(\epsilon)\) and the path from \(\phi\) to the sample \(z\) is differentiable.

### Monte Carlo SGVB estimator (single datapoint)
Using \(L\) samples of \(\epsilon\) we estimate the ELBO as:
``` math
\widehat{\mathcal{L}}(\theta, \phi; x)
= \frac{1}{L}\sum_{l=1}^{L} \big[ \log p_\theta(x, z^{(l)}) - \log q_\phi(z^{(l)}\mid x) \big],
\quad z^{(l)} = g_\phi(\epsilon^{(l)}, x).
```
Usually \(L=1\) works well in practice.

---

## Worked example — simple 1D Gaussian VAE (math)

Consider 1-dimensional data \(x\in\mathbb{R}\) and 1-dimensional latent \(z\in\mathbb{R}\).

Choose:
- Prior: \(p(z)=\mathcal{N}(z; 0, 1)\).
- Encoder (approx. posterior): \(q_\phi(z\mid x)=\mathcal{N}(z; \mu_\phi(x), \sigma^2_\phi(x))\).
- Decoder (likelihood): \(p_\theta(x\mid z)=\mathcal{N}(x; f_\theta(z), \sigma_x^2)\) where \(f_\theta(z)\) is a small neural network (or linear function).

**ELBO per datapoint \(x\):**
``` math
\mathcal{L}(\theta,\phi;x)
= -\mathrm{KL}\big(\mathcal{N}(\mu_\phi,\sigma^2_\phi)\,\|\,\mathcal{N}(0,1)\big)
+ \mathbb{E}_{q_\phi(z\mid x)}\big[\log \mathcal{N}(x; f_\theta(z), \sigma_x^2)\big].
```

The KL between two univariate Gaussians has a closed form:
``` math
\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\big)
= \frac{1}{2}\big( \mu^2 + \sigma^2 - 1 - \log \sigma^2 \big).
```

The reconstruction term is
``` math
\mathbb{E}_{q_\phi(z\mid x)}\big[\log \mathcal{N}(x; f_\theta(z), \sigma_x^2)\big]
= -\frac{1}{2\sigma_x^2}\mathbb{E}_{q_\phi}\big[(x - f_\theta(z))^2\big] + \text{const}.
```

**Using the reparameterization** with \(z=\mu_\phi+\sigma_\phi\epsilon\), we estimate the expectation by sampling \(\epsilon\sim\mathcal{N}(0,1)\) and computing \(z\) and \(f_\theta(z)\). Then the loss to minimize (negative ELBO) is:
``` math
\mathcal{L}_{\text{loss}}(x)
= \mathrm{KL}\big(\mathcal{N}(\mu_\phi,\sigma^2_\phi)\,\|\,\mathcal{N}(0,1)\big)
+ \frac{1}{2\sigma_x^2}(x - f_\theta(z))^2 + \text{const}.
```

This yields a straightforward objective you can backprop through.

---

## Practical algorithm (AEVB / VAE) — pseudocode

```
for minibatch X_b:
    for each x in X_b:
        compute encoder outputs μφ(x), σφ(x)
        sample ε ~ N(0, I)
        z = μφ(x) + σφ(x) * ε
        compute reconstruction log pθ(x | z)
        compute KL( qφ(z|x) || p(z) ) (closed form if Gaussians)
    average ELBO over batch
    compute gradients ∇_{θ,φ} of -ELBO
    update parameters with Adam (or SGD)
```

Notes:
- Use L=1 sample per datapoint in the minibatch for efficiency.
- For binary images, use Bernoulli decoder with cross-entropy reconstruction loss.
- For real-valued outputs, use Gaussian decoder (MSE style reconstruction).

---

## Intuition & why it works 

- The ELBO trades off reconstruction accuracy and how close the approximate posterior is to the prior.
- The reparameterization trick moves randomness to \(\epsilon\) so gradients flow through the encoder network rather than through a sampling operation.
- Learning a recognition model (encoder) amortizes the cost of posterior inference across the dataset — no per-datapoint iterative inference is needed at test time.

---

## Short example: one-step numeric (toy)

Suppose \(x=1.0\), single-dim, and let the encoder outputs for this x be:
- \(\mu_\phi(x)=0.2\), \(\sigma_\phi(x)=0.5\).
Let decoder be \(f_\theta(z)=z\) and \(\sigma_x^2=0.1\).

Compute KL:
``` math
\mathrm{KL} = \frac{1}{2}\big(0.2^2 + 0.5^2 -1 - \log(0.5^2)\big)
= \frac{1}{2}(0.04 + 0.25 -1 - \log 0.25) \approx 0.5( -0.71 + 1.386 ) \approx 0.338.
```

Sample ε = 0.3 → z = 0.2 + 0.5 * 0.3 = 0.35.

Reconstruction squared error: (x - fθ(z))^2 = (1.0 - 0.35)^2 = 0.4225.

Reconstruction term (negative log-likelihood part, ignoring constants):
``` math
\frac{1}{2\sigma_x^2} (x - f_\theta(z))^2 = \frac{1}{2*0.1} * 0.4225 = 2.1125.
```

Total (negative ELBO) approximated:
``` math
\text{loss} \approx KL + reconstruction = 0.338 + 2.1125 \approx 2.4505.
```

This is the scalar objective you would backprop to update both encoder and decoder.

---

