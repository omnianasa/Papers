# AdaGrad Stepsizes: Sharp Convergence over Nonconvex Landscapes

This repository provides an easy-to-understand explanation and mathematical breakdown of the paper **"AdaGrad stepsizes: Sharp convergence over nonconvex landscapes"** by Rachel Ward, Xiaoxia Wu, and Léon Bottou (JMLR 2020).

---

##  Overview

Stochastic Gradient Descent (SGD) is the standard method for training deep learning models. However, plain SGD is highly sensitive to its tuning parameters:
1. The **learning rate (stepsize)**.
2. The unknown **Lipschitz smoothness constant ($L$)** of the loss function.
3. The **noise level ($\sigma$)** of data generation.

If the learning rate is set too high, optimization fluctuates or diverges entirely.

**AdaGrad-Norm** solves this by adjusting its scalar learning rate dynamically based on the norms of past gradients. While adaptive methods work incredibly well in practice, their theoretical properties were historically only proven for convex optimization problems. 

**This paper bridges the gap.** It provides mathematical proof that AdaGrad-Norm guarantees convergence for **smooth, non-convex functions**, automatically tuning itself without requiring prior knowledge of $L$ or $\sigma$.

---

##  The Core Benefits

- **No Fine-Tuning Needed**: It is robust to any choice of initial parameters ($b_0 > 0$ and $\eta > 0$).
- **Sharp Adaptability**: 
  - In a noisy (stochastic) setting, it automatically converges at a rate of $\mathcal{O}\left(\frac{\log(N)}{\sqrt{N}}\right)$.
  - In a noiseless (deterministic batch) setting, it accelerates to the optimal rate of $\mathcal{O}\left(\frac{1}{N}\right)$.

---

##  Mathematical Formulation

### 1. Problem Assumptions
We want to minimize a differentiable function $F: \mathbb{R}^d \rightarrow \mathbb{R}$ where $F^* = \inf_x F(x) > -\infty$. The function must satisfy the **$L$-Lipschitz Smoothness condition**:

$$\|\nabla F(x) - \nabla F(y)\| \le L \|x - y\|, \quad \forall x,y \in \mathbb{R}^d$$

In the stochastic setting, we observe a noisy gradient $G_j$ instead of the exact gradient $\nabla F(x_j)$. The conditions are:
- Unbiased estimator: $\mathbb{E}[G_j] = \nabla F(x_j)$
- Bounded variance: $\mathbb{E}[\|G_j - \nabla F(x_j)\|^2] \le \sigma^2$
- Uniformly bounded exact gradient: $\|\nabla F(x)\|^2 \le \gamma^2$

---

### 2. The AdaGrad-Norm Algorithm

Instead of tracking distinct learning rates for every single weight parameter (like coordinate-based AdaGrad or Adam), **AdaGrad-Norm** tracks a single scalar value $b_j$ representing the cumulative sum of gradient norms.

#### Step-by-Step Update Loop:
For each iteration $j = 1, 2, \dots, N$:

1. Compute the stochastic gradient estimate:
   $$G_{j-1} = G(x_{j-1}, \xi_{j-1})$$

2. Accumulate the squared norm of the direction vector into the denominator state tracker $b_j$:
   $$b_j^2 = b_{j-1}^2 + \|G_{j-1}\|^2$$

3. Update the parameters using the adapted scale:
   $$x_j = x_{j-1} - \frac{\eta}{b_j} G_{j-1}$$

*(Where $\eta > 0$ is a constant scaling factor to balance units, and $b_0 > 0$ is the initial accumulator offset).*

---

## Convergence Performance and Bounds

The paper provides two core theorems verifying the speed of the algorithm.

### Theorem 1: Noisy (Stochastic) Setting
With a sample count $N$ and failure probability bound $\delta$, AdaGrad-Norm achieves an $\epsilon$-approximate stationary point with probability at least $1-\delta$:

$$\min_{l \in [N-1]} \|\nabla F(x_l)\|^2 \le \mathcal{O}\left( \frac{\gamma \left(\sigma + \eta L + \frac{F(x_0) - F^*}{\eta}\right) \log\left(\frac{N\gamma^2}{b_0^2}\right)}{\sqrt{N}} \right)$$

This matches the optimal $\mathcal{O}(1/\sqrt{N})$ rate of fully hand-tuned standard SGD without needing to know $L$ or $\sigma$ ahead of time.

### Theorem 2: Noiseless (Deterministic Batch) Setting
When there is no noise ($\sigma = 0$), the denominator $b_j$ stops accumulating random variations and stabilizes near the true scale of $L$. The absolute convergence rate updates to the absolute optimal mathematical bound:

$$\min_{j \in [N]} \|\nabla F(x_j)\|^2 \le \epsilon$$

The total steps required $N$ scale differently based on the initialization choice:
1. **If $b_0 \ge \eta L$** (Safe Init):
   $$N = 1 + \left\lceil \frac{1}{\epsilon} \left( \frac{4(F(x_0)-F^*)^2}{\eta^2} + \frac{2b_0(F(x_0)-F^*)}{\eta} \right) \right\rceil$$

2. **If $b_0 < \eta L$** (Aggressive Learning Init):
   $$N = 1 + \left\lceil \frac{1}{\epsilon} \left( 2L(F(x_0)-F^*) + \left(\frac{2(F(x_0)-F^*)}{\eta} + \eta L C_{b_0}\right)^2 + (\eta L)^2(1+C_{b_0}) - b_0^2 \right) \right\rceil$$

   *(Where $C_{b_0} = 1 + 2 \log\left(\frac{\eta L}{b_0}\right)$)*

---

##  Proof Intuition: Why is proving this hard?

In normal SGD optimization proofs, the step size rule is fixed or set deterministically. 

In AdaGrad-Norm, the stepsize sequence variable $\frac{\eta}{b_{j+1}}$ is itself a **random variable correlated with current and prior noisy gradients**. This breaks normal expectation formulas:

$$\mathbb{E}\left[\frac{\langle\nabla F_j, \nabla F_j - G_j\rangle}{b_{j+1}}\right] \neq \frac{1}{b_{j+1}} \cdot 0$$

### The Solution:
The authors utilize the **Descent Lemma**:

$$F(x_{j+1}) \le F(x_j) + \langle \nabla F(x_j), x_{j+1} - x_j \rangle + \frac{L}{2}\|x_{j+1} - x_j\|^2$$

They substitute the adaptive adjustment step and introduce a mathematical surrogate frame using Jensen's Inequality to proxy for the unknown value of $\mathbb{E}\left[\frac{1}{b_{j+1}}\right]$:

$$\frac{1}{\sqrt{b_j^2 + \|\nabla F_j\|^2 + \sigma^2}}$$

This mathematical framework enables tracking bounds on the cross-correlations without causing proof divergence.

---

##  Practical Takeaways for Implementation

If you want to use AdaGrad-Norm efficiently based on this paper's insights, apply this parameter setup strategy:

1. **If you know your targeted minimal loss ($F^*$)**: 
   - Set $\eta = F(x_0) - F^*$
   - Initialize $b_0$ to a very small positive number (e.g., $10^{-5}$).
2. **If you don't know $F^*$**:
   - Set $\eta = 1.0$ and initialize $b_0$ to a small number. The algorithm will automatically stabilize itself after a few iterations if the early steps create a large gradient surge.

---
