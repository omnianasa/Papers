# DeepONet: Learning Nonlinear Operators

This repository contains an architectural overview and performance analysis of **Deep Operator Networks (DeepONets)**, based on the foundational paper:
> *DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators* (Lu Lu, Pengzhan Jin, and George Em Karniadakis, 2020).

---

## 1. DeepONet Architecture & Data Flow

Unlike traditional neural networks that learn mappings between finite-dimensional spaces (vectors to vectors), DeepONets are designed to learn **operators** mappings from an infinite-dimensional input function space to another function space. 

The architecture achieves this by decoupling the input function evaluations from the target evaluation coordinates using two distinct sub-networks:

* **The Branch Network:** Processes the input function $u$. Because continuous functions cannot be fed directly into a neural network, $u$ is evaluated at a discrete, fixed set of $m$ locations $\{x_1, x_2, \dots, x_m\}$, termed **sensors**. The vector $[u(x_1), u(x_2), \dots, u(x_m)]^T \in \mathbb{R}^m$ is mapped to a latent feature space $\mathbb{R}^p$.
* **The Trunk Network:** Processes the target domain coordinate $y \in \mathbb{R}^d$ where the output function is to be evaluated. It maps $y$ to the same latent feature space $\mathbb{R}^p$.

### The Merging Operation
The final output of the operator, $G(u)(y)$, is computed by taking the dot product of the latent vectors from both sub-networks, augmented by an optional trainable bias term $b_0$:

$$G(u)(y) \approx \sum_{k=1}^{p} b_k t_k + b_0$$



---

## 2. Stacked vs. Unstacked DeepONets

The paper introduces two structural variants for scaling the branch components to match the output dimension $p$ of the trunk net:

### Stacked DeepONet
* **Structure:** Utilizes $p$ separate, parallel branch networks. Each individual branch net takes the sensor vector as input and outputs a single scalar $b_k$.
* **Drawbacks:** Highly computationally expensive and memory-intensive due to duplicating the network architecture $p$ times.

### Unstacked DeepONet
* **Structure:** Merges the parallel architectures into a single, unified branch network. This network takes the sensor vector as input and outputs the entire $p$-dimensional vector $[b_1, b_2, \dots, b_p]^T$ simultaneously from its final layer.
* **Advantages:** Significantly fewer parameters, faster training speeds, lower memory overhead, and a much tighter correlation between training and test error (lower generalization error).

---

## 3. Generalization Superiority over FNNs

When solving operator learning problems using standard Fully Connected Networks (FNNs), the inputs must be concatenated into a single flat vector: $[u(x_1), u(x_2), \dots, u(x_m), y]^T$. 

DeepONets achieve superior generalization compared to this baseline due to their **strong inductive bias**:

1.  **Structure-Property Alignment:** The target operator output $G(u)(y)$ is fundamentally a function of $y$ conditioned on the global state of $u$. FNNs treat the coordinate $y$ and the sensor evaluations $u(x_i)$ indiscriminately within the same hidden layers, failing to isolate this relationship.
2.  **Mathematical Factorization:** DeepONets explicitly separate cross-dependencies by processing $u$ and $y$ through isolated sub-networks before combining them via an inner product. This architectural constraint acts as a regularizer, preventing overfitting to non-physical combinations of sensors and coordinates.

---

## 4. Key Scaling Trends and Convergence Properties

### Number of Sensors ($m$)
* For smooth input functions, the error decays exponentially ($\text{MSE} \propto 4.6^{-m}$) until hitting a saturation threshold (typically around $m \approx 10$ for simple 1D systems), after which additional sensors yield diminishing returns.
* The required number of sensors scales with domain time horizon $T$ and the input function's smoothness (length-scale $l$ from a Gaussian Random Field):
    $$m \propto \sqrt{T} \quad \text{and} \quad m \propto l^{-1}$$
* When complexity increases (e.g., higher numbers of Chebyshev polynomial bases), $m$ must scale according to $\text{Bases} \propto \sqrt{m}$ to accurately capture the function topology.

### Data Convergence Rates
The generalization error ($\text{MSE}_{\text{gen}}$) and test error ($\text{MSE}_{\text{test}}$) exhibit distinct regimes based on the training dataset size $x$:

| Dataset Size | $\text{MSE}_{\text{test}}$ Convergence Rate | $\text{MSE}_{\text{gen}}$ Convergence Rate |
| :--- | :--- | :--- |
| **Small Datasets ($x < 10^4$)** | Exponential ($\propto e^{-x/2000}$) | Exponential ($\propto e^{-x/2000}$) |
| **Large Datasets** | Polynomial ($\propto x^{-0.5}$) | Polynomial ($\propto x^{-1}$) |

> **Note:** The large-data generalization convergence rate of $\propto x^{-1}$ is notably faster than the classical $\propto x^{-0.5}$ rate typically found in statistical learning theory, verifying DeepONet's data efficiency.