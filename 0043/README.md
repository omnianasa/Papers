# Lookahead Optimizer: k steps forward, 1 step back

## 1. Overview

This paper proposes a new optimization algorithm called Lookahead, introduced by Michael R. Zhang, James Lucas, Geoffrey Hinton, and JimmyBa (arXiv:1907.08610).

The idea: instead of just using one optimizer like SGD or Adam, Lookahead wraps around them and improves their stability and
performance by maintaining two sets of weights fast and slow


## 2. Core Idea

### Fast and Slow Weights

-   **Slow weights (φ)**: represent long-term memory of the model.
-   **Fast weights (θ)**: are updated quickly by an inner optimizer (like SGD or Adam).

After a few inner updates (say `k` steps), the slow weights are moved a little bit toward the new fast weights.

### Update Equation

``` math
\phi_{t+1} = \phi_t + alpha (   heta_{t,k} - \phi_t)
```

where: - ( `\phi`{=tex}*t ): current slow weights - ( heta*{t,k} ): fast
weights after k steps - ( alpha ): the interpolation factor (0 \< α ≤ 1)

This means: take a small step (α) from slow weights toward the final
fast weights after `k` steps.

------------------------------------------------------------------------

## 3. Step-by-Step Algorithm

1.  Initialize **slow weights** φ₀.

2.  Copy them into **fast weights** θ.

3.  Run your inner optimizer for `k` updates:

    ``` math
     heta_{t,i} =    heta_{t,i-1} + A(L,     heta_{t,i-1}, d)
    ```

    where `A` is your optimizer (e.g., SGD, Adam), and `d` is a
    mini-batch of data.

4.  After k steps, update slow weights:

    ``` math
    \phi_t = (1 - alpha)\phi_{t-1} + alpha   heta_{t,k}
    ```

5.  Reset fast weights to the updated slow weights, then repeat.

------------------------------------------------------------------------

## 4. Mathematical Intuition

The slow weights act like an **exponential moving average (EMA)** of
recent fast weights:

``` math
\phi_{t+1} = alpha  heta_{t,k} + (1-alpha)\phi_t
```

Expanding recursively gives:

``` math
\phi_{t+1} = alpha  heta_{t,k} + alpha(1-alpha) heta_{t-1,k} + ... + (1-alpha)^t\phi_0
```

So newer fast weights have more influence, and older ones fade out
smoothly.


## 5. Variance Reduction

In a simple noisy quadratic model, Lookahead reduces the steady-state variance compared to standard SGD.

For SGD:

``` math
V^*_{SGD} = rac{\gamma}{2}A^{-2}\Sigma^2(I - (I - \gamma A)^2)^{-1}
```

For Lookahead:

``` math
V^*_{LA} = rac{alpha^2 (I - (I - \gamma A)^{2k})}
{alpha^2 (I - (I - \gamma A)^{2k}) + 2alpha(1-alpha)(I - (I - \gamma A)^k)} V^*_{SGD}
```

Since the ratio term is less than 1, Lookahead always has **lower
variance**.

---

## 6. Typical Hyperparameters

-   ( k = 5 ): number of inner steps\
-   ( alpha = 0.5 ): update rate for slow weights\
-   Inner optimizer: SGD or Adam\
-   Works well for CIFAR, ImageNet, NLP tasks, etc.


## 7. When to Use

Use Lookahead when: 
- Training deep networks with unstable
convergence. 
- You want smoother updates or faster convergence. 
- You already use Adam/SGD but want an easy stability boost.


**Paper**: [Lookahead Optimizer: k steps forward, 1 step
back](https://arxiv.org/pdf/1907.08610)\
**Authors**: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba\
**Year**: 2019
