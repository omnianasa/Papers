
# ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION

## Summary

- **What is Adam?**  
  Adam is an algorithm for training models that adapts the step size for each parameter using running averages of the gradient (first moment) and the squared gradient (second moment). It works well in practice, needs little tuning, and is efficient.

- **When to use it?**  
  Use Adam when you want a good default optimizer that usually converges quickly and is robust for noisy or sparse gradients 

- **Common default hyper-parameters:**  
  - learning rate `α = 0.001`  
  - `β1 = 0.9` (decay rate for first moment)  
  - `β2 = 0.999` (decay rate for second moment)  
  - `ε = 1e-8` (small number to avoid division by zero)

---

## Algorithm (mathematics)

The core steps of Adam are:

```math function
# gradients at time t
g_t = ∇_θ f_t(θ_{t-1})

# biased first moment estimate (exponential moving average of gradients)
m_t = β1 · m_{t-1} + (1 − β1) · g_t

# biased second moment estimate (exponential moving average of squared gradients)
v_t = β2 · v_{t-1} + (1 − β2) · g_t^2

# bias-corrected estimates
\hat{m}_t = m_t / (1 − β1^t)
\hat{v}_t = v_t / (1 − β2^t)

# parameter update
θ_t = θ_{t-1} − α · \hat{m}_t / (sqrt(\hat{v}_t) + ε)
```

---

## Simple explanation of the math (step by step)

1. **Gradient `g_t`**  
   At each iteration `t` we compute the gradient `g_t` (as in standard stochastic gradient descent).

2. **Moving averages `m_t` and `v_t`**  
   - `m_t` is an exponential moving average of gradients. It tracks the *direction* (the mean) of recent gradients like momentum.  
   - `v_t` is an exponential moving average of squared gradients. It tracks the *magnitude* (uncentered variance) of recent gradients similar to RMSProp.

3. **Biased averages and correction**  
   Because `m_0` and `v_0` are initialized at zero, early `m_t` and `v_t` are biased toward zero. To correct this, Adam divides by `(1 − β1^t)` and `(1 − β2^t)` respectively, producing `\hat{m}_t` and `\hat{v}_t`. This makes the estimates unbiased (or less biased), especially important at the beginning of training.

4. **Scaling the step**  
   The parameter update uses `\hat{m}_t` (direction) divided by `sqrt(\hat{v}_t)` (magnitude). This rescales each parameter's update by its estimated variance: parameters with large recent squared gradients get smaller steps, and small-gradient parameters get relatively larger steps. The small `ε` prevents division by zero.

5. **Intuition**  
   - `m_t` points in the direction we should move (like momentum).  
   - `v_t` tells how noisy or large the gradients are for each parameter (so we scale the step down if the gradient has large variance).  
   - Combining them yields an update that is both direction-aware and size-adaptive for each parameter.

---

## Why bias correction matters (simple view)

Because `m_0 = 0` and `v_0 = 0`, the moving averages start small and are biased toward zero until enough steps have passed. The correction `/(1 − β^t)` compensates for that early shrinkage so the optimizer doesn't under-react in the first iterations.

---

## Practical tips

- Keep the default hyper-parameters (α=0.001, β1=0.9, β2=0.999, ε=1e-8) as a good starting point.  
- For some problems, reducing α or switching to weight decay (instead of L2) helps generalization.  
- Adam is often a great default; for final fine-tuning or best generalization, sometimes SGD with momentum can outperform Adam.

---

## Reference

Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.
(https://arxiv.org/abs/1412.6980)

---
