# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

Batch Normalization (BN) is a simple layer you add to a neural network that normalizes each activation over the training mini-batch (to zero mean and unit variance), then learns a small scale & shift — this speeds up training, lets you use higher learning rates, and often acts as a regularizer.

---

## 1- What problem does Batch Normalization solve?
When you train deep networks, the distribution of inputs to each layer keeps changing as earlier layers learn. This is called internal covariate shift 
Because later layers must constantly adapt to the changing input distribution, training slows and can require careful initialization and small learning rates. BN reduces this by fixing (approximately) the mean and variance of layer inputs during training.

---

## 2- The BN transform (forward pass)
For a single scalar activation \(x\) (one dimension of a layer output), BN does these steps on a mini-batch \(B = \{x_1, x_2, \dots, x_m\}\):

1. Compute mini-batch mean:
```math
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i
```

2. Compute mini-batch variance:
```math
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2
```

3. Normalize each activation (add small \(\epsilon\) for numerical stability):
```math
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
```

4. Scale and shift (learnable parameters \(\gamma\) and \(\beta\) per activation dimension):
```math
y_i = \gamma \,\hat{x}_i + \beta
```

---

## 3- Why include \(\gamma\) and \(\beta\)?
If we only normalized to zero mean and unit variance, we might remove the layer’s ability to represent the identity transformation or other useful scalings.  

So BN learns \(\gamma\) (scale) and \(\beta\) (shift) to let the network recover any needed representational power:

```math
y = \gamma \hat{x} + \beta
```

---

## 4- Training vs inference
- Training: use mini-batch mean \(\mu_B\) and variance \(\sigma_B^2\).  
- Inference (test time): use moving averages of the means and variances collected during training.  

---

## 5- Concrete small numeric example
Mini-batch \(B = \{1.0,\; 2.0,\; 3.0\}\).  
Let \(\epsilon = 10^{-5}\). Choose \(\gamma = 2.0\), \(\beta = 0.5\).

1. Mean:
```math
\mu_B = \frac{1+2+3}{3} = 2.0
```

2. Variance:
```math
\sigma_B^2 = \frac{(1-2)^2+(2-2)^2+(3-2)^2}{3} = \frac{2}{3} \approx 0.6667
```

3. Normalize:
- For \(x_1=1\):
```math
\hat{x}_1 = \frac{1-2}{\sqrt{0.6667 + 10^{-5}}} \approx -1.2247
```
- For \(x_2=2\):
```math
\hat{x}_2 = 0
```
- For \(x_3=3\):
```math
\hat{x}_3 = \frac{1}{\sqrt{0.6667 + 10^{-5}}} \approx +1.2247
```

4. Scale & shift:
- \(y_1 = 2.0 \cdot (-1.2247) + 0.5 \approx -1.95\)  
- \(y_2 = 2.0 \cdot 0 + 0.5 = 0.5\)  
- \(y_3 = 2.0 \cdot 1.2247 + 0.5 \approx 2.95\)  

Final outputs ≈ **{-1.95, 0.5, 2.95}**

---

## 6- Practical notes
- Where to place BN: usually before the nonlinearity (`Conv -> BN -> ReLU`).  
- Batch size matters: very small batches give noisy mean/variance.  
- Higher learning rates: BN allows larger learning rates and less careful initialization.  
- Regularization: BN acts like a regularizer (sometimes you can reduce Dropout).  
- Implementation: use built-in layers in PyTorch/TensorFlow/Keras — they handle stats & backprop automatically.

---

## 7- Pseudocode for training with BN
1. For each BN layer, compute \(\mu_B,\sigma_B^2\).  
2. Normalize: \(\hat{x} = (x - \mu_B) / \sqrt{\sigma_B^2 + \epsilon}\).  
3. Transform: \(y = \gamma \hat{x} + \beta\).  
4. Update weights + \(\gamma,\beta\) with your optimizer.  
5. Track running averages of mean/variance.  
6. At inference, use those running stats.

---

## 8- Main takeaways from the paper
- BN speeds up training and allows higher learning rates.  
- BN makes training less sensitive to initialization
- BN often reduces the need for Dropout 

---

## 9- Reference
Paper: **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**  
Authors: Sergey Ioffe & Christian Szegedy (2015)  
📄 [PDF Link](https://arxiv.org/pdf/1502.03167)
