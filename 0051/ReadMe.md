# PI-Latent-NO: Physics-Informed Latent Neural Operator

An end-to-end, computationally efficient neural operator framework designed for real-time predictions of complex, high-dimensional physical systems governed by Partial Differential Equations (PDEs).

---

## The Core Innovation

Simulating complex physical systems (like fluid dynamics, climate patterns, or structural stress) using standard numerical solvers is computationally expensive. While AI-based surrogate models like **DeepONet** can accelerate these simulations, they face a harsh trade-off: achieving high accuracy typically requires massive, overparameterized networks that suffer from extreme memory overhead when calculating physics laws (PDE gradients).

**PI-Latent-NO** solves this by introducing **Inherent Separability**. By coupling two DeepONets in a single-shot, end-to-end training pipeline, it decouples the spatio-temporal axes. This shifts the computational complexity from **quadratic scaling $O(N^2)$** to **linear scaling $O(N)$**, allowing physics-informed models to scale to massive grid sizes without running out of GPU memory.

---

## Architecture Design

The framework splits the classic operator learning task into two distinct, coupled steps trained simultaneously:


1. **Latent-DeepONet (Time Encoding):**
   * **Branch Net:** Encodes the input configurations ($\xi$) (e.g., source terms or initial conditions).
   * **Trunk Net:** Evaluates **only the temporal coordinate ($t$)**.
   * **Output:** Predicts a compressed, low-dimensional latent trajectory ($\hat{z}$) representing the system's dynamics.

2. **Reconstruction-DeepONet (Space Encoding & Output):**
   * **Branch Net:** Takes the predicted latent representation ($\hat{z}$) from the first network.
   * **Trunk Net:** Evaluates **only the spatial coordinates ($x$)**.
   * **Output:** Reconstructs the full, high-fidelity physical solution field ($\hat{u}$) back into the original space.

---

## Key Advantages over Vanilla Methods

| Feature | Vanilla PI-DeepONet | PI-Latent-NO (This Work) |
| :--- | :--- | :--- |
| **Scaling Complexity** | **Quadratic $O(N_t \times N_x)$** | **Linear $O(N_t + N_x)$** |
| **Data Requirements** | Pure physics requires no data but is slow; data-driven needs thousands of samples. | **Highly Data-Efficient** (Achieves high accuracy with as few as 50–200 samples). |
| **Memory Footprint** | Scales exponentially with grid resolution during backpropagation. | **Nearly Constant** memory usage across fine grid discretizations. |
| **AD Mode** | Reverse-mode Automatic Differentiation (prone to dimension mismatches). | **Forward-mode Automatic Differentiation** (highly efficient gradient calculation). |

---

## Loss Function Formulation

The model parameters ($\theta$) are optimized end-to-end using a hybrid objective function:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{data-driven}}(\theta) + \mathcal{L}_{\text{physics-informed}}(\theta)$$

* **Data-Driven Loss:** Minimizes the Mean Squared Error (MSE) between the predicted latent states ($\hat{z}$) and the ground truth reduction profiles (obtained via PCA or a Pretrained Autoencoder), as well as the final reconstructed output ($\hat{u}$).
* **Physics-Informed Loss:** Enforces structural compliance by evaluating the PDE residuals, initial conditions ($IC$), and boundary conditions ($BC$) across sampled collocation points.

---

## Benchmark Performance Summary

The framework was rigorously validated on multiple high-dimensional parametric PDE systems (e.g., 1D Diffusion-Reaction Dynamics, 1D Burgers' Transport Dynamics) using NVIDIA A100 GPUs:

* **Discretization Scaling:** When spatial grid points were scaled up from $8$ to $4096$ points, the execution runtime per iteration and VRAM usage for **PI-Latent-NO remained completely flat (constant)**, whereas the vanilla physics-informed models scaled quadratically, hitting hardware limitations.
* **Accuracy:** Maintained a competitive test Mean Squared Error (MSE) in the range of $10^{-4}$ to $10^{-5}$ while using significantly less computational overhead.

---

## Authors & Citation
* **Sharmila Karumuri** (Johns Hopkins University)
* **Lori Graham-Brady** (Johns Hopkins University)
* **Somdatta Goswami** (Johns Hopkins University)

```bibtex
@article{karumuri2026pilatentno,
  title={Physics-Informed Latent Neural Operator for Real-time Predictions of Complex Physical Systems},
  author={Karumuri, Sharmila and Graham-Brady, Lori and Goswami, Somdatta},
  journal={arXiv preprint},
  year={2026}
}