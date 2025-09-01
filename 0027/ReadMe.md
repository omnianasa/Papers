# Continual Learning with Deep Generative Replay

## Overview

This README explains the paper **"Continual Learning with Deep Generative Replay"**. The paper introduces a method to solve the problem of *catastrophic forgetting* in neural networks using a framework called **Deep Generative Replay (DGR)**.

## The Problem: Catastrophic Forgetting

In continual learning, neural networks forget old tasks when learning new ones. This is called **catastrophic forgetting**.

**Example:**
If a network learns to recognize digits, and then it learns to recognize letters, it may completely forget the digit recognition task.

## The Proposed Solution: Deep Generative Replay

The authors propose **Deep Generative Replay (DGR)**. The idea is to use two models:

1.  **Generator:** creates fake samples that look like past data.
2.  **Solver:** learns to solve tasks using real and generated samples.

Together, they form what is called a **Scholar Model**.
- The generator remembers past inputs by creating similar fake data.
- The solver continues to learn without forgetting old tasks.

## Mathematical Formulation

The loss for training the solver is a mix between learning from real
data of the current task and replayed data from old tasks:

``` math
L_{train}(\theta_i) = r \; \mathbb{E}_{(x,y)\sim D_i}[L(S(x;\theta_i), y)] + (1-r) \; \mathbb{E}_{x' \sim G_{i-1}}[L(S(x';\theta_i), S(x';\theta_{i-1}))]
```

The test loss evaluates on both current and past tasks:

``` math
L_{test}(\theta_i) = r \; \mathbb{E}_{(x,y)\sim D_i}[L(S(x;\theta_i), y)] + (1-r) \; \mathbb{E}_{(x,y)\sim D_{past}}[L(S(x;\theta_i), y)]
```

## Experiments

-   **Independent tasks:** Shuffling pixels of MNIST images. Generative replay preserved old task performance.
-   **New domains:** Training on MNIST then SVHN (and vice versa). Generative replay retained past knowledge.\
-   **New classes:** Training on subsets of MNIST classes. Generative replay allowed the network to remember old classes while learning new ones.

## Discussion

Compared to other methods like **Elastic Weight Consolidation (EWC)**
and **Learning without Forgetting (LwF):**

-   **EWC:** Protects important weights but requires large networks and fixed architectures.
-   **LwF:** Depends on task relevance and becomes slower as the number of tasks grows.
-   **DGR:** Flexible and effective, but its success depends on the quality of the generator.

## How to Benefit from this Paper

This paper provides scientific insights into catastrophic forgetting and continual learning:

-   It shows how generative models can act as memory systems, similar to the hippocampus in the brain.
-   It demonstrates that networks can learn continuously without storing all past data.
-   It inspires future work in reinforcement learning and lifelong AI systems.
