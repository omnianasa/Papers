# U‑Net: Convolutional Networks for Biomedical Image Segmentation

## Overview  
U‑Net is a convolutional neural network architecture designed for biomedical image segmentation.  
It was proposed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox from the University of Freiburg, Germany.  
([Paper on arXiv](https://arxiv.org/pdf/1505.04597))

The network combines a contracting path to capture context and an expansive path for precise localization.  
It is capable of training with very few annotated samples using extensive data augmentation and a weighted loss to separate touching objects.

---

## Key Features

- **U-shaped architecture** combining global context and fine localization.  
- **No fully connected layers**, only convolutions, up-convolutions, and concatenations.  
- **Overlap-tile strategy** for segmenting arbitrarily large images seamlessly.  
- **Strong data augmentation** using elastic deformations for invariance to shape variations.  
- **Weighted loss** emphasizing borders between touching cells.  
- **Efficient training** with limited data and fast inference (<1s for 512×512 image on GPU).

---

## Architecture

### Contracting Path
- Repeated blocks of two 3×3 convolutions (unpadded) each followed by ReLU, and a 2×2 max pooling with stride 2 for downsampling.  
- The number of feature channels doubles after each downsampling step.

### Expansive Path
- Each step consists of:
  1. **Upsampling** of the feature map.
  2. **2×2 up-convolution** that halves the number of channels.
  3. **Concatenation** with the correspondingly cropped feature map from the contracting path.
  4. **Two 3×3 convolutions**, each followed by a ReLU.

### Final Layer
- A 1×1 convolution maps each 64-component feature vector to the desired number of classes.  
- The network has **23 convolutional layers** in total.

### Tiling Strategy
To enable seamless segmentation, input tiles must be chosen so that all 2×2 max-pooling operations are applied on layers with even x- and y-size

---

## Training Details

- **Optimizer:** Stochastic Gradient Descent (SGD)  
- **Batch size:** often 1 due to memory limits, with momentum = 0.99
- **Loss function:** Softmax + Cross-Entropy with weighted pixels

```math
E = \sum_{x \in \Omega} w(x) \log(p_{\ell(x)}(x))
```

where \( \ell(x) \) is the true label of pixel \( x \).

### Weight Map
To emphasize borders between touching cells, the weight map is defined as:

```math
w(x) = w_c(x) + w_0 \cdot \exp\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)
```

where \( w_c \) balances class frequency, and \( d_1, d_2 \) are distances to the nearest and second nearest cell borders.

### Weight Initialization
Weights are drawn from a Gaussian distribution with standard deviation:

```math
\sqrt{\frac{2}{N}}
```

where \( N \) is the number of incoming nodes for each neuron.

### Data Augmentation
- Random shifts, rotations, and elastic deformations
- Elastic deformation field is generated on a coarse 3×3 grid, with displacements from a Gaussian distribution, interpolated to pixel level.
- Dropout is used at the end of the contracting path.

---

## Experimental Results

### EM Segmentation (ISBI 2012)
- 30 training images (512×512) from electron microscopy stacks.
- Metrics: Warping Error, Rand Error, Pixel Error.

| Metric | U‑Net | Previous Best (Ciresan et al.) |
|---------|-------|--------------------------------|
| Warping Error | 0.0003529 | Higher |
| Rand Error | 0.0382 | Higher |
| Pixel Error | Lower | – |

### Cell Segmentation (ISBI 2015)
- Datasets: **PhC‑U373** (phase contrast) and **DIC‑HeLa** (DIC microscopy).  
- U‑Net outperformed all competitors:

| Dataset | U‑Net IOU | Previous Best |
|----------|------------|----------------|
| PhC‑U373 | 92% | ~83% |
| DIC‑HeLa | 77.5% | ~46% |

---

## Usage

1. Prepare training images and their corresponding segmentation masks.  
2. Define the number of output classes.  
3. Choose input tile size with even dimensions for pooling layers.  
4. Apply strong data augmentation (especially elastic deformations).  
5. Train with SGD + momentum using the weighted loss.  
6. For large images, use the **overlap-tile + mirroring** strategy.  
7. Post-process segmentation maps if needed.

---
